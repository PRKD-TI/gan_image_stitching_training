import re
import random
import numpy as np
import os
import time
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pytorch_msssim import ssim, ms_ssim
import lpips
from torchvision.models import vgg16
from utils.metrics import compute_all_metrics
from gan_structure_deep import DualEncoderUNet_CBAM_SA_Deep, PatchDiscriminator
from tqdm import tqdm
from datetime import datetime, timedelta

# ------------------ losses / helpers ------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights='DEFAULT').features[:16]
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return F.l1_loss(input_vgg, target_vgg)

lpips_fn = lpips.LPIPS(net='alex').to('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ------------------ visualization/comparison ------------------
def comparar_generator_vs_ema(generator, ema_generator, part1, part2, target, writer, step, tag="comparison"):
    generator.eval()
    ema_generator.eval()
    with torch.no_grad():
        fake = generator(part1, part2)
        fake_ema = ema_generator(part1, part2)
        linha = torch.cat([part1, part2, fake, fake_ema, target], dim=3)
        grid = vutils.make_grid(linha, normalize=True, value_range=(0, 1))
        writer.add_image(f"{tag}/gen_vs_ema", grid, global_step=step)
    generator.train()

# ------------------ dataset utilities ------------------
def carregar_amostras_fixas(train_loader, caminho='fixed_samples.pt', device='cuda'):
    if os.path.exists(caminho):
        print(f'[INFO] Carregando amostras fixas de {caminho}')
        fixed_samples = torch.load(caminho)
    else:
        print('[INFO] Selecionando novas amostras fixas...')
        fixed_samples = []
        for i, ((p1, p2), gt) in enumerate(train_loader):
            if i >= 5:
                break
            fixed_samples.append(((p1[0].unsqueeze(0), p2[0].unsqueeze(0)), gt[0].unsqueeze(0)))
        torch.save(fixed_samples, caminho)
        print(f'[INFO] Amostras fixas salvas em {caminho}')
    fixed_samples = [((p1.to(device), p2.to(device)), gt.to(device)) for ((p1, p2), gt) in fixed_samples]
    return fixed_samples

# usa EMA se for tupla (generator, ema)
def salvar_resultados_fixos(generator, fixed_samples, output_dir, step_tag):
    if isinstance(generator, tuple):
        gen_for_eval = generator[1]
    else:
        gen_for_eval = generator
    gen_for_eval.eval()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, ((p1, p2), gt) in enumerate(fixed_samples):
            fake = gen_for_eval(p1, p2)
            linha = torch.cat([p1, p2, fake, gt], dim=3)
            vutils.save_image(linha, os.path.join(output_dir, f"sample_{idx+1}_{step_tag}.png"), normalize=True)
    if not isinstance(generator, tuple):
        gen_for_eval.train()

# ------------------ checkpoint utils ------------------
def carregar_checkpoint_mais_recente(checkpoints_epoch_dir, checkpoints_batch_dir):
    pattern_epoch = re.compile(r"checkpoint_epoch(\d+)\.pt")
    pattern_batch = re.compile(r"checkpoint_epoch(\d+)_batch(\d+)\.pt")
    def extract_epoch_batch(filename):
        match = pattern_batch.match(filename)
        if match:
            return int(match.group(1)), int(match.group(2)), filename
        match = pattern_epoch.match(filename)
        if match:
            return int(match.group(1)), -1, filename
        return None
    all_checkpoints = []
    for fname in os.listdir(checkpoints_epoch_dir):
        result = extract_epoch_batch(fname)
        if result:
            epoch, batch, name = result
            all_checkpoints.append((epoch, batch, os.path.join(checkpoints_epoch_dir, name)))
    for fname in os.listdir(checkpoints_batch_dir):
        result = extract_epoch_batch(fname)
        if result:
            epoch, batch, name = result
            all_checkpoints.append((epoch, batch, os.path.join(checkpoints_batch_dir, name)))
    if not all_checkpoints:
        return None, 0, -1
    all_checkpoints.sort(key=lambda x: (x[0], x[1]))
    epoch, batch, path = all_checkpoints[-1]
    return path, epoch, batch

# ------------------ adaptive generator steps ------------------
def get_dynamic_gen_steps(loss_D, max_steps=5):
    if loss_D > 1.0:
        return max_steps
    elif loss_D > 0.5:
        return 3
    elif loss_D > 0.2:
        return 2
    return 1

# ------------------ main train function ------------------
def train(
    generator, discriminator, dataloader, device, epochs,
    save_every, checkpoint_dir, checkpoint_batch_dir,
    tensorboard_dir, metrics, lr_g=2e-4, lr_d=2e-4,
    lr_min=1e-6, gen_steps_mode='adaptive', max_gen_steps=5,
    vgg_weight=0.5, fixeSampleTime=5,  # minutos
    fixed_samples_source='../fixed_samples.pt',
    fixed_samples_dest='./fixed_samples'
):
    set_seed(42)
    fixed_samples = carregar_amostras_fixas(dataloader, caminho=fixed_samples_source, device=device)
    last_fixed_sample_time = datetime.now()

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_batch_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()
    criterion_VGG = VGGPerceptualLoss().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=20, eta_min=lr_min)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=20, eta_min=lr_min)

    # EMA do generator (cópia suave)
    ema_generator = DualEncoderUNet_CBAM_SA_Deep().to(device)
    ema_generator.load_state_dict(generator.state_dict())
    ema_generator.eval()
    ema_beta = 0.999

    checkpoint_path, start_epoch, start_batch = carregar_checkpoint_mais_recente(checkpoint_dir, checkpoint_batch_dir)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        if 'theta_ema_state_dict' in checkpoint:
            ema_generator.load_state_dict(checkpoint['theta_ema_state_dict'])
        if start_batch == -1:
            start_epoch += 1
            start_batch = 0
    else:
        start_epoch = 0
        start_batch = 0

    avg_loss_D = None
    last_checkpoint_time = time.time()


    for epoch in range(start_epoch, epochs):
        total_loss_G = 0.0
        total_loss_D = 0.0
        total_loss_VGG = 0.0
        total_metrics = {k: 0.0 for k in ['PSNR', 'SSIM', 'MS-SSIM', 'LPIPS', 'L1', 'CPU_Usage_%', 'RAM_Usage_MB', 'GPU_Usage_%', 'GPU_Memory_MB']}
        count = 0
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, ((part1, part2), target) in pbar:
            if epoch == start_epoch and i < start_batch:
                continue

            part1, part2, target = part1.to(device), part2.to(device), target.to(device)
            real_input = torch.cat([part1, part2, target], dim=1)
            fake = generator(part1, part2)
            fake_input = torch.cat([part1, part2, fake.detach()], dim=1)

            # ruído decaindo nas entradas do discriminador
            noise_std = 0.1 * max(0.0, 1.0 - epoch / epochs)
            real_noisy = real_input + torch.randn_like(real_input) * noise_std
            fake_noisy = fake_input + torch.randn_like(fake_input) * noise_std

            # Train Discriminator com label smoothing
            optimizer_D.zero_grad()
            pred_real = discriminator(real_noisy)
            pred_fake = discriminator(fake_noisy)
            real_labels = torch.full_like(pred_real, 0.9, device=pred_real.device)
            loss_D_real = criterion_GAN(pred_real, real_labels)
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()

            # média móvel para gen_steps
            if avg_loss_D is None:
                avg_loss_D = loss_D.item()
            else:
                avg_loss_D = 0.9 * avg_loss_D + 0.1 * loss_D.item()
            gen_steps = get_dynamic_gen_steps(avg_loss_D, max_gen_steps) if gen_steps_mode == 'adaptive' else gen_steps_mode

            # Train Generator
            for _ in range(gen_steps):
                fake = generator(part1, part2)
                fake_input = torch.cat([part1, part2, fake], dim=1)
                optimizer_G.zero_grad()
                pred_fake = discriminator(fake_input)
                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterion_L1(fake, target)
                loss_G_VGG = criterion_VGG(fake, target)
                loss_G = 8.0 * loss_G_GAN + 2.0 * loss_G_L1 + vgg_weight * loss_G_VGG
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
                # EMA update
                for ema_param, param in zip(ema_generator.parameters(), generator.parameters()):
                    ema_param.data.mul_(ema_beta).add_(param.data * (1 - ema_beta))

            pbar.set_postfix({
                "loss_G": f"{loss_G.item():.4f}",
                "loss_D": f"{loss_D.item():.4f}"
            })

            step = epoch * len(dataloader) + i

            lpips_val = lpips_fn(fake, target)
            # agrega para escalar; ajustar caso a forma seja diferente (ex: [B,1,1,1])
            lpips_scalar = lpips_val.mean().item()

            writer.add_scalar("Loss/GAN", loss_G_GAN.item(), step)
            writer.add_scalar("Loss/L1", loss_G_L1.item(), step)
            writer.add_scalar("Loss/VGG_component", loss_G_VGG.item(), step)
            writer.add_scalar("Loss/Generator", loss_G.item(), step)
            writer.add_scalar("Loss/Discriminator", loss_D.item(), step)
            writer.add_scalar("Loss/LPIPS", lpips_scalar, step)
            writer.add_scalar("LR/Generator", optimizer_G.param_groups[0]['lr'], step)
            writer.add_scalar("LR/Discriminator", optimizer_D.param_groups[0]['lr'], step)

            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            total_loss_VGG += loss_G_VGG.item()
            count += 1

            with torch.no_grad():
                eval_metrics = compute_all_metrics(fake, target, part1, part2, writer, step)
                for k, v in eval_metrics.items():
                    if v is not None:
                        writer.add_scalar(f"Epoch/Metrics/{k}", v, step)
                        total_metrics[k] += v

            comparar_generator_vs_ema(generator, ema_generator, part1, part2, target, writer, step)

            if datetime.now() - last_fixed_sample_time >= timedelta(minutes=fixeSampleTime):
                salvar_resultados_fixos((generator, ema_generator), fixed_samples, output_dir=fixed_samples_dest, step_tag=f"{epoch}_{step}")
                last_fixed_sample_time = datetime.now()

            # Checkpoint a cada
            if time.time() - last_checkpoint_time > save_every:
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, os.path.join(checkpoint_batch_dir, f'checkpoint_epoch{epoch}_batch{i}.pt'))
                last_checkpoint_time = time.time()

        writer.add_scalar("Epoch/Loss_Generator", total_loss_G / count, epoch)
        writer.add_scalar("Epoch/Loss_Discriminator", total_loss_D / count, epoch)
        writer.add_scalar("Epoch/Loss_VGG", total_loss_VGG / count, epoch)
        for k, v in total_metrics.items():
            writer.add_scalar(f"Epoch/Metrics/{k}", v / count, epoch)

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'theta_ema_state_dict': ema_generator.state_dict()
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

        scheduler_G.step()
        scheduler_D.step()

    writer.close()
