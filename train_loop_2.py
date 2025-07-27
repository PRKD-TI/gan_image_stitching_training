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
from metrics import compute_all_metrics  # Importa a função de métricas do arquivo metrics.py
# from gan_structure import DualEncoderUNet_CBAM_SA_Small, PatchDiscriminator  # Importa os modelos do arquivo gan-structure.py
from gan_structure_2 import DualEncoderUNet_CBAM_SA_Small, PatchDiscriminator  # A partir da época 9

from tqdm import tqdm

debug = 0
# from generator import DualEncoderUNet_CBAM_SA_Small  # novo gerador com 2 encoders, CBAM e SelfAttention
# from discriminator import PatchDiscriminator  # ou caminho equivalente

# LPIPS usa um modelo de rede para comparação perceptual
lpips_fn = lpips.LPIPS(net='alex')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    
    # Envia pro device
    fixed_samples = [((p1.to(device), p2.to(device)), gt.to(device)) for ((p1, p2), gt) in fixed_samples]
    return fixed_samples

def salvar_resultados_fixos(generator, fixed_samples, output_dir, step_tag):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, ((p1, p2), gt) in enumerate(fixed_samples):
            fake = generator(p1, p2)

            # Concatena horizontalmente: parte1 | parte2 | fake | groundtruth
            linha = torch.cat([p1, p2, fake, gt], dim=3)

            vutils.save_image(linha, os.path.join(output_dir, f"sample_{idx+1}_{step_tag}.png"), normalize=True)
    generator.train()


def carregar_checkpoint_mais_recente(checkpoints_epoch_dir, checkpoints_batch_dir):
    """
    Retorna o caminho, epoch e batch do checkpoint mais recente (epoch ou batch).
    
    Returns:
        (str caminho, int epoch, int batch)
        batch = -1 significa checkpoint de final de epoch
    """
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
        return None, 0, -1  # Nada encontrado

    # Ordena por (epoch, batch)
    all_checkpoints.sort(key=lambda x: (x[0], x[1]))
    epoch, batch, path = all_checkpoints[-1]
    return path, epoch, batch

from torchvision.models import vgg16
import torch.nn.functional as F

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16]  # até conv3_3
        self.vgg = vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_vgg = self.vgg(input)
        target_vgg = self.vgg(target)
        return F.l1_loss(input_vgg, target_vgg)


def train(
    generator, discriminator, dataloader, device, epochs,
    save_every, checkpoint_dir, checkpoint_batch_dir,
    tensorboard_dir, metrics, lr_g=2e-4, lr_d=2e-4,
    lr_min=1e-6, gen_steps_per_batch=1,
    fixeSampleTime=5  # minutos
):

    # Carregar amostras fixas para comparação entre os batches
    from datetime import datetime, timedelta
    set_seed(42)
    fixed_samples = carregar_amostras_fixas(dataloader, caminho='fixed_samples.pt', device=device)
    last_fixed_sample_time = datetime.now()


    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_batch_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=epochs, eta_min=lr_min)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=epochs, eta_min=lr_min)

    start_epoch = 0
    # checkpoint_path = carregar_checkpoint_mais_recente(checkpoint_dir, checkpoint_batch_dir)
    checkpoint_path, start_epoch, start_batch = carregar_checkpoint_mais_recente("checkpoints_epoch", "checkpoints_batch")

    if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)

            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            # ... quaisquer outras coisas

            # ⚠️ Ajuste aqui: para continuar da mesma epoch
            # start_epoch permanece igual se batch != -1
            if start_batch == -1:
                start_epoch += 1
                start_batch = 0
    else:
        start_epoch = 0
        start_batch = 0

    last_checkpoint_time = time.time()

    for epoch in range(start_epoch, epochs):
        total_loss_G = 0.0
        total_loss_D = 0.0
        total_metrics = {
            k: 0.0 for k in [
                'PSNR',
                'SSIM',
                'MS-SSIM',
                'LPIPS',
                'L1',
                'CPU_Usage_%',
                'RAM_Usage_MB',
                'GPU_Usage_%',
                'GPU_Memory_MB',
            ]
        }
        count = 0

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}")
        for i, ((part1, part2), target) in pbar:

            # Se estiver retomando a primeira epoch, pule os batches anteriores
            if epoch == start_epoch and i < start_batch:
                continue
            
            part1 = part1.to(device)
            part2 = part2.to(device)
            target = target.to(device)

            real_input = torch.cat([part1, part2, target], dim=1)
            fake = generator(part1, part2)
            fake_input = torch.cat([part1, part2, fake.detach()], dim=1)

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(real_input)
            pred_fake = discriminator(fake_input)

            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # Train Generator
            for _ in range(gen_steps_per_batch):
                fake = generator(part1, part2)
                fake_input = torch.cat([part1, part2, fake], dim=1)
                optimizer_G.zero_grad()
                pred_fake = discriminator(fake_input)
                loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
                loss_G_L1 = criterion_L1(fake, target)
                loss_G = 8.0 * loss_G_GAN + 2.0 * loss_G_L1
                loss_G.backward()
                optimizer_G.step()

            pbar.set_postfix({
                "loss_G": f"{loss_G.item():.4f}",
                "loss_D": f"{loss_D.item():.4f}"
            })

            step = epoch * len(dataloader) + i
            writer.add_scalar("Loss/Generator", loss_G.item(), step)
            writer.add_scalar("Loss/Discriminator", loss_D.item(), step)
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
            count += 1

            with torch.no_grad():
                eval_metrics = compute_all_metrics(fake, target, part1, part2, writer, step)
                for k, v in eval_metrics.items():
                    if v is not None:
                        writer.add_scalar(f"Metrics/{k}", v, epoch * len(dataloader) + i)
                        total_metrics[k] += v
                        

            if datetime.now() - last_fixed_sample_time >= timedelta(minutes=fixeSampleTime):
                salvar_resultados_fixos(generator, fixed_samples, output_dir='./fixed_samples', step_tag=f"{epoch}_{epoch * len(dataloader) + i}")
                last_fixed_sample_time = datetime.now()

            # Checkpoint a cada 10 minutos
            if time.time() - last_checkpoint_time > 600:
                torch.save({
                    'epoch': epoch,
                    'batch': i,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                }, os.path.join(checkpoint_batch_dir, f'checkpoint_epoch{epoch}_batch{i}.pt'))
                last_checkpoint_time = time.time()

        # Salvar médias da época no TensorBoard
        writer.add_scalar("Epoch/Loss_Generator", total_loss_G / count, epoch)
        writer.add_scalar("Epoch/Loss_Discriminator", total_loss_D / count, epoch)

        for k, v in total_metrics.items():
            if v is not None:
                writer.add_scalar(f"Metrics/{k}", v/count, epoch)

        # Fim da época: salvar checkpoint principal
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
        }, os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

        scheduler_G.step()
        scheduler_D.step()

    writer.close()

