import torch
import torch.nn.functional as F
from torch.autograd import grad
from pytorch_msssim import ssim, ms_ssim
import lpips

# LPIPS usa um modelo de rede para comparação perceptual
lpips_fn = lpips.LPIPS(net='alex')


from torchvision.utils import make_grid

def compute_all_metrics(fake, target, part1=None, part2=None, writer=None, step=None):
    """
    Calcula métricas e registra imagens no TensorBoard (opcionalmente).

    Parâmetros:
        fake (Tensor): Imagem gerada (N, C, H, W) [0,1] ou [-1,1]
        target (Tensor): Ground truth (N, C, H, W)
        part1, part2 (Tensor): Entradas (N, C, H, W) opcionais
        writer (SummaryWriter): TensorBoard writer opcional
        step (int): Etapa global (epoch * len(loader) + i)

    Retorna:
        dict com métricas
    """
    # Garantir que estão no mesmo range [0, 1]
    if fake.min() < 0:
        fake = (fake + 1) / 2
        target = (target + 1) / 2
        if part1 is not None:
            part1 = (part1 + 1) / 2
        if part2 is not None:
            part2 = (part2 + 1) / 2

    metrics = {}

    # PSNR
    mse = F.mse_loss(fake, target)
    psnr = -10 * torch.log10(mse + 1e-8)
    metrics['PSNR'] = psnr.item()

    # SSIM
    ssim_val = ssim(fake, target, data_range=1.0, size_average=True)
    metrics['SSIM'] = ssim_val.item()

    # MS-SSIM
    min_dim = min(fake.shape[-2], fake.shape[-1])
    if min_dim >= 160:
        try:
            ms_ssim_val = ms_ssim(fake, target, data_range=1.0, size_average=True)
            metrics['MS-SSIM'] = ms_ssim_val.item()
        except Exception as e:
            print(f"[AVISO] Falha ao calcular MS-SSIM: {e}")
            metrics['MS-SSIM'] = None
    else:
        metrics['MS-SSIM'] = None  # ou "Pequena" para indicar o motivo

    # LPIPS
    with torch.no_grad():
        lpips_vals = []
        for f, t in zip(fake, target):
            lp = lpips_fn(f.unsqueeze(0), t.unsqueeze(0))
            lpips_vals.append(lp.item())
        metrics['LPIPS'] = sum(lpips_vals) / len(lpips_vals)

    # L1
    l1 = F.l1_loss(fake, target).item()
    metrics['L1'] = l1

    # Adiciona imagens ao TensorBoard
    if writer is not None and step is not None:
        def log_image(tag, tensor):
            grid = make_grid(tensor[:4], nrow=4, normalize=True, value_range=(0, 1))
            writer.add_image(tag, grid, global_step=step)

        log_image("Image/Part1", part1)
        log_image("Image/Part2", part2)
        log_image("Image/Fake", fake)
        log_image("Image/Target", target)

    return metrics

def compute_all_metrics_old(fake, target):
    """
    Calcula múltiplas métricas de avaliação entre a imagem gerada (fake) e o alvo (target).

    Parâmetros:
        fake (Tensor): Imagem gerada (N, C, H, W) [0,1] ou [-1,1]
        target (Tensor): Ground truth (N, C, H, W)

    Retorna:
        dict: {'PSNR': ..., 'SSIM': ..., 'MS-SSIM': ..., 'LPIPS': ..., 'L1': ...}
    """
    if fake.min() < 0:
        fake = (fake + 1) / 2
        target = (target + 1) / 2

    metrics = {}

    # PSNR
    mse = F.mse_loss(fake, target)
    psnr = -10 * torch.log10(mse + 1e-8)
    metrics['PSNR'] = psnr.item()

    # SSIM
    ssim_val = ssim(fake, target, data_range=1.0, size_average=True)
    metrics['SSIM'] = ssim_val.item()

    # MS-SSIM (apenas se imagem for grande o suficiente)
    min_dim = min(fake.shape[-2], fake.shape[-1])
    if min_dim >= 160:
        try:
            ms_ssim_val = ms_ssim(fake, target, data_range=1.0, size_average=True)
            metrics['MS-SSIM'] = ms_ssim_val.item()
        except Exception as e:
            print(f"[AVISO] Falha ao calcular MS-SSIM: {e}")
            metrics['MS-SSIM'] = None
    else:
        metrics['MS-SSIM'] = None  # ou "Pequena" para indicar o motivo

    # LPIPS
    with torch.no_grad():
        lpips_vals = []
        for f, t in zip(fake, target):
            lp = lpips_fn(f.unsqueeze(0), t.unsqueeze(0))
            lpips_vals.append(lp.item())
        metrics['LPIPS'] = sum(lpips_vals) / len(lpips_vals)

    # L1
    metrics['L1'] = F.l1_loss(fake, target).item()

    return metrics





def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    Calcula o gradient penalty para WGAN-GP.

    Parâmetros:
        D (nn.Module): Discriminador
        real_samples (Tensor): Amostras reais (N, C, H, W)
        fake_samples (Tensor): Amostras geradas (N, C, H, W)
        device (torch.device): Dispositivo (CPU/GPU)

    Retorna:
        Tensor: Gradient penalty escalar
    """
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)

    # Interpolação entre real e fake
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Avaliação do discriminador
    d_interpolates = D(interpolates)

    # Criar tensor para grad_outputs com mesmo shape de d_interpolates
    fake = torch.ones_like(d_interpolates, device=device)

    # Gradiente da saída do discriminador em relação à interpolação
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty
