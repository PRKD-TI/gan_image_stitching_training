import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import lpips

# LPIPS usa um modelo de rede para comparação perceptual
lpips_fn = lpips.LPIPS(net='alex')


def compute_all_metrics(fake, target):
    """
    Calcula múltiplas métricas de avaliação entre a imagem gerada (fake) e o alvo (target).

    Parâmetros:
        fake (Tensor): Imagem gerada (N, C, H, W) [0,1] ou [-1,1]
        target (Tensor): Ground truth (N, C, H, W)

    Retorna:
        dict: {'PSNR': ..., 'SSIM': ..., 'MS-SSIM': ..., 'LPIPS': ..., 'L1': ...}
    """
    # Garantir que estão no mesmo range [0, 1]
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

    # MS-SSIM
    ms_ssim_val = ms_ssim(fake, target, data_range=1.0, size_average=True)
    metrics['MS-SSIM'] = ms_ssim_val.item()

    # LPIPS (usa batch size 1 por vez)
    with torch.no_grad():
        lpips_vals = []
        for f, t in zip(fake, target):
            lp = lpips_fn(f.unsqueeze(0), t.unsqueeze(0))
            lpips_vals.append(lp.item())
        metrics['LPIPS'] = sum(lpips_vals) / len(lpips_vals)

    # L1 Loss (MAE)
    l1 = F.l1_loss(fake, target).item()
    metrics['L1'] = l1

    return metrics
