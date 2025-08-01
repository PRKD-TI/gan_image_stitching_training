import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch.utils.data import DataLoader

# Dataset e dataloader
from utils.dataset import ImageStitchingDatasetFiles
from utils.extrair_zip_train_dir import descompactar_zip_com_progresso

# Treinamento e checkpoint
from train_loop_3 import train

from gan_structure_deep import DualEncoderUNet_CBAM_SA_Deep, PatchDiscriminator




def main():
    debug = 0

    # Descompactar dataset se necessário
    filename = "dataset_48_32.zip"
    descompactar_zip_com_progresso(f"../{filename}", "../train")

    # Dataset e DataLoader
    dataset = ImageStitchingDatasetFiles("../train", use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar modelos
    generator = DualEncoderUNet_CBAM_SA_Deep().to(device)
    discriminator = PatchDiscriminator().to(device)

    # Treinamento
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=100,
        save_every=3600,  # segundos
        checkpoint_dir="../checkpoints_epoch_local",
        checkpoint_batch_dir="../checkpoints_batch_local",
        tensorboard_dir="../logs/32x48",
        metrics=True,
        gen_steps_per_batch=20,
        fixeSampleTime=5,  # minutos
        fixed_samples_source="../fixed_samples.pt",
        fixed_samples_dest="./fixed_samples",
    )


if __name__ == "__main__":
    main()
