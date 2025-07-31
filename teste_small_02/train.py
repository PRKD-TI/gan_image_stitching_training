import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch.utils.data import DataLoader

# Dataset e dataloader
from dataset import ImageStitchingDatasetFiles
from utils_gan.extrair_zip_train_dir import descompactar_zip_com_progresso

# Treinamento e checkpoint
from train_loop_2 import train

# Modelos
import gan_structure
# from gan_structure import DualEncoderUNet_CBAM_SA_Small, PatchDiscriminator
from gan_structure_2 import DualEncoderUNet_CBAM_SA_Small, PatchDiscriminator # A partir da época 9




def main():
    debug = 0

    # Descompactar dataset se necessário
    filename = "dataset_48_32.zip"
    descompactar_zip_com_progresso(f"./{filename}", "./train")

    # Dataset e DataLoader
    dataset = ImageStitchingDatasetFiles("./train", use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True
    )

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar modelos
    generator = DualEncoderUNet_CBAM_SA_Small().to(device)
    discriminator = PatchDiscriminator().to(device)

    # Treinamento
    train(
        generator=generator,
        discriminator=discriminator,
        dataloader=dataloader,
        device=device,
        epochs=100,
        save_every=600,  # segundos
        checkpoint_dir="./checkpoints_epoch",
        checkpoint_batch_dir="./checkpoints_batch",
        tensorboard_dir="./logs/32x48",
        metrics=True,
        gen_steps_per_batch=20,
        fixeSampleTime=5  # minutos
    )


if __name__ == "__main__":
    main()
