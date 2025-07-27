import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

import torch
from torch.utils.data import DataLoader

# Dataset e dataloader
from utils.dataset import ImageStitchingDatasetFiles
from extrair_zip_train_dir import descompactar_zip_com_progresso

# Treinamento e checkpoint
from train_loop import train

# Modelos
import gan_structure
from gan_structure import SelfAttention, CBAM  # Importa os módulos necessários
from gan_structure import DualEncoderUNet_CBAM_SA_Small, PatchDiscriminator




def main():
    debug = 0

    # Descompactar dataset se necessário
    filename = "dataset_48_32.zip"
    descompactar_zip_com_progresso(f"./{filename}", "./train")

    # Dataset e DataLoader
    dataset = ImageStitchingDatasetFiles("./train", use_gradiente=False)
    dataloader = DataLoader(
        dataset,
        batch_size=256,
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
        gen_steps_per_batch=20
    )


if __name__ == "__main__":
    main()
