# Novo `train.py` com configs atualizadas
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import os
import time
from model import DualEncoderUNet_CBAM_SA_Deep, PatchDiscriminator
from train_loop import train
from dataset import ImageStitchingDataset

# Diretórios
log_dir = './runs/gan_experiment'
os.makedirs(log_dir, exist_ok=True)

# TensorBoard
writer = SummaryWriter(log_dir=log_dir)

# Hiperparâmetros
epochs = 100
batch_size = 64
learning_rate = 2e-4
image_size = (64, 96)

# Dataset e DataLoader
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])

train_dataset = ImageStitchingDataset('./train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Modelos
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
G = DualEncoderUNet_CBAM_SA_Deep().to(device)
D = PatchDiscriminator().to(device)

# Otimizadores e agendadores
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))

scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=20, eta_min=1e-6)
scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=20, eta_min=1e-6)

# Treinamento
train(
    generator=G,
    discriminator=D,
    dataloader=train_loader,
    optimizer_G=optimizer_G,
    optimizer_D=optimizer_D,
    scheduler_G=scheduler_G,
    scheduler_D=scheduler_D,
    device=device,
    writer=writer,
    epochs=epochs,
    start_epoch=0,
    save_every=1,
    sample_dir='./samples',
    checkpoint_dir='./checkpoints',
    gen_steps_mode='adaptive',  # novo modo dinâmico
    vgg_weight=0.5
)

writer.close()
