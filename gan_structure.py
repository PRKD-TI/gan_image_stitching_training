import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM (Convolutional Block Attention Module)
# Aplica atenção canal + espacial separadamente
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Atenção no canal
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x_out = x * self.sigmoid_channel(avg_out + max_out)  # salva num novo tensor para não perder o input original

        # Atenção espacial
        avg_out = torch.mean(x_out, dim=1, keepdim=True)
        max_out, _ = torch.max(x_out, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)  # [N, 2, H, W]
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(spatial_attention))  # [N, 1, H, W]

        # Multiplica o resultado da atenção espacial pelo tensor original (com canais corretos)
        out = x_out * spatial_attention

        return out

# Self-Attention simples no bottleneck
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)  # matriz de atenção
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

# Bloco de codificação padrão
class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Bloco de decodificação com upsample + concat + convoluções
class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out + ch_skip, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# Rede UNet com dois encoders, CBAM e self-attention no bottleneck
class DualEncoderUNet_CBAM_SA_Small(nn.Module):
    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()

        # Dois encoders independentes (parte1 e parte2)
        self.enc1_1 = EncoderBlock(in_channels, base_ch)
        self.enc2_1 = EncoderBlock(base_ch, base_ch * 2)

        self.enc1_2 = EncoderBlock(in_channels, base_ch)
        self.enc2_2 = EncoderBlock(base_ch, base_ch * 2)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck com self-attention
        self.bottleneck = EncoderBlock(base_ch * 4, base_ch * 4)
        self.attn = SelfAttention(base_ch * 4)

        # CBAM nas skip connections
        self.cbam2 = CBAM(base_ch * 4)
        self.cbam1 = CBAM(base_ch * 2)

        # Decoder com três parâmetros por bloco
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)  # 128, 128, 64
        self.dec1 = DecoderBlock(base_ch * 2, base_ch * 2, base_ch)      # 64, 64, 32

        self.final = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x1, x2):
        # Encoder para parte1
        e1_1 = self.enc1_1(x1)
        e2_1 = self.enc2_1(self.pool(e1_1))

        # Encoder para parte2
        e1_2 = self.enc1_2(x2)
        e2_2 = self.enc2_2(self.pool(e1_2))

        # Garantir que as features estejam com mesmas dimensões (por segurança)
        if e1_1.shape[2:] != e1_2.shape[2:]:
            e1_2 = F.interpolate(e1_2, size=e1_1.shape[2:], mode='bilinear', align_corners=False)
        if e2_1.shape[2:] != e2_2.shape[2:]:
            e2_2 = F.interpolate(e2_2, size=e2_1.shape[2:], mode='bilinear', align_corners=False)

        # Bottleneck: concatenação + atenção
        b = self.bottleneck(torch.cat([self.pool(e2_1), self.pool(e2_2)], dim=1))
        b = self.attn(b)

        # Decoder com CBAM nas skip connections
        d2 = self.dec2(b, self.cbam2(torch.cat([e2_1, e2_2], dim=1)))
        d1 = self.dec1(d2, self.cbam1(torch.cat([e1_1, e1_2], dim=1)))

        return torch.sigmoid(self.final(d1))  # saída com valo
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=9):  # Agora espera parte1 (3) + parte2 (3) + target/fake (3)
        super(PatchDiscriminator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, normalize=False),  # in_channels = 9
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)  # saída do PatchGAN (mapa de decisão)
        )

    def forward(self, img):
        return self.model(img)