import torch
import torch.nn as nn
import torch.nn.functional as F

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
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x_out = x * self.sigmoid_channel(avg_out + max_out)

        avg_out = torch.mean(x_out, dim=1, keepdim=True)
        max_out, _ = torch.max(x_out, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid_spatial(self.conv_spatial(spatial_attention))

        out = x_out * spatial_attention
        return out

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
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x

class ResidualBlock(nn.Module):  # utilizado a partir da época 9
    def __init__(self, ch):  # utilizado a partir da época 9
        super().__init__()  # utilizado a partir da época 9
        self.block = nn.Sequential(  # utilizado a partir da época 9
            nn.Conv2d(ch, ch, 3, padding=1),  # utilizado a partir da época 9
            nn.BatchNorm2d(ch),  # utilizado a partir da época 9
            nn.ReLU(inplace=True),  # utilizado a partir da época 9
            nn.Conv2d(ch, ch, 3, padding=1),  # utilizado a partir da época 9
            nn.BatchNorm2d(ch)  # utilizado a partir da época 9
        )  # utilizado a partir da época 9
        self.relu = nn.ReLU(inplace=True)  # utilizado a partir da época 9

    def forward(self, x):  # utilizado a partir da época 9
        return self.relu(self.block(x) + x)  # utilizado a partir da época 9

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
            ResidualBlock(out_ch)  # utilizado a partir da época 9
        )

    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out + ch_skip, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            CBAM(ch_out),  # utilizado a partir da época 9
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            ResidualBlock(ch_out)  # utilizado a partir da época 9
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class DualEncoderUNet_CBAM_SA_Deep(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()

        # Encoder parte1
        self.enc1_1 = EncoderBlock(in_channels, base_ch)
        self.enc2_1 = EncoderBlock(base_ch, base_ch * 2)
        self.enc3_1 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4_1 = EncoderBlock(base_ch * 4, base_ch * 8)

        # Encoder parte2
        self.enc1_2 = EncoderBlock(in_channels, base_ch)
        self.enc2_2 = EncoderBlock(base_ch, base_ch * 2)
        self.enc3_2 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4_2 = EncoderBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck com atenção e resblocks
        self.bottleneck = nn.Sequential(
            EncoderBlock(base_ch * 16, base_ch * 16),
            SelfAttention(base_ch * 16),
            ResidualBlock(base_ch * 16),
            ResidualBlock(base_ch * 16)
        )

        # CBAM nas features de skip connection 
        self.cbam4 = CBAM(base_ch * 16)
        self.cbam3 = CBAM(base_ch * 8)
        self.cbam2 = CBAM(base_ch * 4)
        self.cbam1 = CBAM(base_ch * 2)

        # Decoders
        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 16, base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8, base_ch * 8, base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4, base_ch * 4, base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2, base_ch * 2, base_ch)

        self.final = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, x1, x2):
        # Encode parte1
        e1_1 = self.enc1_1(x1)
        e2_1 = self.enc2_1(self.pool(e1_1))
        e3_1 = self.enc3_1(self.pool(e2_1))
        e4_1 = self.enc4_1(self.pool(e3_1))

        # Encode parte2
        e1_2 = self.enc1_2(x2)
        e2_2 = self.enc2_2(self.pool(e1_2))
        e3_2 = self.enc3_2(self.pool(e2_2))
        e4_2 = self.enc4_2(self.pool(e3_2))

        # Corrigir dimensões
        if e4_1.shape[2:] != e4_2.shape[2:]:
            e4_2 = F.interpolate(e4_2, size=e4_1.shape[2:], mode='bilinear', align_corners=False)

        # Bottleneck
        b = self.bottleneck(torch.cat([e4_1, e4_2], dim=1))

        # Decode
        d4 = self.dec4(b, self.cbam4(torch.cat([e4_1, e4_2], dim=1)))
        d3 = self.dec3(d4, self.cbam3(torch.cat([e3_1, e3_2], dim=1)))
        d2 = self.dec2(d3, self.cbam2(torch.cat([e2_1, e2_2], dim=1)))
        d1 = self.dec1(d2, self.cbam1(torch.cat([e1_1, e1_2], dim=1)))

        return torch.sigmoid(self.final(d1))

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