from .AG_Module import Scar_Specific_AG
from torch import nn
from Config import AGType
import torch
import torch.nn.functional as F 


class DecoderBlock(nn.Module):
    """
    Standard convolutional block for the decoder.
    Consists of two Conv–BN–ReLU layers.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_block(x)
        return self.relu(x)


class Upsample_Block(nn.Module):
    """
    Decoder upsampling block with optional attention gate.
    - Upsamples decoder features
    - Applies attention gate to encoder skip connection (if enabled)
    - Concatenates and processes with a decoder block
    """
    def __init__(self, in_channels, skip_channels, out_channels, ag_type=AGType.NONE):
        super().__init__()
        # Attention gate
        if ag_type != AGType.NONE:
            self.attentionGate = Scar_Specific_AG(
                F_g=in_channels, 
                F_l=skip_channels, 
                F_int=out_channels, 
                novel=(ag_type == AGType.NOVEL)
            )
        else:
            self.attentionGate = None

        # Upsampling 
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # Decoder block  
        self.decoder_conv = DecoderBlock(out_channels + skip_channels, out_channels)

    def forward(self, dec_features, enc_features):
        # Apply attention gate if enabled
        enc_att = self.attentionGate(dec_features, enc_features) if self.attentionGate else enc_features 

        # Upsample decoder features
        dec_upsampled = self.upsample(dec_features)

        # Resize encoder features to match upsampled decoder features
        enc_att = F.interpolate(enc_att, size=dec_upsampled.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate encoder and decoder features
        concat = torch.cat([dec_upsampled, enc_att], dim=1)

        # Convolutional refinement
        out = self.decoder_conv(concat)

        return out
