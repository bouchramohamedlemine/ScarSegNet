from torch import nn
from Config import ModelType, SEType, AGType
from .Encoder import EnhancedEncoder
from .Bottleneck import Bottleneck
from .Decoder import Upsample_Block
import torch.nn.functional as F 
import timm


class Scar_Net(nn.Module):
    """
    Architecture:
      - ConvNeXt encoder backbone
      - Enhanced encoder with adaptive SE blocks
      - Bottleneck layer
      - Decoder with custom attention-gated upsampling blocks
      - Deep supervision heads at the penultimate decoder levels
      - Final prediction head for scar mask
    """
    def __init__(self, model_type, num_classes=1):
        super().__init__()

        self.model_type = model_type

        encoder = timm.create_model('convnext_base', pretrained=True, features_only=False, in_chans=1)
        
        # Map model type to SE block configuration
        SE_MAP = {
            ModelType.NASE: SEType.ORIGINAL,
            ModelType.NSE: SEType.NONE,
        }

        # Map model type to AG block configuration
        AG_MAP = {
            ModelType.NAG: AGType.NONE,
            ModelType.NCAG: AGType.ORIGINAL,
        }

        self.encoder = EnhancedEncoder(encoder, se_type=SE_MAP.get(model_type, SEType.NOVEL))

        self.bottleneck = Bottleneck(1024)

        # Decoder with upsampling blocks and configurable AG blocks
        self.up1 = Upsample_Block(1024, 512, 256, ag_type=AGType.ORIGINAL)
        self.up2 = Upsample_Block(256, 256, 128, ag_type=AGType.ORIGINAL)
        self.up3 = Upsample_Block(128, 128, 64, ag_type=AG_MAP.get(model_type, AGType.NOVEL))
        self.up4 = Upsample_Block(64, 128, 32, ag_type=AG_MAP.get(model_type, AGType.NOVEL))

        # Deep supervision heads (auxiliary outputs)
        self.ds1 = nn.Conv2d(256, num_classes, 1)
        self.ds2 = nn.Conv2d(128, num_classes, 1)
        self.ds3 = nn.Conv2d(64, num_classes, 1)  

        # Final prediction head
        self.final = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, num_classes, 1)
        )


    def forward(self, x):
        """
        Forward pass of Scar_Net.
        Returns:
          - final_out : final scar mask prediction (upsampled to input size)
          - ds_out3   : deep supervision output from intermediate decoder stage
        """
        input_h, input_w = x.shape[2], x.shape[3]   

        # Encoder forward pass
        stem_x, f1, f2, f3, f4 = self.encoder(x)

        # Bottleneck
        bottleneck = self.bottleneck(f4)

        # Decoder with skip connections
        up1 = self.up1(bottleneck, f3)
        up2 = self.up2(up1, f2)
        up3 = self.up3(up2, f1)
        up4 = self.up4(up3, stem_x)

        # Final prediction head
        final_out = self.final(up4)
        final_out = F.interpolate(final_out, size=(input_h, input_w), mode='bilinear', align_corners=False)

        # Deep supervision  
        ds_out3 = F.interpolate(self.ds3(up3), size=(input_h, input_w), mode='bilinear', align_corners=False)

        return final_out, ds_out3
