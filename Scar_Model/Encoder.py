from .Adaptive_pooling import T_Max_Avg_pooling
from torch import nn
import torch 
from Config import SEType


class FeatureAttentionBlock(nn.Module):
    """
    Channel attention block.
    Uses either the proposed adaptive T-Max-Avg pooling or
    standard average pooling, followed
    by the FC layers like in standard SE.
    """
    def __init__(self, height, width, in_channels, init_T=0, k_ratio=0, novel=False, reduction=16):
        super(FeatureAttentionBlock, self).__init__()
        self.reduction = reduction

        # Pooling layer
        self.pool = T_Max_Avg_pooling(input_size=height, init_T=init_T, k_ratio=k_ratio) if novel else nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for channel reweighting
        self.fc = nn.Sequential(
          nn.Linear(in_channels, in_channels // reduction, bias=False),
          nn.ReLU(inplace=True),
          nn.Linear(in_channels // reduction, in_channels, bias=False),
          nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()

        # Global pooling  
        pooled = self.pool(x).view(b, c)

        # Channel-wise gating weights
        y = self.fc(pooled).view(b, c, 1, 1)

        # Apply attention weights
        out = x * y

        return out


class EnhancedEncoder(nn.Module):
    """
    Encoder based on a ConvNeXt encoder backbone.
    """
    def __init__(self, model, se_type=SEType.NONE):
        super().__init__()
        # Extract encoder stages from ConvNeXt
        self.stem = model.stem
        self.stage1 = model.stages[0]   
        self.stage2 = model.stages[1]  
        self.stage3 = model.stages[2]   
        self.stage4 = model.stages[3]   

        # Choose SE blocks based on SEType
        if se_type == SEType.NOVEL:
          self.att1 = FeatureAttentionBlock(56, 56, 128, init_T=0.8, k_ratio=0.3, novel=True)  
          self.att2 = FeatureAttentionBlock(28, 28, 256, init_T=0.7, k_ratio=0.2, novel=True)    
          self.att3 = FeatureAttentionBlock(14, 14, 512, init_T=0.6, k_ratio=0.1, novel=True)    
          self.att4 = FeatureAttentionBlock(7, 7, 1024, init_T=0.5, k_ratio=0.05, novel=True) 

        elif se_type == SEType.ORIGINAL:
          self.att1 = FeatureAttentionBlock(56, 56, 128)  
          self.att2 = FeatureAttentionBlock(28, 28, 256)    
          self.att3 = FeatureAttentionBlock(14, 14, 512)    
          self.att4 = FeatureAttentionBlock(7, 7, 1024) 

        else:
          self.att1 = nn.Identity()
          self.att2 = nn.Identity()
          self.att3 = nn.Identity()
          self.att4 = nn.Identity()
                    
                                            
    def forward(self, x):
        # Initial stem convs
        stem_out = self.stem(x)           

        # Stage 1 features + attention
        f1 = self.stage1(stem_out)     
        f1 = self.att1(f1)

        # Stage 2 features + attention
        f2 = self.stage2(f1)              
        f2 = self.att2(f2)

        # Stage 3 features + attention
        f3 = self.stage3(f2)             
        f3 = self.att3(f3)

        # Stage 4 features + attention
        f4 = self.stage4(f3)             
        f4 = self.att4(f4)

        return stem_out, f1, f2, f3, f4
