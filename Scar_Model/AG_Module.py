from torch import nn
import torch
import torch.nn.functional as F


class Scar_Specific_AG(nn.Module):
    def __init__(self, F_g, F_l, F_int, novel=False):
        super(Scar_Specific_AG, self).__init__()

        # Decoder gating signal transform
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Encoder skip transform
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.relu = nn.ReLU(inplace=True)

        # Local attention branch  
        self.psi_local = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Global attention branch with larger receptive field
        self.psi_global = nn.Conv2d(F_int, 1, kernel_size=3, stride=1, padding=3, dilation=3, bias=True) if novel else None

        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, g, x):
        # Upsample g to match skip connection resolution
        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Linear transforms
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Combine and activate
        psi_feats = self.relu(g1 + x1)

        # Global–local fusion
        psi = self.psi_local(psi_feats) 
        if self.psi_global is not None:
            psi += self.psi_global(psi_feats) 

        # Normalise 
        psi = self.bn(psi)
        psi = self.sigmoid(psi)

        # Apply attention mask
        return x * psi
