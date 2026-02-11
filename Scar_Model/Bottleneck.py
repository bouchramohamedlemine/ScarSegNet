from torch import nn

class Bottleneck(nn.Module):
    """
    Bottleneck block with residual connection.
    Applies two conv–BN–ReLU layers for feature refinement
    and adds the input back to preserve original information.
    """
    def __init__(self, channels, ppm_out_channels=None):
        super().__init__()
        ppm_out_channels = ppm_out_channels or channels

        # Local refinement block
        self.local_refine = nn.Sequential(
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(ppm_out_channels, ppm_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(ppm_out_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.local_refine(x) + x
        return self.relu(out)
