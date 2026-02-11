
from enum import Enum


# Squeeze-and-Excitation (SE) block types
class SEType(Enum):
    NOVEL = "novel"        # Use a novel SE block
    ORIGINAL = "original"  # Use the original SE block 
    NONE = "none"          # Do not use SE block


# Attention Gate (AG) types
class AGType(Enum):
    NOVEL = "novel"        # Use a custom AG block
    ORIGINAL = "original"  # Use the original AG block
    NONE = "none"          # Do not use AG block


# Model types
class ModelType(Enum):
    SCAR = "SCAR"      # Full scar segmentation model
    NASE = "NASE"      # Model with no novel SE blocks
    NSE = "NSE"        # Model with no SE blocks at all
    NCAG = "NCAG"      # Model with no novel AG blocks
    NAG = "NAG"        # Model with no AG blocks at all
    NNL = "NNL"        # Model with no custom loss
    NDS = "NDS"        # Model with no deep supervision
