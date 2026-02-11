
import numpy as np
import os 
from torch.utils.data import Dataset    


# Custom Dataset 
class Dataset(Dataset):
    """
    Custom Dataset for loading cardiac MRI scans from .npz files.
    Each .npz file contains:
        - 'image'           : MRI slice  
        - 'mask'            : binary scar mask  
        - 'myocardium_mask' : binary LV myocardium mask  
    """
    def __init__(self, npz_dir, transform=None):
        # Collect all .npz files in the directory
        self.files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        self.transform = transform  # Albumentations transform

    def __len__(self):
        """Return total number of files in dataset."""
        return len(self.files)

    def __getitem__(self, idx):
        """
        Load a single sample (image + masks) at given index.
        Returns (image, scar_mask, lv_mask).
        """
        # Load .npz file
        with np.load(self.files[idx]) as data:
            image = data['image'].astype(np.float32)           # MRI image  
            scar_mask = data['mask'].astype(np.uint8)          # Scar segmentation mask  
            lv_mask = data['myocardium_mask'].astype(np.uint8) # Myocardium mask  

        # Expand dimensions to make channel explicit → (H, W, 1)
        image = np.expand_dims(image, axis=-1)
        scar_mask = np.expand_dims(scar_mask, axis=-1)
        lv_mask = np.expand_dims(lv_mask, axis=-1)

        # Apply augmentations if provided
        if self.transform:
            augmented = self.transform(image=image, scar_mask=scar_mask, lv_mask=lv_mask)
            image = augmented['image']           
            scar_mask = augmented['scar_mask']   
            lv_mask = augmented['lv_mask']      

        return image, scar_mask.float(), lv_mask.float()
