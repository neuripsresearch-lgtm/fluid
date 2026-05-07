import os
import torch
import numpy as np
from PIL import Image
from torch.utils import data

class SmithLoader(data.Dataset):
    """Data loader for Smith Faces (Helen) dataset with 11 facial components."""
    
    def __init__(self, root, split="training", img_size=(64, 64), is_transform=True, augmentations=None, subsample_pct=1.0):
        self.root = root     
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 11
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        if split == "training":
            split_file = os.path.join(root, "exemplars.txt")
        elif split == "validation":
            split_file = os.path.join(root, "testing.txt")
        else:
            split_file = os.path.join(root, "tuning.txt")
            
        self.files = []
        with open(split_file, 'r') as f:
            for line in f:
                if ',' in line:
                    idx, file_id = line.strip().split(',')
                    self.files.append(file_id.strip())
                    
        self.images_base = os.path.join(root, 'images')
        self.annotations_base = os.path.join(root, 'labels')
        self.number_of_images = len(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_id = self.files[index]
        img_path = os.path.join(self.images_base, f"{file_id}.jpg")
        
        img = Image.open(img_path).convert('RGB')
        
        lbl_folder = os.path.join(self.annotations_base, file_id)
        lbl_shape = np.array(img).shape[:2]
        masks = []
        for i in range(11):
            lbl_path = os.path.join(lbl_folder, f"{file_id}_lbl{i:02d}.png")
            if os.path.exists(lbl_path):
                mask = np.array(Image.open(lbl_path).convert('L'))
                masks.append(mask)
            else:
                masks.append(np.zeros(lbl_shape, dtype=np.uint8))
                
        masks_stack = np.stack(masks, axis=0)
        lbl_np = np.argmax(masks_stack, axis=0).astype(np.uint8)
        lbl = Image.fromarray(lbl_np)

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        img = img.resize((self.img_size[1], self.img_size[0]), resample=Image.LANCZOS)
        lbl = lbl.resize((self.img_size[1], self.img_size[0]), resample=Image.NEAREST)
        
        img = np.array(img).astype(np.float64) / 255.0
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl
