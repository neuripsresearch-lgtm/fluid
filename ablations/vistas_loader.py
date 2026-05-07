import os
import json
import torch
import numpy as np
import random
from PIL import Image
from torch.utils import data
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import *

class MapillaryVistasLoader(data.Dataset):
    """Data loader for Mapillary Vistas street-level imagery dataset (66 classes)."""
    
    def __init__(self, root, split="training", img_size=(512, 1024), 
                 is_transform=True, augmentations=None, 
                 subsample_pct=1.0, seed=1337):
        self.root = root     
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 66
        self.seed = seed

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        # Standard Vistas mean
        self.mean = np.array([125.0, 125.0, 125.0]) # Simplified or use Vistas specific
        
        self.images_base = os.path.join(self.root, self.split, 'images')
        self.annotations_base = os.path.join(self.root, self.split, 'v1.2', 'labels')

        self.all_files = recursive_glob(rootdir=self.images_base, suffix='.jpg')
        
        if not self.all_files:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        # Handle subsampling
        if subsample_pct < 1.0:
            random.seed(self.seed)
            n_total = len(self.all_files)
            n_sample = max(1, int(n_total * subsample_pct))
            self.files = random.sample(self.all_files, n_sample)
            print(f"Subsampled Mapillary {split}: {n_sample}/{n_total} images ({subsample_pct*100:.1f}%)")
        else:
            self.files = self.all_files
            print(f"Found {len(self.files)} {split} images")

        self.number_of_images = len(self.files)
        self.class_names, self.class_ids, self.class_colors = self.parse_config()
        self.ignore_id = 250

    def parse_config(self):
        # We use config_v1.2.json as preferred
        config_path = os.path.join(self.root, 'config_v1.2.json')
        if not os.path.exists(config_path):
            config_path = os.path.join(self.root, 'config.json')

        with open(config_path) as config_file:
            config = json.load(config_file)

        labels = config['labels']
        class_names = []
        class_ids = []
        class_colors = []
        
        for label_id, label in enumerate(labels):
            class_names.append(label["name"].split('--')[-1])
            class_ids.append(label_id)
            class_colors.append(label["color"])

        return class_names, class_ids, class_colors

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(self.annotations_base, os.path.basename(img_path).replace(".jpg", ".png"))

        img = Image.open(img_path).convert('RGB')
        lbl = Image.open(lbl_path)
        
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
        
        # Mapping evaluation=False labels to ignore_id
        # In V1.2 config, 'Unlabeled' is at index 65 and evaluate=false
        lbl[lbl == 65] = self.ignore_id
        
        return img, lbl
