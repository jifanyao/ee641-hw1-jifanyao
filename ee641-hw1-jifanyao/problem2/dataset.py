import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import json
import torchvision.transforms as T

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0):
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        with open(annotation_file, 'r') as f:
            data = json.load(f)

    
        self.entries = data["images"]
        self.num_keypoints = data["num_keypoints"]

        self.transform = T.Compose([
            T.Grayscale(),
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def generate_heatmap(self, keypoints, height, width):
        heatmaps = np.zeros((self.num_keypoints, height, width), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0:  # skip invalid
                continue
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            heatmaps[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * self.sigma ** 2))
        return torch.tensor(heatmaps)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_file = os.path.join(self.image_dir, entry["file_name"])
        image = Image.open(img_file).convert("L")
        image = self.transform(image)  # [1,128,128]

        keypoints = np.array(entry["keypoints"], dtype=np.float32)

        if self.output_type == "heatmap":
            scale_x = self.heatmap_size / 128
            scale_y = self.heatmap_size / 128
            kp_scaled = keypoints * [scale_x, scale_y]
            targets = self.generate_heatmap(kp_scaled, self.heatmap_size, self.heatmap_size)
        else:  # regression
            targets = keypoints / 128.0
            targets = torch.tensor(targets.flatten(), dtype=torch.float32)

        return image, targets

