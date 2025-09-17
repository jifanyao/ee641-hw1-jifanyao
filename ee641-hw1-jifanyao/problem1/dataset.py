import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform

        with open(annotation_file, "r") as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]

        self.image_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

    def __len__(self):
        """Return the total number of samples."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict with keys:
                - boxes: Tensor [N, 4] (x1,y1,x2,y2)
                - labels: Tensor [N]
        """
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        anns = self.image_to_anns.get(img_info["id"], [])
        boxes, labels = [], []
        for ann in anns:
            boxes.append(ann["bbox"])
            labels.append(ann["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4))
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels}

        if self.transform:
            image = self.transform(image)

        return image, targets

