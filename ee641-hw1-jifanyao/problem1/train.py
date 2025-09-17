import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors


def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes per image.
    """
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, list(targets)


def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    model.train()
    total_loss = 0
    for images, targets in dataloader:
        images = images.to(device)
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)

        preds = model(images)
        loss_dict = criterion(preds, targets, anchors)
        loss = loss_dict["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, anchors):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            for t in targets:
                t["boxes"] = t["boxes"].to(device)
                t["labels"] = t["labels"].to(device)

            preds = model(images)
            loss_dict = criterion(preds, targets, anchors)
            total_loss += loss_dict["loss_total"].item()
    return total_loss / len(dataloader)


def main():
    # Config
    batch_size = 16  
    lr = 0.001
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # ⚡ 提速

    # Dataset
    train_set = ShapeDetectionDataset(
        "datasets/detection/train",
        "datasets/detection/train_annotations.json",
        transform=transforms.ToTensor()
    )
    val_set = ShapeDetectionDataset(
        "datasets/detection/val",
        "datasets/detection/val_annotations.json",
        transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True  
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        collate_fn=collate_fn, num_workers=4, pin_memory=True 
    )

    # Model, loss, optimizer
    model = MultiScaleDetector().to(device)
    criterion = DetectionLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # Anchors (⚡ 直接生成到 GPU)
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, device=device)

    # Logging
    best_val = float("inf")
    log = {"train_loss": [], "val_loss": []}
    os.makedirs("results", exist_ok=True)
    print("start")

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        val_loss = validate(model, val_loader, criterion, device, anchors)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "results/best_model.pth")

    # Save log
    with open("results/training_log.json", "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    main()

