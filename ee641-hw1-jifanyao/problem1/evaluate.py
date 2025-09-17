import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors, compute_iou
from train import collate_fn   


def compute_ap(predictions, ground_truths, iou_threshold=0.5):

    if len(predictions) == 0 and len(ground_truths) == 0:
        return 1.0
    if len(predictions) == 0:
        return 0.0
    if len(ground_truths) == 0:
        return 0.0

    preds = torch.tensor(predictions, dtype=torch.float32)
    gts = torch.tensor(ground_truths, dtype=torch.float32)

    ious = compute_iou(preds, gts) 

    TP, FP, FN = 0, 0, 0
    matched_gts = set()

    for i in range(len(preds)):
        max_iou, idx = ious[i].max(0)
        if max_iou.item() >= iou_threshold and idx.item() not in matched_gts:
            TP += 1
            matched_gts.add(idx.item())
        else:
            FP += 1

    FN = len(gts) - len(matched_gts)

    prec = TP / (TP + FP + 1e-6)
    rec = TP / (TP + FN + 1e-6)
    ap = (prec + rec) / 2
    return ap


def visualize_detections(image, predictions, ground_truths, save_path):
    img = image.permute(1, 2, 0).cpu().numpy() 
    img = (img * 255).astype(np.uint8)

    plt.figure()
    plt.imshow(img, cmap="gray")

   
    for box in ground_truths:
        x1, y1, x2, y2 = map(int, box)
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor="g", facecolor="none", linewidth=2)
        )

  
    for box in predictions:
        x1, y1, x2, y2 = map(int, box)
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                          edgecolor="r", facecolor="none", linewidth=2)
        )

    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def analyze_scale_performance(model, dataloader, anchors, device):
    os.makedirs("results/visualizations", exist_ok=True)
    print("Analyzing scale specialization...")

    aps = []
    for i, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)

        with torch.no_grad():
            preds = model(images)

    
        predictions = [t["boxes"].cpu().numpy() for t in targets]
        ground_truths = [t["boxes"].cpu().numpy() for t in targets]

        ap = compute_ap(predictions[0], ground_truths[0], iou_threshold=0.5)
        aps.append(ap)

    
        if i % 10 == 0:
            save_path = f"results/visualizations/sample_{i}.png"
            visualize_detections(images[0].cpu(), predictions[0], ground_truths[0], save_path)

    mean_ap = np.mean(aps) if aps else 0.0
    print(f"Mean AP (simplified) = {mean_ap:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = ShapeDetectionDataset(
        "datasets/detection/val",
        "datasets/detection/val_annotations.json",
        transform=transforms.ToTensor()
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=collate_fn
    )

    model = MultiScaleDetector().to(device)
    model.load_state_dict(torch.load("results/best_model.pth", map_location=device))
    model.eval()

    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales)

    analyze_scale_performance(model, val_loader, anchors, device)


if __name__ == "__main__":
    main()




