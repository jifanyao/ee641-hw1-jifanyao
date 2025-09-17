import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import (
    plot_pck_curves,
    visualize_predictions,
    extract_keypoints_from_heatmaps,
    compute_pck,
)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, save_path):
    best_val_loss = float("inf")
    history = {"train": [], "val": []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.cuda(), targets.cuda()
            preds = model(imgs)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.cuda(), targets.cuda()
                preds = model(imgs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    return history


def main():
    os.makedirs("results/visualizations", exist_ok=True)

    
    train_dataset = KeypointDataset(
        "datasets/keypoints/train",
        "datasets/keypoints/train_annotations.json",
        output_type="heatmap"
    )
    val_dataset = KeypointDataset(
        "datasets/keypoints/val",
        "datasets/keypoints/val_annotations.json",
        output_type="heatmap"
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model_hm = HeatmapNet().cuda()
    hist_hm = train_model(
        model_hm, train_loader, val_loader, nn.MSELoss(),
        optim.Adam(model_hm.parameters(), lr=0.001),
        30, "results/heatmap_model.pth"
    )

   
    train_dataset_reg = KeypointDataset(
        "datasets/keypoints/train",
        "datasets/keypoints/train_annotations.json",
        output_type="regression"
    )
    val_dataset_reg = KeypointDataset(
        "datasets/keypoints/val",
        "datasets/keypoints/val_annotations.json",
        output_type="regression"
    )
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32)

    model_reg = RegressionNet().cuda()
    hist_reg = train_model(
        model_reg, train_loader_reg, val_loader_reg, nn.MSELoss(),
        optim.Adam(model_reg.parameters(), lr=0.001),
        30, "results/regression_model.pth"
    )

  
    with open("results/training_log.json", "w") as f:
        json.dump({"heatmap": hist_hm, "regression": hist_reg}, f)

   
    print("Running evaluation & visualization...")
    model_hm.load_state_dict(torch.load("results/heatmap_model.pth"))
    model_reg.load_state_dict(torch.load("results/regression_model.pth"))
    model_hm.eval(); model_reg.eval()

    
    val_imgs_hm, val_targets_hm = next(iter(val_loader))  
    val_imgs_hm = val_imgs_hm.cuda()

    with torch.no_grad():
        preds_hm = model_hm(val_imgs_hm)  

    pred_coords_hm = extract_keypoints_from_heatmaps(preds_hm)       
    gt_coords_hm = extract_keypoints_from_heatmaps(val_targets_hm)   

    
    val_imgs_reg, val_targets_reg = next(iter(val_loader_reg))  
    val_imgs_reg = val_imgs_reg.cuda()

    with torch.no_grad():
        preds_reg = model_reg(val_imgs_reg)  

    pred_coords_reg = preds_reg.view(-1, 5, 2)        
    gt_coords_reg = val_targets_reg.view(-1, 5, 2)     

    
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_hm = compute_pck(pred_coords_hm, gt_coords_hm, thresholds)
    pck_reg = compute_pck(pred_coords_reg, gt_coords_reg, thresholds)

    
    plot_pck_curves(pck_hm, pck_reg, "results/visualizations/pck_curve.png")


    visualize_predictions(
        val_imgs_hm[0].cpu(), pred_coords_hm[0], gt_coords_hm[0],
        "results/visualizations/sample_heatmap.png"
    )

    
    visualize_predictions(
        val_imgs_reg[0].cpu(), pred_coords_reg[0], gt_coords_reg[0],
        "results/visualizations/sample_regression.png"
    )

    
    with open("results/pck_results.json", "w") as f:
        json.dump({"heatmap": pck_hm, "regression": pck_reg}, f, indent=2)

    print("All results saved in results/ and results/visualizations/")


if __name__ == "__main__":
    main()

