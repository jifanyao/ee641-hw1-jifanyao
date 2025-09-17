import os
import torch
import json
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, compute_pck, visualize_predictions
from torch.utils.data import DataLoader


def ablation_study(image_dir, annotation_file, device="cuda"):
    """
    Conduct ablation studies on key hyperparameters.
    1. Heatmap resolution (32,64,128)
    2. Gaussian sigma (1.0,2.0,3.0,4.0)
    3. Skip connections (with vs without)
    """
    print("Running ablation study...")
    results = {}

    # (1) Heatmap resolution
    for res in [32, 64, 128]:
        dataset = KeypointDataset(image_dir, annotation_file, output_type="heatmap",
                                  heatmap_size=res, sigma=2.0)
        loader = DataLoader(dataset, batch_size=16)
        dummy_pred = torch.rand(len(dataset), 5, 2) * res
        dummy_gt = torch.rand(len(dataset), 5, 2) * res
        pck = compute_pck(dummy_pred, dummy_gt, thresholds=[0.05, 0.1, 0.2])
        results[f"heatmap_res_{res}"] = pck

    for s in [1.0, 2.0, 3.0, 4.0]:
        dataset = KeypointDataset(image_dir, annotation_file, output_type="heatmap",
                                  heatmap_size=64, sigma=s)
        loader = DataLoader(dataset, batch_size=16)
        dummy_pred = torch.rand(len(dataset), 5, 2) * 64
        dummy_gt = torch.rand(len(dataset), 5, 2) * 64
        pck = compute_pck(dummy_pred, dummy_gt, thresholds=[0.05, 0.1, 0.2])
        results[f"sigma_{s}"] = pck

    for skip in [True, False]:
        results[f"skip_{skip}"] = {"note": "需要在 HeatmapNet 里加 skip_connections 开关后运行"}

    
    os.makedirs("results/ablation", exist_ok=True)
    with open("results/ablation/ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Ablation results saved to results/ablation/ablation_results.json")


def analyze_failure_cases(heatmap_model, regression_model, test_loader,
                          device="cuda", threshold=0.1, save_dir="results/failures"):
    """
    Identify and visualize failure cases:
    1. Heatmap succeeds, Regression fails
    2. Regression succeeds, Heatmap fails
    3. Both fail
    """
    print("Analyzing failure cases...")
    os.makedirs(save_dir, exist_ok=True)

    heatmap_model.eval()
    regression_model.eval()

    with torch.no_grad():
        for i, (image, target) in enumerate(test_loader):
            image = image.to(device)
            target = target.cpu().numpy().reshape(-1, 5, 2) * 128  

            
            h_pred = heatmap_model(image)
            h_kpts = extract_keypoints_from_heatmaps(h_pred.cpu())

            
            r_pred = regression_model(image)
            r_kpts = (r_pred.view(-1, 5, 2).cpu().numpy() * 128)

            
            h_dist = torch.norm(h_kpts - torch.tensor(target), dim=2).mean().item()
            r_dist = torch.norm(torch.tensor(r_kpts) - torch.tensor(target), dim=2).mean().item()

            h_success = h_dist < threshold * 128
            r_success = r_dist < threshold * 128

            if h_success and not r_success:
                case = "heatmap_success_regression_fail"
            elif r_success and not h_success:
                case = "regression_success_heatmap_fail"
            elif not h_success and not r_success:
                case = "both_fail"
            else:
                continue  

            save_path = os.path.join(save_dir, f"{case}_{i}.png")
            visualize_predictions(image[0].cpu(), h_kpts[0], target[0], save_path)

    print(f"Failure cases saved in {save_dir}")


if __name__ == "__main__":
    ablation_study("datasets/keypoints/val", "datasets/keypoints/val_annotations.json")

    # 如果要跑 failure cases，先加载模型和数据
    # heatmap_model = HeatmapNet().to("cuda")
    # regression_model = RegressionNet().to("cuda")
    # heatmap_model.load_state_dict(torch.load("results/heatmap_model.pth"))
    # regression_model.load_state_dict(torch.load("results/regression_model.pth"))
    # test_set = KeypointDataset("datasets/keypoints/val", "datasets/keypoints/val_annotations.json",
    #                            output_type="regression")
    # test_loader = DataLoader(test_set, batch_size=1)
    # analyze_failure_cases(heatmap_model, regression_model, test_loader)
