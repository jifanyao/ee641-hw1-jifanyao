import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_keypoints_from_heatmaps(heatmaps):
    N, K, H, W = heatmaps.shape
    coords = []
    for i in range(N):
        kp = []
        for k in range(K):
            flat_idx = torch.argmax(heatmaps[i, k])
            y = flat_idx // W
            x = flat_idx % W
            kp.append([x.item() / W, y.item() / H]) 
        coords.append(kp)
    return torch.tensor(coords)

def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    pck = {}
    gt = ground_truths.cpu().numpy()
    pred = predictions.cpu().numpy()
    N, K, _ = gt.shape
    for t in thresholds:
        correct = 0
        total = N * K
        for i in range(N):
            norm = 1.0 
            for k in range(K):
                dist = np.linalg.norm(pred[i, k] - gt[i, k])
                if dist < t * norm:
                    correct += 1
        pck[t] = correct / total
    return pck

def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    thresholds = list(pck_heatmap.keys())
    plt.figure()
    plt.plot(thresholds, list(pck_heatmap.values()), marker="o", label="Heatmap")
    plt.plot(thresholds, list(pck_regression.values()), marker="s", label="Regression")
    plt.xlabel("Threshold")
    plt.ylabel("PCK")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    img = image.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap="gray")

    
    px, py = pred_keypoints[:, 0].cpu().numpy() * 128, pred_keypoints[:, 1].cpu().numpy() * 128
    gx, gy = gt_keypoints[:, 0].cpu().numpy() * 128, gt_keypoints[:, 1].cpu().numpy() * 128

    plt.scatter(px, py, c="r", label="pred", marker="x")
    plt.scatter(gx, gy, c="g", label="gt", marker="o")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

