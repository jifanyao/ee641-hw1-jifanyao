import torch

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224, device="cpu"):
    """
    Generate anchors for multiple feature maps.
    Args:
        feature_map_sizes: [(H, W), ...]
        anchor_scales: [[s1,s2,...], ...]
        image_size: input image size
        device: "cpu" or "cuda"
    Returns:
        anchors_all: list of [N,4] tensors in (x1,y1,x2,y2)
    """
    anchors_all = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / H
        stride_x = image_size / W

        # 网格中心点 (H*W, 2)
        shifts_x = (torch.arange(W, device=device) + 0.5) * stride_x
        shifts_y = (torch.arange(H, device=device) + 0.5) * stride_y
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack((shift_x, shift_y), dim=-1).reshape(-1, 2)

        # 生成 anchors
        anchors = []
        for s in scales:
            half = s / 2
            xy1 = centers - half
            xy2 = centers + half
            anchors.append(torch.cat([xy1, xy2], dim=1))
        anchors_all.append(torch.cat(anchors, dim=0))
    return anchors_all


def compute_iou(boxes1, boxes2):
    """
    Vectorized IoU computation.
    Args:
        boxes1: [N,4]
        boxes2: [M,4]
    Returns:
        IoU: [N,M]
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N,M,2]
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def match_anchors_to_targets(anchors, target_boxes, target_labels,
                             pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors with targets.
    """
    N = anchors.size(0)
    matched_labels = torch.zeros(N, dtype=torch.int64, device=anchors.device)
    matched_boxes = torch.zeros((N, 4), dtype=torch.float32, device=anchors.device)
    pos_mask = torch.zeros(N, dtype=torch.bool, device=anchors.device)
    neg_mask = torch.zeros(N, dtype=torch.bool, device=anchors.device)

    if target_boxes.numel() == 0:
        neg_mask[:] = True
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes)  
    max_iou, max_idx = iou.max(dim=1)

    pos_mask = max_iou >= pos_threshold
    neg_mask = max_iou < neg_threshold

    matched_boxes[pos_mask] = target_boxes[max_idx[pos_mask]]
    matched_labels[pos_mask] = target_labels[max_idx[pos_mask]] + 1 

    return matched_labels, matched_boxes, pos_mask, neg_mask


