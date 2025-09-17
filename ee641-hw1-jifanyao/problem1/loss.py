import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets


class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, targets, anchors):
        """
        Args:
            predictions: list of 3 tensors [B, A*(5+C), H, W]
            targets: list of dicts [{boxes: [N,4], labels: [N]}, ...]
            anchors: list of 3 tensors [num_anchors, 4]
        """
        loss_obj, loss_cls, loss_loc = 0.0, 0.0, 0.0
        total_pos = 0 
        B = predictions[0].size(0)

        for b in range(B):
            tgt = targets[b]
            boxes = tgt["boxes"]
            labels = tgt["labels"]

            for pred, anchor in zip(predictions, anchors):
                
                pred_b = pred[b].permute(1, 2, 0).contiguous()
                pred_b = pred_b.view(-1, 5 + self.num_classes)

                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anchor, boxes, labels
                )
                device = pred_b.device

                pred_boxes = pred_b[:, :4]
                pred_obj = pred_b[:, 4]
                pred_cls = pred_b[:, 5:]

                # ----------------------------
                # Objectness loss
                # ----------------------------
                target_obj = pos_mask.float().to(device)
                obj_loss_all = F.binary_cross_entropy_with_logits(
                    pred_obj, target_obj, reduction="none"
                )

                # Hard Negative Mining (3:1)
                num_pos = pos_mask.sum().item()
                num_neg = min(int(num_pos * 3), neg_mask.sum().item())

                if num_neg > 0:
                    neg_loss_vals = obj_loss_all[neg_mask]
                    topk_neg_loss, _ = torch.topk(neg_loss_vals, num_neg)
                    loss_obj += obj_loss_all[pos_mask].sum() + topk_neg_loss.sum()
                else:
                    loss_obj += obj_loss_all.sum()

                total_pos += num_pos  

        
                pos_idx = pos_mask.nonzero(as_tuple=True)[0]
                if len(pos_idx) > 0:
                    cls_tgt = (matched_labels[pos_idx] - 1).to(device)  
                    loss_cls += F.cross_entropy(
                        pred_cls[pos_idx], cls_tgt, reduction="sum"
                    )
                    loss_loc += F.smooth_l1_loss(
                        pred_boxes[pos_idx], matched_boxes[pos_idx].to(device), reduction="sum"
                    )
        normalizer = max(1, total_pos) 
        loss_obj = loss_obj / normalizer
        loss_cls = loss_cls / normalizer
        loss_loc = loss_loc / normalizer

    
        loss_total = loss_obj + loss_cls + 2.0 * loss_loc

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total,
        }


