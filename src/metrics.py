import torch

@torch.no_grad()
def mean_iou(outputs, masks, num_classes=7):
    preds = torch.argmax(outputs, dim=1)
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = masks == cls

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if not ious:
        return 0.0

    return sum(ious) / len(ious)