import torch
from tqdm import tqdm
from metrics import mean_iou


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


@torch.no_grad()
def validate_one_epoch(model, dataloader, loss_fn, device, num_classes=7):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        outputs = model(images)
        loss = loss_fn(outputs, masks)

        running_loss += loss.item()
        running_iou += mean_iou(outputs, masks, num_classes=num_classes)

    return running_loss / len(dataloader), running_iou / len(dataloader)