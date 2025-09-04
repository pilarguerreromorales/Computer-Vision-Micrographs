import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from segmentation_models_pytorch import UnetPlusPlus, Unet, DeepLabV3Plus
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F

def cryo_loss(pred, target, contamination_ratio=0.15, smooth=1e-6):
    pred = pred.clamp(min=-20, max=20)  # Prevent exploding logits
    bce = F.binary_cross_entropy_with_logits(pred, target)

    probs = torch.sigmoid(pred)
    intersection = (probs * target).sum()
    union = probs.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)

    pt = torch.where(target == 1, probs, 1 - probs)
    pt = pt.clamp(min=1e-4, max=1.0)  # Avoid log(0)
    focal = -contamination_ratio * (1 - pt) ** 2 * torch.log(pt)
    focal_loss = focal.mean()

    return bce + (1 - dice) + focal_loss

class MicrographCleaner(pl.LightningModule):
    def __init__(self, architecture='unetplusplus', learning_rate=1e-5, encoder_name="resnet34"):
        super().__init__()
        self.save_hyperparameters()

        # Initialize model based on chosen architecture
        if architecture == 'unetplusplus':
            self.model = UnetPlusPlus(
                encoder_name="timm-efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation="sigmoid"
            )
        elif architecture == 'unet':
            self.model = Unet(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation="sigmoid"
            )
        elif architecture == 'deeplabv3':
            self.model = DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights="imagenet",
                in_channels=1,
                classes=1,
                activation="sigmoid"
            )
        elif architecture == 'self-att':
            self.model = UnetPlusPlus(
                encoder_name="timm-efficientnet-b5",
                encoder_weights="advprop",
                in_channels=1,
                classes=1,
                activation=None,
                decoder_attention_type="scse"
            )

        self.freeze_encoder()
        self.learning_rate = learning_rate

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def compute_dice_loss(self, pred, target, smooth=1.):
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
        return 1 - dice.mean()

    def compute_iou(self, pred, target):
        best_iou = 0
        best_thresh = 0.5

        for threshold in np.arange(0.3, 0.7, 0.05):
            bin_pred = (pred > threshold).float()
            intersection = (bin_pred * target).sum(dim=(2, 3))
            union = (bin_pred + target).clamp(0, 1).sum(dim=(2, 3))
            iou = (intersection + 1e-6) / (union + 1e-6)
            mean_iou = iou.mean().item()

            if mean_iou > best_iou:
                best_iou = mean_iou
                best_thresh = threshold

        self.log('best_threshold', best_thresh, on_epoch=True, prog_bar=True)
        return best_iou

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = cryo_loss(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.compute_dice_loss(outputs, masks)
        iou = self.compute_iou(outputs, masks)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_iou', iou, on_epoch=True, prog_bar=True)
        return {"val_loss": loss, "val_iou": iou}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        # Compute steps per epoch safely
        steps_per_epoch = self.trainer.estimated_stepping_batches if self.trainer else 1000  # Fallback value

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,  # 🔥 Peak LR (Adjust based on LR Finder)
            total_steps=steps_per_epoch,  # 🔥 Dynamically computed
            pct_start=0.1,  # 10% warm-up
            anneal_strategy='cos',  # Cosine decay
            div_factor=10,  # Initial LR = max_lr / div_factor
            final_div_factor=100  # End LR = max_lr / final_div_factor
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # ✅ Adjust per batch (OneCycleLR works per step)
                "frequency": 1,
            }
        }


    def on_epoch_end(self):
      if self.current_epoch == 2:
          self.unfreeze_encoder()
          optimizer = self.trainer.optimizers[0]
          for param_group in optimizer.param_groups:
              param_group['lr'] = self.learning_rate / 2  # Reduce LR to stabilize fine-tuning