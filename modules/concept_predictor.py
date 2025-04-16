import torch
import torch.nn as nn
import torch.nn.functional as F
import math

############################## Concept predictors ##############################
class ConceptClassifierSegmenter(nn.Module):
    """
    A simple multi-label classification + location network to predict:
      (1) which concept tokens appear in a given latent (multi-label classification), and
      (2) where each concept appears via a per-concept mask (logits_mask).
    """
    def __init__(self, latent_channels=4, latent_size=64, out_dim=8, hidden_dim=256):
        super().__init__()
        """
        Args:
            latent_channels: Number of channels in the latents (often 4 for SD).
            latent_size: Height/width of the latent (often 64 for SD).
            out_dim: Number of concepts (e.g. <asset0>..<asset7>).
            hidden_dim: Hidden dimension for the fully-connected layers.
        """

        self.conv1 = nn.Conv2d(latent_channels, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        # After these, shape ~ (B, 64, latent_size/8, latent_size/8)
        # i.e. (B, 64, 8, 8) if latent_size=64

        self.fc = nn.Linear(64 * (latent_size // 8) * (latent_size // 8), hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

        #################################################################
        #  A "location head" to produce per-concept mask logits
        #         shape => (B, out_dim, 8, 8) which we then upsample back to (64, 64)
        #################################################################
        self.mask_conv = nn.Conv2d(64, out_dim, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        #################################################################

    def forward(self, x):
        """
        Args:
            x: Latents of shape (B, latent_channels, latent_size, latent_size)
        Returns:
            logits_cls:  (B, out_dim) for multi-label classification
            logits_mask: (B, out_dim, latent_size, latent_size) per-pixel location logits
                         (e.g. 64x64 if latent_size=64)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))        # shape (B, 64, 8, 8) if latent_size=64

        #################################################################
        # location head
        #################################################################
        mask_logits_8x8 = self.mask_conv(x)  # (B, out_dim, 8, 8)
        logits_mask = self.upsample(mask_logits_8x8)  # (B, out_dim, 64, 64)
        #################################################################

        # Classification head
        x_flat = x.view(x.size(0), -1)   # flatten
        x_fc = F.relu(self.fc(x_flat))
        logits_cls = self.out(x_fc)

        #################################################################
        # return both classification logits + mask logits
        #################################################################
        return logits_cls, logits_mask