import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # UPDATED HEAD
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            #nn.Dropout(0.2)
        )

    def forward_once(self, x):
        f = self.encoder(x)
        f = f.view(f.size(0), -1)
        return self.fc(f)

    def forward(self, a, b):
        fa = self.forward_once(a)
        fb = self.forward_once(b)
        return torch.norm(fa - fb, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class DisentangledSiamese(nn.Module):
    def __init__(self, anat_dim=128, noise_dim=64):
        super().__init__()

        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        self.anat_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, anat_dim),
        )

        self.noise_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, noise_dim),
        )

        self.noise_ortho_proj = nn.Linear(noise_dim, anat_dim, bias=False)

        self.noise_pred = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward_once(self, x):
        h = self.encoder(x).flatten(1)
        z_anat = F.normalize(self.anat_head(h), dim=1)
        z_noise = F.normalize(self.noise_head(h), dim=1)
        z_noise_ortho = F.normalize(self.noise_ortho_proj(z_noise), dim=1)
        a_hat = self.noise_pred(z_noise).squeeze(1)
        return z_anat, z_noise, z_noise_ortho, a_hat

    def similarity(self, x1, x2):
        z1a, _, _, _ = self.forward_once(x1)
        z2a, _, _, _ = self.forward_once(x2)
        score = F.cosine_similarity(z1a, z2a, dim=1)
        return score