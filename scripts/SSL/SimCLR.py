import numpy as np, os
import matplotlib.pyplot as plt
import pandas as pd, random
import torch, torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.optim import SGD, AdamW
from torch.utils import data
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.models.self_supervised import SimCLR as simclr_lib
from torchvision import transforms, models
from sklearn.metrics import classification_report

class Projection(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
class SimCLR(pl.LightningModule):

    def __init__(self, n_classes,pretrained_checkpoint,temperature=0.1,weight_decay=1e-6,lr=0.1,max_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.max_epochs =max_epochs
        self.weight_decay = weight_decay
        self.tou = temperature
        
        self.backbone = simclr_lib.load_from_checkpoint(pretrained_checkpoint,strict=False)
        self.fc = Projection()
        
    def contrastive_loss(self,data,mode='train'):
        
        out = data
        out_dist = data

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.tou)
        neg = sim.sum(dim=-1)
        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.tou)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.tou)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
    #   optimizer = LARS(
    #             params,
    #             lr=self.lr,
    #             momentum=0.9,
    #             weight_decay=self.weight_decay,
    #             trust_coefficient=0.001,
    #         )
      

        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=self.max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        image1,image2,_ = batch
        
        
        h1 = self.backbone(image1)
        h2 = self.backbone(image2)
        z1 = self.fc(h1)
        z2=self.fc(h2)
        feat = torch.cat((z1,z2),dim=0)
        loss = self.contrastive_loss(feat,mode='train')
        self.log('Contrastive loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image1,image2,_ = batch
        
        h1 = self.backbone(image1)
        h2 = self.backbone(image2)
        z1 = self.fc(h1)
        z2=self.fc(h2)
        feat = torch.cat((z1,z2),dim=0)
        loss = self.contrastive_loss(feat, mode='val')
        self.log('Contrastive loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss