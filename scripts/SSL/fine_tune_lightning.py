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
from SimCLR import SimCLR
from torchvision import transforms, models
from sklearn.metrics import classification_report
from pl_bolts.models.self_supervised import SSLEvaluator
from torchmetrics import Accuracy


def train_test_split(path):
    train_path = path + "train_images/"
    train_list = [os.path.join(train_path, img) for img in os.listdir(train_path)]
    random.shuffle(train_list)
    threshold = int(0.8 * len(train_list))
    valid_list = train_list[threshold:]
    train_list = train_list[:threshold]
    test_path = path + "test_images/"
    test_list = [os.path.join(test_path, img) for img in os.listdir(test_path)]
    return train_list, valid_list, test_list
    
path = 'physionet.org/files/vindr-spinexr/tiny_vindr/'
train, valid, test = train_test_split(path)
annotations = pd.read_csv(path + "/annotations/train.csv")
train_annot = [
    annotations.iloc[idx] for idx, row in enumerate(annotations.iterrows()) \
        if "{}train_images/{}.jpg".format(path, row[1]["image_id"]) in train
]
train_annot = pd.DataFrame(train_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])
valid_annot = [
    annotations.iloc[idx] for idx, row in enumerate(annotations.iterrows()) \
        if "{}train_images/{}.jpg".format(path, row[1]["image_id"]) in valid
]
valid_annot = pd.DataFrame(valid_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])
annotations = pd.read_csv(path + "/annotations/test.csv")
test_annot = [
    annotations.iloc[idx] for idx, row in enumerate(annotations.iterrows()) \
        if "{}test_images/{}.jpg".format(path, row[1]["image_id"]) in test
]
test_annot = pd.DataFrame(test_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])

class VinDrSpineXR(data.Dataset):
    def __init__(self, root_dir, annot_df, transform=None):
        self.img_paths = root_dir
        self.img_paths.sort()
        self.transform = transform
        self.labels = annot_df[["image_id", "lesion_type"]]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        #  0: Abnormal  |  1: Normal
        label = self.labels.iloc[idx, 1] == "No finding"
        image = self.transform(image)
        return image, label*1
contrastive_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(size=(224,224)),
                                          transforms.RandomApply([
                                              transforms.ColorJitter(brightness=(0.2),
                                                                     contrast=0.2,
                                                                     saturation=0.2,
                                                                     hue=0.1)
                                          ], p=0.8),
                                          transforms.RandomRotation(45),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



batch_size = 128
num_workers=8
n_classes=2
checkpoint_path = 'saved_models/SimCLR_ResNet50_adam_.ckpt'#'resnet50_backbone_weights.ckpt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device {device}")
supervised_train_dataset = VinDrSpineXR(train, train_annot,train_transform)
train_loader = data.DataLoader(supervised_train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_workers)

supervised_val_dataset = VinDrSpineXR(valid, valid_annot,train_transform)
valid_loader = data.DataLoader(supervised_val_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

supervised_test_dataset = VinDrSpineXR(test, test_annot,test_transform)
test_loader = data.DataLoader(supervised_test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_workers)

#### Build and Train Network Model
# class XRModel(nn.Module):
#     def __init__(self,n_classes,checkpoint_path, use_pretrained=False, freeze=True,evaluation=True):
#         super(XRModel, self).__init__()
#         self.base_net = models.resnet50(pretrained=use_pretrained)
#         checkpoint = torch.load(checkpoint_path)
#         self.base_net.load_state_dict(checkpoint['state_dict'])
#         print(f"base net {self.base_net}")
#         n_feats = self.base_net.fc.input_dim
#         # if evaluation:
#         #     self.base_net.eval()
#         if freeze:
#             for child in self.base_net.children():
#                 for param in child.parameters():
#                     param.requires_grad = False
#         self.base_net.fc = nn.Sequential(nn.Linear(n_feats, 128),nn.ReLU(), nn.Linear(128, n_classes))
#     def forward(self, input):
#         return (self.base_net(input)).to(device)
        
# model = XRModel(n_classes=2,checkpoint_path= checkpoint_path,use_pretrained=False,freeze=False).to(device)
# optim = AdamW(model.parameters(), lr=1e-3)
# lossf = nn.CrossEntropyLoss()
# n_epochs = 1

# model = model.to(device)
# n_epochs = 30
# batch_size=32
# best_val_loss = 999999999
# acc_list = []
# val_acc_list = []
# loss_list = []
# val_loss_list = []

# save_model_path = os.path.join(os.getcwd(), "saved_models/")
# filename='FineTune_adam_'
# save_name = save_model_path +filename + '.ckpt'
# checkpoint_callback = ModelCheckpoint(filename=filename, dirpath=save_model_path,every_n_epochs=1,
#                                         save_last=True, save_top_k=2,monitor='Contrastive loss_epoch',mode='min')


class SSLFineTuner(pl.LightningModule):
    """Finetunes a self-supervised learning backbone using the standard evaluation protocol of a singler layer MLP
    with 1024 units.
    Example::
        from pl_bolts.utils.self_supervised import SSLFineTuner
        from pl_bolts.models.self_supervised import CPC_v2
        from pl_bolts.datamodules import CIFAR10DataModule
        from pl_bolts.models.self_supervised.cpc.transforms import CPCEvalTransformsCIFAR10,
                                                                    CPCTrainTransformsCIFAR10
        # pretrained model
        backbone = CPC_v2.load_from_checkpoint(PATH, strict=False)
        # dataset + transforms
        dm = CIFAR10DataModule(data_dir='.')
        dm.train_transforms = CPCTrainTransformsCIFAR10()
        dm.val_transforms = CPCEvalTransformsCIFAR10()
        # finetuner
        finetuner = SSLFineTuner(backbone, in_features=backbone.z_dim, num_classes=backbone.num_classes)
        # train
        trainer = pl.Trainer()
        trainer.fit(finetuner, dm)
        # test
        trainer.test(datamodule=dm)
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        in_features: int = 2048,
        num_classes: int = 2,
        epochs: int = 100,
        hidden_dim = None,
        dropout: float = 0.0,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = "cosine",
        decay_epochs= 10,
        gamma: float = 0.1,
        final_lr: float = 0.0,
    ):
        """
        Args:
            backbone: a pretrained model
            in_features: feature dim of backbone outputs
            num_classes: classes of the dataset
            hidden_dim: dim of the MLP (1024 default used in self-supervised literature)
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr

        self.backbone = backbone
        self.linear_layer = SSLEvaluator(n_input=in_features, n_classes=num_classes, p=dropout, n_hidden=hidden_dim)

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits.softmax(-1), y)
        self.train_auroc = AUROC(num_classes=self.num_classes)
        self.log("Train AUROC" , self.train_auroc(logits.softmax(-1), y))
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc_step", acc, prog_bar=True)
        self.log("train_acc_epoch", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.val_acc(logits.softmax(-1), y)
        self.val_auroc = AUROC(num_classes=self.num_classes)
        self.log("Val AUROC" , self.val_auroc(logits.softmax(-1), y))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc)

        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        self.test_acc(logits.softmax(-1), y)
        self.test_auroc = AUROC(num_classes=self.num_classes)
        self.log("Test AUROC" , self.test_auroc(logits.softmax(-1), y))
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", self.test_acc)

        return loss

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.backbone(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=0.9,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )

        return [optimizer], [scheduler]
        
#hyperparams
n_epochs = 20
n_classes=2
learning_rate = 0.1
weight_decay = 1e-6

save_model_path = os.path.join(os.getcwd(), "saved_models/")
filename='SimCLR_FineTune_'
save_name = save_model_path +filename + '.ckpt'

backbone = SimCLR.load_from_checkpoint(checkpoint_path, strict=False)

model = SSLFineTuner(backbone.backbone, num_classes=n_classes, learning_rate=learning_rate,weight_decay=weight_decay)
# train
trainer =  pl.Trainer(accelerator='gpu', devices=1,
                  max_epochs=n_epochs) 
trainer.fit(model, train_loader, valid_loader)
trainer.save_checkpoint(save_name)
# test
# trainer.test(datamodule=dm)







