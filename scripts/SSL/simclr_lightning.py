import numpy as np, os
import math
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
from SimCLR import SimCLR
from sklearn.metrics import classification_report


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


path = "/data/avramidi/tiny_vindr/"
train, valid, test = train_test_split(path)
annotations = pd.read_csv(path + "/annotations/train.csv")
train_annot = [
    annotations.iloc[idx]
    for idx, row in enumerate(annotations.iterrows())
    if "{}train_images/{}.jpg".format(path, row[1]["image_id"]) in train
]
train_annot = (
    pd.DataFrame(train_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])
)
valid_annot = [
    annotations.iloc[idx]
    for idx, row in enumerate(annotations.iterrows())
    if "{}train_images/{}.jpg".format(path, row[1]["image_id"]) in valid
]
valid_annot = (
    pd.DataFrame(valid_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])
)
annotations = pd.read_csv(path + "/annotations/test.csv")
test_annot = [
    annotations.iloc[idx]
    for idx, row in enumerate(annotations.iterrows())
    if "{}test_images/{}.jpg".format(path, row[1]["image_id"]) in test
]
test_annot = (
    pd.DataFrame(test_annot).drop_duplicates(subset=["image_id"]).sort_values(by=["image_id"])
)


class VinDrSpineXR(data.Dataset):
    def __init__(self, root_dir, annot_df, transform=None):
        self.img_paths = root_dir
        self.img_paths.sort()
        self.transform = transform
        print(annot_df.keys())
        self.labels = annot_df[["image_id", "lesion_type"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        #  0: Abnormal  |  1: Normal
        label = self.labels.iloc[idx, 1] == "No finding"
        image1 = self.transform(image)
        image2 = self.transform(image)
        return image1, image2, label * 1


contrastive_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=(0.4), contrast=0.4, saturation=0.4, hue=0.1)], p=0.8
        ),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

batch_size = 32
num_workers = 4

train_dataset = VinDrSpineXR(train, train_annot, contrastive_transform)
valid_dataset = VinDrSpineXR(valid, valid_annot, contrastive_transform)
train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
valid_loader = data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
)


# hyperparams
weight_decay = 1e-6
learning_rate = 0.1
n_epochs = 20
temperature = 0.1
n_classes = 2

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

pretrained_checkpoint = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
model = SimCLR(
    n_classes,
    pretrained_checkpoint=pretrained_checkpoint,
    temperature=temperature,
    weight_decay=weight_decay,
    lr=learning_rate,
    max_epochs=n_epochs,
).to(device)

save_model_path = os.path.join(os.getcwd(), "saved_models/")
filename = "SimCLR_ResNet50_adam_"
save_name = save_model_path + filename + ".ckpt"
checkpoint_callback = ModelCheckpoint(
    filename=filename,
    dirpath=save_model_path,
    every_n_epochs=1,
    save_last=True,
    save_top_k=2,
    monitor="Contrastive loss_epoch",
    mode="min",
)


def train_simclr(model, max_epochs, train_loader, val_loader, pretrained_checkpoint):
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=1 if str(device) == "cuda:3" else 0,
        # accelerator='gpu', devices=1,
        max_epochs=max_epochs,
    )
    pl.seed_everything(42)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.save_checkpoint(save_name)
    return model


if __name__ == "__main__":
    train_simclr(model, n_epochs, train_loader, valid_loader, pretrained_checkpoint)
