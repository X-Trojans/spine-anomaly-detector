import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd, random
import torch, torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils import data
from torchvision import transforms, models
from sklearn.metrics import classification_report


path = "/data/avramidi/tiny_vindr/"
train_path = path + "train_images/"
test_path = path + "test_images/"
annot_path = path + "annotations/"


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


train, valid, test = train_test_split(path)
annotations = pd.read_csv(annot_path + "train.csv")
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
annotations = pd.read_csv(annot_path + "test.csv")
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
        self.labels = annot_df[["image_id", "lesion_type"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        #  0: Abnormal  |  1: Normal
        label = self.labels.iloc[idx, 1] == "No finding"
        image = self.transform(image)
        return image, label * 1


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

batch_size = 16
train_dataset = VinDrSpineXR(train, train_annot, train_transform)
valid_dataset = VinDrSpineXR(valid, valid_annot, train_transform)
test_dataset = VinDrSpineXR(test, test_annot, test_transform)
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#### Build and Train Network Model
class XRModel(nn.Module):
    def __init__(self, n_classes, use_pretrained=True, freeze=True):
        super(XRModel, self).__init__()
        self.base_net = models.resnet50(pretrained=use_pretrained)
        if freeze:
            for child in self.base_net.children():
                for param in child.parameters():
                    param.requires_grad = False
        n_feats = self.base_net.fc.in_features
        self.base_net.fc = nn.Linear(n_feats, 128)
        self.cls = nn.Sequential(nn.LeakyReLU(), nn.Linear(128, n_classes))

    def forward(self, input):
        return self.cls(self.base_net(input))


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
model = XRModel(n_classes=2, freeze=False).to(device)
optim = AdamW(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()
n_epochs = 1  # 38 + 12

for i in range(1, n_epochs + 1):
    print(f"\nEpoch {i}:")
    print("-" * 10)
    running_loss, running_hits = 0.0, 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optim.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(images)
            loss = lossf(out, labels)
            _, pred = torch.max(out, 1)
            loss.backward()
            optim.step()

        running_loss += loss.item()
        # running_hits += torch.true_divide(torch.sum(pred == labels),len(labels))
        running_hits += (torch.sum(pred == labels)).item() / batch_size

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_hits / len(train_loader)
    print("Train Loss: {:.3f}\t Acc: {:.3f}".format(epoch_loss, epoch_acc))

    running_loss, running_hits = 0.0, 0.0
    for images, labels in tqdm(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            out = model(images)
            loss = lossf(out, labels)
            _, pred = torch.max(out, 1)

        running_loss += loss.item()
        # running_hits += torch.true_divide(torch.sum(pred == labels),len(labels))
        running_hits += (torch.sum(pred == labels)).item() / batch_size

    epoch_loss = running_loss / len(valid_loader)
    epoch_acc = running_hits / len(valid_loader)
    print("Valid Loss: {:.3f}\t Acc: {:.3f}".format(epoch_loss, epoch_acc))

torch.save(model, path + "model.pt")


#### Test on Unseen Data
predictions, trues = [], []
with torch.set_grad_enabled(False):
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        _, pred = torch.max(out, 1)
        predictions.append(pred)
        trues.append(labels)

predictions = torch.cat(predictions).detach().cpu().numpy()
trues = torch.cat(trues).detach().cpu().numpy()
print(classification_report(trues, predictions))
