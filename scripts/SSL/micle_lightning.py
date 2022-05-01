import pandas as pd, random
import os, pickle, torch
from PIL import Image
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision import transforms
from SimCLR import SimCLR

path = "/data/avramidi/tiny_vindr/train_images/"
img_list = [os.path.join(path, img) for img in os.listdir(path)]

with open("../../MICLe_image_dict.pickle", "rb") as f:
    std_dict = pickle.load(f)
with open("../../corrupt_studies.pickle", "rb") as f:
    corrupted = pickle.load(f)
std_list = [s for s in std_dict.keys() if s not in corrupted]

random.shuffle(std_list)
threshold = int(0.8 * len(std_list))
train, valid = std_list[:threshold], std_list[threshold:]

annotations = pd.read_csv("../../train_orig.csv")
train_annot = [
    annotations.iloc[idx]
    for idx, row in enumerate(annotations.iterrows())
    if row[1]["study_id"] in train
]
train_annot = pd.DataFrame(train_annot).drop_duplicates(subset=["image_id"])
valid_annot = [
    annotations.iloc[idx]
    for idx, row in enumerate(annotations.iterrows())
    if row[1]["study_id"] in valid
]
valid_annot = pd.DataFrame(valid_annot).drop_duplicates(subset=["image_id"])


class VinDrSpineXR_studies(data.Dataset):
    def __init__(self, studies, annot_df, transform=None):
        self.studies = studies
        self.studies.sort()
        self.transform = transform
        self.micle_transform = transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.labels = annot_df[["image_id", "lesion_type"]]

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study = self.studies[idx]
        flag = random.choices([0, 1], weights=[0.25, 0.75], k=1)

        if flag:
            image1, image2 = random.sample(std_dict[study], 2)

            image1 = Image.open(path + image1 + ".jpg")
            image1 = self.micle_transform(image1)

            image2 = Image.open(path + image2 + ".jpg")
            image2 = self.micle_transform(image2)
        else:
            image = random.choice(std_dict[study])
            image = Image.open(path + image + ".jpg")

            image1 = self.transform(image)
            image2 = self.transform(image)

        #  0: Abnormal  |  1: Normal
        label = self.labels.iloc[idx, 1] == "No finding"
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

weight_decay = 1e-6
learning_rate = 0.1
n_epochs = 20
temperature = 0.1
n_classes = 2
batch_size = 16
num_workers = 4

train_dataset = VinDrSpineXR_studies(train, train_annot, contrastive_transform)
valid_dataset = VinDrSpineXR_studies(valid, valid_annot, contrastive_transform)
train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
ckpt = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
model = SimCLR(
    n_classes,
    pretrained_checkpoint=ckpt,
    temperature=temperature,
    weight_decay=weight_decay,
    lr=learning_rate,
    max_epochs=n_epochs,
).to(device)

save_model_path = os.path.join(os.getcwd(), "saved_models/")
filename = "MICLe_ResNet50_adam_"
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


def train_micle(model, max_epochs, train_loader, val_loader):
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        gpus=1 if str(device) == "cuda:1" else 0,
        max_epochs=max_epochs,
    )
    pl.seed_everything(42)
    trainer.fit(model, train_loader, val_loader)
    model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.save_checkpoint(save_name)
    return model


if __name__ == "__main__":
    train_micle(model, n_epochs, train_loader, valid_loader)
