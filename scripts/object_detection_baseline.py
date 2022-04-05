from policy import Policy
import numpy as np
import os
import pandas as pd
import random
import torch
import pickle
from PIL import Image
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.optim import AdamW
from torch.utils import data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes
from torchmetrics.detection import MeanAveragePrecision
import torchvision.transforms.functional as F
import time


DISABLE_TQDM = True
torch.manual_seed(42)

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

def compute_annotation_map(annotations):
    annotation_map = {}
    for _, row in annotations.iterrows():
        boxes = annotation_map.get(row['image_id'], [])
        boxes.append((row['lesion_type'], [row['xmin'] if row['xmin']>=0 else -1,
                                           row['ymin'] if row['ymin']>=0 else -1,\
                                           row['xmax'] if row['xmax']>=0 else -1,\
                                           row['ymax'] if row['ymax']>=0 else -1\
                                           ]))
        annotation_map[row['image_id']] = boxes
    return annotation_map

def split_annotation_map(annotation_map, train, valid):
    train_ids = [image.split('/')[-1].split('.')[0] for image in train]
    valid_ids = [image.split('/')[-1].split('.')[0] for image in valid]
    return { image:annotation_map[image] for image in train_ids} , { image:annotation_map[image] for image in valid_ids}

class SpineObjectDetection(data.Dataset):
    
    def __init__(self, root_dir, annotaion_map,anomaly_map,image_id_map,transform):
        self.img_paths = root_dir
        self.img_paths.sort()
        self.image_ids = [image.split('/')[-1].split('.')[0] for image in self.img_paths]
        
        self.annotaion_map = annotaion_map
        self.anomaly_map = anomaly_map
        self.image_id_map = image_id_map
        self.transform = transform       
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx])
        image_id = self.image_ids[idx]
        
        image_width = image.size[0]
        image_height = image.size[1]
        
        image = torchvision.transforms.Resize(np.random.randint(640,800))(image)
        transformed_w = image.size[0]
        transformed_h = image.size[1]

        labels = []
        boxes = []
        area = []
        for label, box in self.annotaion_map[image_id]:
            labels.append(self.anomaly_map[label])
            
            if box[0]==-1 and box[1]==-1:
                boxes.append([0,0,1,1])
            else:
                boxes.append([
                    (box[0]/image_width)*transformed_w,
                    (box[1]/image_height)*transformed_h,
                    (box[2]/image_width)*transformed_w,
                    (box[3]/image_height)*transformed_h
                    ])
            
            area.append((boxes[-1][2] - boxes[-1][0])*(boxes[-1][3] - boxes[-1][1]))

        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = torch.as_tensor(area, dtype=torch.float32)       
        
        image_new,boxes_new = self.transform(image,boxes)
        area = (boxes_new[:,2] - boxes_new[:,0])*(boxes_new[:,3] - boxes_new[:,1])
                                          
        if (area>0).sum() == area.shape[0]:
            image,boxes = image_new,boxes_new 
        else:
            image, boxes = ToTensor()(image, boxes)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["image_id"] = torch.tensor(self.image_id_map[image_id])
        
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes):
        for t in self.transforms:
            img, boxes = t(img,boxes)
        return img, boxes

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class RandomHorizontalFlip():
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img, boxes):
        if torch.rand(1) < self.p:
            
            img_width = img.size[0]
            
            boxes[:,0] = img_width - boxes[:,0]
            boxes[:,2] = img_width - boxes[:,2]
            boxes_w = torch.abs(boxes[:,0] - boxes[:,2])
             
            boxes[:,0] -= boxes_w
            boxes[:,2] += boxes_w

            return F.hflip(img),boxes
        return img,boxes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"
    
class ToTensor:
    def __call__(self, img, boxes):
        return F.to_tensor(img), boxes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def collate_fn(batch):
    return tuple(zip(*batch))

def create_faster_rcnn_model(num_classes,trainable_backbone_layers=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,trainable_backbone_layers=trainable_backbone_layers,min_size=640,max_size=2699)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model

def save_object(path,file_name,obj):
    with open(os.path.join(path,file_name + ".pickle"), 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_object(path,file_name):
    with open(os.path.join(path,file_name + ".pickle"), 'rb') as file:
        return pickle.load(file)

class SaveBestModel:
    def __init__(self, model_name, path="/"):
        self.best_valid_loss = float('inf')
        self.path = path
        self.model_name = model_name
        
    def update(self, model, current_valid_loss):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            torch.save(model, os.path.join(self.path,self.model_name))
            print(f"Saved Model. Best validation loss: {self.best_valid_loss}")
            
    def fetch(self):
        return torch.load(os.path.join(self.path,self.model_name))
        
        
class SaveBestModelMAP:
    def __init__(self, model_name, path="/"):
        self.best_mAP = float('-inf')
        self.path = path
        self.model_name = model_name
        
    def update(self, model, current_mAP):
        if current_mAP > self.best_mAP:
            self.best_mAP = current_mAP
            torch.save(model, os.path.join(self.path,self.model_name))
            print(f"Saved Model. Best mAP: {self.best_mAP}")
            
    def fetch(self):
        return torch.load(os.path.join(self.path,self.model_name))
    
class LossHistory:
    def __init__(self,path,file_name):
        self.loss = {
            'total_loss': [],
            'classifier_loss' : [],
            'box_reg_loss':[],
            'objectness_loss': [],
            'rpn_box_reg_loss':[]
        }
        self.file_name = file_name
        self.path = path
    
    def update(self,total_loss,classifier_loss,box_reg_loss,objectness_loss,rpn_box_reg_loss):
        self.loss['total_loss'].append(total_loss)
        self.loss['classifier_loss'].append(classifier_loss)
        self.loss['box_reg_loss'].append(box_reg_loss)
        self.loss['objectness_loss'].append(objectness_loss)
        self.loss['rpn_box_reg_loss'].append(rpn_box_reg_loss)
        
    def save(self):
        save_object(self.path,self.file_name,self.loss)
    
    def load(self):
        self.loss = load_object(self.path,self.file_name)


def train_one_epoch(model,train_loader,optim):
    running_total_loss, running_loss_classifier, running_loss_box_reg, running_loss_objectness,running_loss_rpn_box_reg = 0,0,0,0,0
    for images, targets in tqdm(train_loader, disable=DISABLE_TQDM):
        images_device = list(image.to(device) for image in images)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optim.zero_grad()
        model.train()
        with torch.set_grad_enabled(True):
            loss_dict = model.forward(images_device,targets_device)
            loss = loss_dict['loss_classifier']  + loss_dict['loss_box_reg'] + loss_dict['loss_objectness']  + loss_dict['loss_rpn_box_reg']
            loss.backward()
            optim.step()

        running_total_loss += loss.item()
        running_loss_classifier +=loss_dict['loss_classifier'].item()
        running_loss_box_reg +=loss_dict['loss_box_reg'].item()
        running_loss_objectness +=loss_dict['loss_objectness'].item()
        running_loss_rpn_box_reg +=loss_dict['loss_rpn_box_reg'].item()


    epoch_total_loss = running_total_loss / len(train_loader)
    epoch_loss_classifier = running_loss_classifier / len(train_loader)
    epoch_loss_box_reg = running_loss_box_reg / len(train_loader)
    epoch_loss_objectness = running_loss_objectness / len(train_loader)
    epoch_loss_rpn_box_reg = running_loss_rpn_box_reg / len(train_loader)
    print("Total Loss: {:.6f}\t Classifier Loss: {:.6f}\t Box Reg Loss: {:.6f}\t Objectness Loss: {:.6f}\t RPN box Loss: {:.6f} "\
    .format(epoch_total_loss, epoch_loss_classifier,epoch_loss_box_reg,epoch_loss_objectness,epoch_loss_rpn_box_reg))
    return epoch_total_loss,epoch_loss_classifier,epoch_loss_box_reg,epoch_loss_objectness, epoch_loss_rpn_box_reg

def evaluate_loss(model,loader):
    running_total_loss, running_loss_classifier, running_loss_box_reg, running_loss_objectness,running_loss_rpn_box_reg = 0,0,0,0,0
    for images, targets in tqdm(loader, disable=DISABLE_TQDM):
        images_device = list(image.to(device) for image in images)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optim.zero_grad()
        model.train()
        with torch.set_grad_enabled(False):
            loss_dict = model.forward(images_device,targets_device)
            loss = loss_dict['loss_classifier']  + loss_dict['loss_box_reg'] + loss_dict['loss_objectness']  + loss_dict['loss_rpn_box_reg']

        running_total_loss += loss.item()
        running_loss_classifier +=loss_dict['loss_classifier'].item()
        running_loss_box_reg +=loss_dict['loss_box_reg'].item()
        running_loss_objectness +=loss_dict['loss_objectness'].item()
        running_loss_rpn_box_reg +=loss_dict['loss_rpn_box_reg'].item()

    epoch_total_loss = running_total_loss / len(loader)
    epoch_loss_classifier = running_loss_classifier / len(loader)
    epoch_loss_box_reg = running_loss_box_reg / len(loader)
    epoch_loss_objectness = running_loss_objectness / len(loader)
    epoch_loss_rpn_box_reg = running_loss_rpn_box_reg / len(loader)
    print("Total Loss: {:.6f}\t Classifier Loss: {:.6f}\t Box Reg Loss: {:.6f}\t Objectness Loss: {:.6f}\t RPN box Loss: {:.6f} "\
    .format(epoch_total_loss, epoch_loss_classifier,epoch_loss_box_reg,epoch_loss_objectness,epoch_loss_rpn_box_reg))
    
    return epoch_total_loss,epoch_loss_classifier,epoch_loss_box_reg,epoch_loss_objectness, epoch_loss_rpn_box_reg

def evaluate_average_precision(model,loader):
    metric = MeanAveragePrecision(class_metrics=True)
    for images, targets in tqdm(loader, disable=DISABLE_TQDM):
        images_device = list(image.to(device) for image in images)
        targets_device = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.eval()
        with torch.set_grad_enabled(False):
            preds = model.forward(images_device)
            metric.update(preds,targets_device)
    mAP =  metric.compute()
    print(mAP)
    return mAP



def train_model(model, best_model,best_model_map, train_loader,valid_loader, optim, epochs, path=".",evaluate_map_every=5):
    train_history = LossHistory(path, "train_history")
    valid_history = LossHistory(path, "valid_history")
    mAP_history = []
    mAP50_history = []
    for i in range(1, epochs+1):
        start = time.time()
        print(f"\nEpoch {i}:")
        print("-"*100)

        ## Training
        print("Train")
        total_loss,classifier_loss,box_reg_loss,objectness_loss, rpn_box_reg_loss = train_one_epoch(model,train_loader,optim)
        train_history.update(total_loss,classifier_loss,box_reg_loss,objectness_loss, rpn_box_reg_loss)
        
        ## Validation
        print("\nValidation")
        total_loss,classifier_loss,box_reg_loss,objectness_loss, rpn_box_reg_loss = evaluate_loss(model,valid_loader)
        valid_history.update(total_loss,classifier_loss,box_reg_loss,objectness_loss, rpn_box_reg_loss)
        best_model.update(model,total_loss)
        
        ## Validation Eval mAP
        print("\nmAP Validation")
        if i%evaluate_map_every ==0:
            mAP = evaluate_average_precision(model,valid_loader)
            mAP_history.append(mAP)
            mAP_50 = torch.mean(mAP["map_per_class"][1:]).item()
            print("MAP@0.5 :", mAP_50)
            mAP50_history.append(mAP_50)
            best_model_map.update(model,mAP_50)
            
        train_history.save(),
        valid_history.save()
        save_object(path,"map_history", mAP_history)
        save_object(path,"map_history_50", mAP50_history)
        print("\n Time Elapsed Per Epoch: ", time.time() - start)
    return best_model.fetch(),train_history,valid_history,mAP_history


path = '/scratch1/knarasim/physionet.org/files/vindr-spinexr/tiny_vindr/'
train_path = path + "train_images/"
test_path  = path + "test_images/"
annot_path = path + "annotations/"

anomaly_map = {
    'No finding': 0,
    'Disc space narrowing': 1,
    'Foraminal stenosis': 2,
    'Osteophytes': 3,
    'Spondylolysthesis': 4,
    'Surgical implant': 5,
    'Vertebral collapse': 6,   
    'Other lesions': 7,
 }

train, valid, test = train_test_split(path)

train_annotations = pd.read_csv(annot_path + "train.csv")
test_annotations = pd.read_csv(annot_path + "test.csv")

train_annotation_map = compute_annotation_map(train_annotations)
train_annotation_map, valid_annotation_map = split_annotation_map(train_annotation_map, train, valid)

test_annotation_map = compute_annotation_map(test_annotations)


all_image_ids = [image.split('/')[-1].split('.')[0] for image in train]\
     + [image.split('/')[-1].split('.')[0] for image in valid]\
     + [image.split('/')[-1].split('.')[0] for image in test]

image_id_map = { img_id:i+1 for i,img_id in enumerate(all_image_ids)}


train_transform = Policy('policy_v1',pre_transform=[], post_transform=[ToTensor()])

test_transform = Compose([
    ToTensor()
])



train_dataset = SpineObjectDetection(train, train_annotation_map,anomaly_map,image_id_map,train_transform)
valid_dataset = SpineObjectDetection(valid, valid_annotation_map,anomaly_map,image_id_map,train_transform)
test_dataset = SpineObjectDetection(test, test_annotation_map,anomaly_map,image_id_map,test_transform)


batch_size = 10
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=8)
valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=8)
test_loader  = data.DataLoader(test_dataset , batch_size=batch_size, shuffle=True, collate_fn=collate_fn,num_workers=8)




path = '/scratch2/knarasim/models/object_detection_auto_augment3/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_faster_rcnn_model(num_classes=8,trainable_backbone_layers=5).to(device)
optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
best_model = SaveBestModel("model_object.pt",path)
best_model_map = SaveBestModelMAP("model_object_map.pt",path)

epochs = 50
model, train_history,valid_history, mAP_history = train_model(model, best_model,best_model_map, train_loader,valid_loader, optim, epochs, path=path,evaluate_map_every=1)

results = evaluate_average_precision(model,test_loader)