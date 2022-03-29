import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from pydicom import read_file
from pydicom.pixel_data_handlers.util import apply_voi_lut
import multiprocessing

# change the path
path = '/scratch1/knarasim/physionet.org/files/vindr-spinexr/1.0.0/'

train_path = path + "train_images/"
test_path  = path + "test_images/"
annot_path = path + "annotations/"
tiny_path = "/".join(path.split("/")[:-2]) + "/tiny_vindr/"

os.makedirs(tiny_path, exist_ok=True)
os.makedirs(tiny_path + "train_images/", exist_ok=True)
os.makedirs(tiny_path + "test_images/", exist_ok=True)
os.makedirs(tiny_path + "annotations/", exist_ok=True)

def read_xray(path, voi_lut=True, fix_monochrome=True):
    
    dicom = read_file(path)
    # transform raw DICOM data to "human-friendly" view
    if voi_lut and len(dicom.get("VOILUTSequence", [])):
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array.astype("float")
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data -= np.min(data)
    data /= np.max(data)
    data *= 255
        
    return data.astype(np.uint8)

def save_tiny_dicom_train(dicom_file):
    name = dicom_file.split(".")[0]
    if not dicom_file.endswith("dicom"):
        return
    a = read_xray(train_path + dicom_file)
    plt.imsave(tiny_path + f"train_images/{name}.jpg", a)
    
def save_tiny_dicom_test(dicom_file):
    name = dicom_file.split(".")[0]
    if not dicom_file.endswith("dicom"):
        return
    a = read_xray(test_path + dicom_file)
    plt.imsave(tiny_path + f"test_images/{name}.jpg", a)

#### Make Tiny Version
annotations = pd.read_csv(annot_path + "train.csv")
annotations = annotations[["image_id", "lesion_type", "xmin", "ymin", "xmax", "ymax"]]
annotations.to_csv(tiny_path + "annotations/train.csv")

pool_obj = multiprocessing.Pool(200)
train_dicom_files = os.listdir(train_path)
_ = pool_obj.map(save_tiny_dicom_train,train_dicom_files)
pool_obj.close()

annotations = pd.read_csv(annot_path + "test.csv")
annotations = annotations[["image_id", "lesion_type", "xmin", "ymin", "xmax", "ymax"]]
annotations.to_csv(tiny_path + "annotations/test.csv")

pool_obj = multiprocessing.Pool(200)
test_dicom_files = os.listdir(test_path)
_ = pool_obj.map(save_tiny_dicom_test,test_dicom_files)
pool_obj.close()