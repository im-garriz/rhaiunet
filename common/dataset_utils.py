import os
import pandas as pd
import random
import cv2
import numpy as np


def get_expandedUnetDataset(path):

    images_path = os.path.join(path, "images")
    gt_path = os.path.join(path, "gt")

    image_files = [f for f in os.listdir(images_path)]

    for image in image_files:
        image_id = int(image.split('.')[0])
        
        if image_id >= 178:
            label = "malignant"
        else:
            label = "benign"

        _image = cv2.imread(os.path.join(images_path, image))
        _mask = cv2.imread(os.path.join(gt_path, image.replace(".jpg", ".png")))

        new_image_name = label + '_' + image.replace(".jpg", ".png")

        print(f"Label: {label}, max: {np.max(_mask)}")

        _mask[_mask > 100] = 255

        cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/images/images_expandedUnetPaper/{new_image_name}", _image)
        cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/gt/gt_expandedUnetPaper/{new_image_name}", _mask)



def get_datasetB(path):
    images = []
    
    images_path = os.path.join(path, "original")
    gt_path = os.path.join(path, "GT")

    malignant = [18,23,24,25,27,28,29,30,31,41,43,44,45,46,47,48,49,50,53,54,55,57,132]

    image_files = [f for f in os.listdir(images_path)]

    for image in image_files:
        image_id = int(image.split('.')[0])
        
        if image_id in malignant or image_id >= malignant[-1]:
            label = "malignant"
        else:
            label = "benign"

        images.append((os.path.join(images_path, image), os.path.join(gt_path, image), label))

        _image = cv2.imread(os.path.join(images_path, image))
        _mask = cv2.imread(os.path.join(gt_path, image))

        new_image_name = label + '_' + image

        cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/images_DatasetB/{new_image_name}", _image)
        cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/gt_DatasetB/{new_image_name}", _mask)

    return images

def get_dataset112(path):
   
    folders = ["benign", "malignant"]

    for folder in folders:

        images_path = os.path.join(path, folder, "images")
        gt_path = os.path.join(path, folder, "labels")

        image_files = os.listdir(images_path)

        for image in image_files:

            _image = cv2.imread(os.path.join(images_path, image))
            _mask = cv2.imread(os.path.join(gt_path, image))

            new_name = f"D1-12_{folder}_{image}".replace(".bmp", ".png")

            cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/images/Dataset_paper1-12/{new_name}", _image)
            cv2.imwrite(f"/home/inaki/shared_files/TFM_Dataset/gt/Dataset_paper1-12/{new_name}", _mask)

def gen_dataset(path, val_p = 0.20):

    folders = ["BUSI", "DatasetB", "ExpandedUnetPaper", "Dataset_paper1-12"]

    images = []

    for folder in folders:

        image_files = os.listdir(os.path.join(path, folder))
        for file in image_files:
            images.append(os.path.join(path.replace("/home/inaki", "/workspace"), folder, file))

    print(f"N of images: {len(images)}")

    random.shuffle(images)

    n_val = int(len(images) * val_p)
    val_set = random.sample(images, k=n_val)

    print(f"N_val: {n_val}, N_train: {len(images) - n_val}")

    if True:
        train_set = []

        for image in images:
            if image not in val_set:
                train_set.append(image)

        
        with open("train_dataset2.csv", 'w') as file:
            file.write("image_path,gt_path,label\n")
            for image in train_set:
                label = "benign" if "benign" in image else "malignant"
                mask = image.replace("/images/", "/gt/")
                file.write(f"{image},{mask},{label}\n")       

        with open("val_dataset2.csv", 'w') as file:
            file.write("image_path,gt_path,label\n")
            for image in val_set:
                label = "benign" if "benign" in image else "malignant"
                mask = image.replace("/images/", "/gt/")
                file.write(f"{image},{mask},{label}\n")  
    

if __name__ == '__main__':
    path = "/home/inaki/shared_files/BUS"
    #dataset_B = get_datasetB(path)
    #gen_dataset(dataset_B)
    #get_expandedUnetDataset("/home/inaki/shared_files/nuevas/pone.0253202.s001/Data/TrainingDataSet")
    gen_dataset("/home/inaki/shared_files/TFM_Dataset/images")
    #get_dataset112("/home/inaki/shared_files/Dataset_BUSI_with_GT/11/code_data_results/dataset/BUS_images")
