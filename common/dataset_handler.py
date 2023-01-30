from PIL import Image
import torch
import os
import pandas as pd
import numpy as np
from torchvision import transforms
import yaml
import cv2
import PIL


def preprocess(image_orig, grayscale):
    image_orig = np.array(image_orig)
    image = cv2.bilateralFilter(image_orig, 10, 15, 15)
    image = cv2.equalizeHist(image)
    if not grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(image).convert('RGB')

        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_GRAY2RGB)
        image_orig = Image.fromarray(image_orig).convert('RGB')
    else:
        image = Image.fromarray(image).convert('L')
        image_orig = Image.fromarray(image_orig).convert('L')

    return image, image_orig


class Set(torch.utils.data.Dataset):

    def __init__(self, csv_file, id, transform=None, augmentation_pipeline=None, cache="disk", remote=False, replacement=("", ""), generation=False, grayscale=True):

        self.id = id
        self.data = pd.read_csv(csv_file)

        self.cache = cache
        self.transform = transform

        self.remote = remote
        self.replacement = replacement
        self.generation = generation

        self.grayscale = grayscale

        if self.cache == "ram":
            self.images = []
            for idx in range(len(self.data)):

                if remote:
                    img_filename = self.data.iloc[idx, 0].replace(
                        replacement[0], replacement[1])
                    mask_filename = self.data.iloc[idx, 1].replace(
                        replacement[0], replacement[1])
                    image_orig = Image.open(img_filename).convert("L")
                    mask = Image.open(mask_filename).convert("L")
                else:
                    image_orig = Image.open(
                        self.data.iloc[idx, 0]).convert("L")
                    mask = Image.open(self.data.iloc[idx, 1]).convert("L")

                if True:
                    image, image_orig = preprocess(image_orig, self.grayscale)

                self.images.append((image, mask, image_orig))

        self.augmentation_pipeline = augmentation_pipeline

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.cache == "ram":
            image, mask, image_orig = self.images[idx]
        else:
            if self.remote:
                img_filename = self.data.iloc[idx, 0].replace(
                    self.replacement[0], self.replacement[1])
                mask_filename = self.data.iloc[idx, 1].replace(
                    self.replacement[0], self.replacement[1])
                image_orig = Image.open(img_filename).convert("L")
                mask = Image.open(mask_filename).convert("L")
            else:
                image_orig = Image.open(self.data.iloc[idx, 0]).convert("L")
                mask = Image.open(self.data.iloc[idx, 1]).convert("L")

            if True:
                image, image_orig = preprocess(image_orig, self.grayscale)

        if "malignant" in self.data.iloc[idx, 0]:
            Class = "malignant"
        elif "benign" in self.data.iloc[idx, 0]:
            Class = "benign"
        else:
            Class = "normal"

        filename = self.data.iloc[idx, 0]

        if self.augmentation_pipeline is not None:
            image, mask, image_orig = self.augmentation_pipeline(
                image, mask, image_orig)

        resize_t = transforms.Resize(
            (128, 128), interpolation=PIL.Image.NEAREST)
        to_torch_tensor_t = transforms.ToTensor()

        image = resize_t(image)
        mask = resize_t(mask)
        image_orig = resize_t(image_orig)

        image = to_torch_tensor_t(image)
        mask = to_torch_tensor_t(mask)
        image_orig = to_torch_tensor_t(image_orig)

        image = image.sub_(0.5).div_(0.5)
        image_orig = image_orig.sub_(0.5).div_(0.5)

        sample = {
            "image": image,
            "mask": mask,
            "image_p_mask": torch.cat([image, mask], axis=0),
            "class": Class,
            "filename": filename,
            "image_orig": image_orig,
        }

        return sample


class DataSet():

    def __init__(self, dataset_file, augmentation_pipelines, batchsize, workers, cache, remote=False, replacement=("", ""), generation=False, grayscale=True):

        file = open(dataset_file, 'r')
        dataset_files = yaml.safe_load(file)

        self.trainset = Set(dataset_files["train"], "train", None,
                            augmentation_pipelines["train"], cache, remote, replacement, generation, grayscale)
        self.valset = Set(dataset_files["val"], "val", None,
                          augmentation_pipelines["val"], cache, remote, replacement, generation, grayscale)
        self.testset = Set(dataset_files["test"], "test", None,
                           augmentation_pipelines["test"], cache, remote, replacement, generation, grayscale)

        self.batchsize = batchsize
        self.workers = workers

        self.trainset_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batchsize, shuffle=True,
                                                           num_workers=workers)
        self.valset_loader = torch.utils.data.DataLoader(self.valset, batch_size=batchsize, shuffle=True,
                                                         num_workers=workers)
        self.testset_loader = torch.utils.data.DataLoader(self.testset, batch_size=batchsize, shuffle=True,
                                                          num_workers=workers)


def load_dataset(parameter_dict, print_info=True):

    if print_info:
        print("[I] Loading dataset")

    #transforms = parameter_dict["transforms"]
    augmentation_pipelines = parameter_dict["augmentation_pipelines"]

    batchsize = parameter_dict["batch_size"]
    workers = parameter_dict["workers"]

    cache = parameter_dict["cache"]

    dataset_file = os.path.join("datasets", parameter_dict["dataset"])

    if not os.path.isfile(dataset_file) and dataset_file.endswith('.yaml'):
        raise Exception(f"Dataset file {dataset_file} does not exist")

    dataset = DataSet(dataset_file,
                      augmentation_pipelines, batchsize, workers, cache,
                      remote=parameter_dict['remote'],
                      replacement=(
                          parameter_dict['change_string'], parameter_dict['new_string']),
                      grayscale=parameter_dict['grayscale'])

    if print_info:
        print("-----------------------------------------------------------------")
        print("[I] DATASET INFO:")
        print(f"\tTrain set length: {len(dataset.trainset)} images")
        print(f"\tVal set length: {len(dataset.valset)} images")
        print(f"\tTest set length: {len(dataset.testset)} images\n")
        print("\n\tMini-batches size:")
        print(f"\t\tTrain set: {len(dataset.trainset_loader)} batches")
        print(f"\t\tVal set: {len(dataset.valset_loader)} batches")
        print(f"\t\tTest set: {len(dataset.testset_loader)} batches")
        print("-----------------------------------------------------------------\n\n")

    return dataset
