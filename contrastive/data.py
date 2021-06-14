import os
import glob
import random

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def get_instance_filenames(data_source, split):
    instance_list = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                img_list = []
                img_dir_path = os.path.join(data_source, class_name, instance_name, '*', 'image_rgb')
                for img_path in glob.glob(os.path.join(img_dir_path, '*.png')):
                    img_list.append(img_path)
                instance_list.append(img_list)

    return instance_list

def read_image(img_path):
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)

    return img

class ContrastiveDataset(Dataset):
    def __init__(self, data_source, split):
        self.instance_list = get_instance_filenames(data_source, split)

    def __len__(self):
        return len(self.instance_list)

    def __getitem__(self, idx):
        img_list = self.instance_list[idx]
        img1 = read_image(random.choice(img_list))
        img2 = read_image(random.choice(img_list))
        return img1, img2

def get_dataloader(data_source, split, batch_size):
    dataset = ContrastiveDataset(data_source, split)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, drop_last=True)
    return dataloader
