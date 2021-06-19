import logging
import torch
import os
import glob
import deep_sdf.workspace as ws
from PIL import Image
import torchvision.transforms as transforms
import pickle

def get_instance_files(data_source, split, level='easy'):
    instance_names =[]

    images = []
    for dataset in split:
        for class_name in split[dataset]:
            for i, instance_name in enumerate(split[dataset][class_name]):
                
                #list all images
                img_dir_path = os.path.join(data_source, class_name, instance_name, level, 'image_rgb')
                img_path_list = glob.glob(os.path.join(img_dir_path, '*.png'))
                if len(img_path_list) == 0:
                    logging.warning(
                        "No 2d image for '{}'".format(instance_name)
                    )
                    continue        
                for img in img_path_list:
                    images.append(img)
                    instance_names.append(instance_name)

    return images, instance_names

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

class EmbeddingSet(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        level = 'easy',
        pretrain_embedding_folder = "./pretrained_embedding/"
    ):
        
        self.pretrain_embedding_folder = pretrain_embedding_folder
        self.__load_embedding(split)

        self.data_source = data_source
        #get latent_vector groundtruth
        
        self.instance_images, self.instance_names = get_instance_files(data_source, split, level=level)
     
        logging.debug(f"load {len(self.embedding_dict)} embeddings and load {len(set(self.instance_names))} instance")
        assert (len(set(self.instance_names)) == len(self.embedding_dict))
    
    def __load_embedding(self, split):
        for dataset in split:
            for class_name in split[dataset]:
                self.class_name = class_name
                break
                
        with open(os.path.join(self.pretrain_embedding_folder, f"{self.class_name}_embedding.pkl"), 'rb') as f:
            embedding_dict = pickle.load(f)
        self.embedding_dict = embedding_dict

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, idx):
        img_filename = self.instance_images[idx]
        img_instance = self.instance_names[idx]
        return read_image(img_filename), self.embedding_dict[img_instance]
