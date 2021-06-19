import logging
import torch
import os
import glob
import deep_sdf.workspace as ws
from PIL import Image
import torchvision.transforms as transforms

def get_instance_files(data_source, split, level='easy'):
    embedding_ids =[]

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
                    embedding_ids.append(i)

    return images, embedding_ids

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
        experiment_directory,
        split,
        checkpoint,
        level = 'easy'
    ):


        self.data_source = data_source
        #get latent_vector groundtruth
        latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
        self.latent_vectors = latent_vectors
        
        self.instance_images, self.embedding_ids = get_instance_files(data_source, split, level=level)
     
    
        logging.debug(
            "using "
            + str(len(self.latent_vectors))
            + " shapes from data source "
            + data_source
        )


    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, idx):
        img_filename = self.instance_images[idx]
        return read_image(img_filename), self.latent_vectors[self.embedding_ids[idx]]
