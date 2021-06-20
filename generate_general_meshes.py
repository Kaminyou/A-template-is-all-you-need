#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import logging
import json
import numpy as np
import os
import glob
import torch
from PIL import Image
import torchvision.transforms as transforms
import sys
import deep_sdf
import deep_sdf.workspace as ws
from networks.encoder import Encoder

import shutil

# slightly different organization of files than the one in deep_sdf.data
def get_instance_filenames(data_source):
    images = []
    available_images = sorted(os.listdir(data_source))
    if len(available_images) == 0:
        logging.warning("No images for inference")
    else:
        for image in available_images:
            images.append(os.path.join(data_source, image))
    return images

def read_image(img_path):
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)

    return img

def code_to_mesh(experiment_directory, checkpoint, image_path, use_octree=True, resolution=256):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    arch = __import__("networks." + specs["NetworkArch"], fromlist=["Decoder"])

    latent_size = specs["CodeLength"]

    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])

    decoder = torch.nn.DataParallel(decoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.model_params_subdir, checkpoint + ".pth")
    )
    saved_model_epoch = saved_model_state["epoch"]

    decoder.load_state_dict(saved_model_state["model_state_dict"])

    decoder = decoder.module.cuda()

    decoder.eval()

    #===========================================================================
    #                   Use Encoder to replace embedding                       #
    #===========================================================================
    encoder = Encoder(latent_size=latent_size)
    #encoder = torch.nn.DataParallel(encoder)

    saved_model_state = torch.load(
        os.path.join(experiment_directory, ws.latent_codes_subdir, checkpoint + ".pth")
    )

    encoder.load_state_dict(saved_model_state["latent_codes"])

    encoder = encoder.cuda()

    encoder.eval()

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    instance_images = get_instance_filenames(image_path)

    for instance_image_path in instance_images:

        print(os.path.normpath(instance_image_path))
        mesh_dir = os.path.join(image_path, "mesh")

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, os.path.basename(instance_image_path))

        offset = None
        scale = None

        instance_image = read_image(instance_image_path).unsqueeze(0).cuda()
        with torch.no_grad():
            latent_vector = encoder(instance_image)
        if use_octree:
            with torch.no_grad():
                deep_sdf.mesh.create_mesh_octree(
                    decoder,
                    latent_vector,
                    mesh_filename,
                    N=resolution,
                    max_batch=int(2 ** 17),
                    offset=offset,
                    scale=scale,
                    clamp_func=clamping_function
                )
        else:
            with torch.no_grad():
                deep_sdf.mesh.create_mesh(
                    decoder,
                    latent_vector,
                    mesh_filename,
                    N=resolution,
                    max_batch=int(2 ** 17),
                    offset=offset,
                    scale=scale
                )

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(
        description="Use a trained DeepSDF decoder to generate a mesh given a latent code."
    )
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory which includes specifications and saved model "
        + "files to use for reconstruction",
    )
    arg_parser.add_argument(
        "--checkpoint",
        "-c",
        dest="checkpoint",
        default="latest",
        help="The checkpoint weights to use. This can be a number indicated an epoch "
        + "or 'latest' for the latest weights (this is the default)",
    )

    arg_parser.add_argument(
        "--image_path",
        dest="image_path",
        type=str,
        required=True,
        help="path to general images.",
    )

    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )
    
    use_octree_group = arg_parser.add_mutually_exclusive_group()
    use_octree_group.add_argument(
        '--octree',
        dest='use_octree',
        action='store_true',
        help='Use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )
    use_octree_group.add_argument(
        '--no_octree',
        dest='use_octree',
        action='store_false',
        help='Don\'t use octree to accelerate inference. Octree is recommend for most object categories '
             'except those with thin structures like planes'
    )

    deep_sdf.add_common_args(arg_parser)
    args = arg_parser.parse_args()
    deep_sdf.configure_logging(args)
    code_to_mesh(args.experiment_directory, args.checkpoint, args.image_path, args.use_octree, args.resolution)
