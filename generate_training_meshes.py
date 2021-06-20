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
def get_instance_filenames(data_source, split, level='easy'):
    npzfiles = []
    images = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                    continue
                #list all images
                img_dir_path = os.path.join(data_source, class_name, instance_name, level, 'image_rgb')
                img_path_list = glob.glob(os.path.join(img_dir_path, '*.png'))
                if len(img_path_list) == 0:
                    logging.warning(
                        "No 2d image for '{}'".format(instance_filename)
                    )
                    continue
                images.append([])
                npzfiles.append(instance_filename)
                for img in img_path_list:
                    images[-1].append(img)

    return images, npzfiles

def read_image(img_path):
    img = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform(img)

    return img

def code_to_mesh(experiment_directory, checkpoint, start_id, end_id, view_id, 
                 keep_normalized=False, use_octree=True, resolution=256, mode):

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

    if mode == "train":
        train_split_file = specs["TrainSplit"]
    if mode == "test"
        train_split_file = specs["TestSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    data_source = specs["DataSource"]

    instance_images, instance_filenames = get_instance_filenames(data_source, train_split)

    for i in range(len(instance_filenames)):
        if i < start_id:
            continue

        print(os.path.normpath(instance_filenames[i]))
        if sys.platform.startswith('linux'):
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("/")
        else:
            dataset_name, class_name, instance_name = os.path.normpath(instance_filenames[i]).split("\\")
        instance_name = instance_name.split(".")[0]

        mesh_dir = os.path.join(
            experiment_directory,
            ws.training_meshes_subdir,
            str(saved_model_epoch),
            dataset_name,
            class_name,
        )

        if not os.path.isdir(mesh_dir):
            os.makedirs(mesh_dir)

        mesh_filename = os.path.join(mesh_dir, instance_name)

        offset = None
        scale = None

        if not keep_normalized:

            normalization_params = np.load(
                ws.get_normalization_params_filename(
                    data_source, dataset_name, class_name, instance_name
                )
            )
            offset = normalization_params["offset"]
            scale = normalization_params["scale"]

        # pick a view to encode, also copy the image to the current directory
        shutil.copy2(instance_images[i][view_id], mesh_filename+'.png')
        instance_image = read_image(instance_images[i][view_id]).unsqueeze(0).cuda()
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

        if i >= end_id:
            break


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
        "--keep_normalization",
        dest="keep_normalized",
        default=False,
        action="store_true",
        help="If set, keep the meshes in the normalized scale.",
    )
    arg_parser.add_argument(
        "--start_id",
        dest="start_id",
        type=int,
        default=0,
        help="start_id.",
    )
    arg_parser.add_argument(
        "--end_id",
        dest="end_id",
        type=int,
        default=5,
        help="end_id.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        type=int,
        default=256,
        help="Marching cube resolution.",
    )
    arg_parser.add_argument(
        "--view_id",
        dest="view_id",
        type=int,
        default=30,
        help="Which view of all the rgb images to encode.",
    )
    arg_parser.add_argument(
        "--mode",
        dest="mode",
        type=str,
        default="train",
        help="Generate instance from training or testing data. Option=[train, test]",
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

    code_to_mesh(
        args.experiment_directory,
        args.checkpoint,
        args.start_id,
        args.end_id,
        args.view_id,
        args.keep_normalized,
        args.use_octree,
        args.resolution
        args.mode)
