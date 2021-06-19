import argparse
import json
import numpy as np
import os
import torch
import sys
import deep_sdf
import deep_sdf.workspace as ws
import pickle

def get_instance_filenames(data_source, split):
    npzfiles = []
    index_list = []
    for dataset in split:
        for class_name in split[dataset]:
            for idx, instance_name in enumerate(split[dataset][class_name]):
                
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                else:
                    npzfiles += [instance_filename]
                    index_list.append(idx)
    return index_list, npzfiles

def extract_embedding(experiment_directory, checkpoint):

    specs_filename = os.path.join(experiment_directory, "specs.json")

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))

    clamping_function = None
    if specs["NetworkArch"] == "deep_sdf_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])
    elif specs["NetworkArch"] == "deep_implicit_template_decoder":
        clamping_function = lambda x : torch.clamp(x, -specs["ClampingDistance"], specs["ClampingDistance"])

    latent_vectors = ws.load_pre_trained_latent_vectors(experiment_directory, checkpoint)
    print(latent_vectors.shape)

    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    for dataset in train_split:
        for class_name in train_split[dataset]:
            class_name_current = class_name


    data_source = specs["DataSource"]

    index_list, instance_filenames = get_instance_filenames(data_source, train_split)
    latent_vectors = latent_vectors[torch.LongTensor(index_list)]

    print(f"{len(instance_filenames)} vs {len(latent_vectors)}")

    assert (len(instance_filenames) == len(latent_vectors)) 
    embedding_dict = {}
    for instance_id, latent_vector in zip(instance_filenames, latent_vectors):
        instance_id = instance_id.split("/")[-1].split(".")[0]
        embedding_dict[instance_id] = latent_vector.numpy()
    
    with open(f"./{class_name_current}_embedding.pkl", 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=4)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Use a trained DeepSDF decoder to generate a mesh given a latent code.")
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True, help="The experiment directory which includes specifications and saved model files to use for reconstruction")
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="latest", help="The checkpoint weights to use. This can be a number indicated an epoch or 'latest' for the latest weights (this is the default)")
    args = arg_parser.parse_args()
    extract_embedding(args.experiment_directory, args.checkpoint)
