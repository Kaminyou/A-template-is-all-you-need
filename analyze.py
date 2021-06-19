from networks.encoder import Encoder
from networks.deep_implicit_template_decoder import Decoder
import deep_sdf
import deep_sdf.workspace as ws
import numpy as np
from numpy.linalg import norm
import torch
import torch.utils.data as data_utils
import os
import json
import tqdm
import pickle
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

def extract_instance_name_from_image_name(image_name):
    return image_name.split("/")[-4]

def difference_matrix(a):
    x = np.reshape(a, (len(a), 1))
    return x - x.transpose()

def extract_upper_tri(X):
    m = X.shape[0]
    r,c = np.triu_indices(m,1)
    return X[r,c]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Analyze the embedding.")
    arg_parser.add_argument("--experiment", "-e", dest="experiment_directory", required=True, help="The experiment directory which includes specifications and saved model files to use for reconstruction")
    arg_parser.add_argument("--pretrained_embedding", "-p", dest="pretrained_embedding", action="store_true", help="Analyze the pretrained embedding")
    arg_parser.add_argument("--data_source", "-d", dest="data_source", default="../ShapeNet/DeepSDF", help="Data source path")
    arg_parser.add_argument("--checkpoint", "-c", dest="checkpoint", default="latest", help="The checkpoint weights to use. This can be a number indicated an epoch or 'latest' for the latest weights (this is the default)")
    arg_parser.add_argument("--inference_batch_size", "-b", dest="inference_batch_size", default=64, help="Batch size for inference")
    arg_parser.add_argument("--batch_split", dest="batch_split", type = int, default=2, help="Batch split")
    arg_parser.add_argument("--early_stop", dest="early_stop", type = int, default=20, help="Set a value of batch iterations")
    arg_parser.add_argument("--thread", "-t", dest="thread", type = int, default=8, help="Thread")
    args = arg_parser.parse_args()

    specs = ws.load_experiment_specifications(args.experiment_directory)
    train_split_file = specs["TrainSplit"]
    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    for key in train_split["ShapeNetV2"].keys():
        instance_class = key
        break

    saving_path = os.path.join(args.experiment_directory, "plot")
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)

    if args.pretrained_embedding:
        with open(f"./pretrained_embedding/{instance_class}_embedding.pkl", 'rb') as f:
            data = pickle.load(f)
            
        keys = []
        embeddings = []
        for key, value in data.items():
            keys.append(key)
            embeddings.append(value)
        embeddings = np.array(embeddings)
        tsne_embedded = TSNE(n_components=2, init="pca").fit_transform(embeddings)
        plt.figure(figsize=(10,10))
        sns.scatterplot(x=tsne_embedded[:,0],y=tsne_embedded[:,1])

        x_max = tsne_embedded[:,0].mean() + 3 * tsne_embedded[:,0].std()
        x_min = tsne_embedded[:,0].mean() - 3 * tsne_embedded[:,0].std()
        y_max = tsne_embedded[:,1].mean() + 3 * tsne_embedded[:,1].std()
        y_min = tsne_embedded[:,1].mean() - 3 * tsne_embedded[:,1].std()
        for i in range(len(keys)):
            if not (x_min <= tsne_embedded[i,0] <= x_max) or not (y_min <= tsne_embedded[i,1] <= y_max):
                plt.text(tsne_embedded[i,0] + 1, tsne_embedded[i,1], keys[i][:6], horizontalalignment='left', size='medium', color='black')
        plt.savefig(os.path.join(saving_path, "pretrained_embedding_distribution.png"))
        print(f"Generate plot at {os.path.join(saving_path, 'pretrained_embedding_distribution.png')}")

    else:
        latent_size = specs["CodeLength"]

        encoder = Encoder(latent_size=latent_size)
        lat_epoch = ws.load_encoder_parameters(args.experiment_directory, args.checkpoint, encoder)

        decoder = Decoder(specs["CodeLength"], **specs["NetworkSpecs"])
        decoder = torch.nn.DataParallel(decoder)
        model_epoch = ws.load_model_parameters(args.experiment_directory, args.checkpoint, decoder)

        train_split_file = specs["TrainSplit"]
        num_samp_per_scene = specs["SamplesPerScene"]
        batch_split = args.batch_split
        scene_per_split = args.inference_batch_size // batch_split

        clamp_dist = specs["ClampingDistance"]
        minT = -clamp_dist
        maxT = clamp_dist
        enforce_minmax = True

        with open(train_split_file, "r") as f:
            train_split = json.load(f)
        sdf_dataset = deep_sdf.data.SDFSamples(args.data_source, train_split, num_samp_per_scene, load_ram=False, level = 'easy', analysis_mode = True)
        sdf_loader = data_utils.DataLoader(sdf_dataset, batch_size=args.inference_batch_size, shuffle=False, num_workers=args.thread, drop_last=True)

        device = "cuda"
        keys = []
        embeddings = np.zeros((0,256))
        wrapping_diff_np = np.zeros(0)

        encoder.to(device)
        decoder.to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            loader = tqdm.tqdm(sdf_loader,desc="Processing...", position=0, leave=True)
            for bi, (sdf_data, images, indices, image_filenames) in enumerate(loader):
                sdf_data = sdf_data.reshape(-1, 4)  #(B*N, 4)

                num_sdf_samples = sdf_data.shape[0]  #(B*N)

                xyz = sdf_data[:, 0:3]                #(B*N, 3)   
                sdf_gt = sdf_data[:, 3].unsqueeze(1)  #(B*N)

                if enforce_minmax:
                    sdf_gt = torch.clamp(sdf_gt, minT, maxT)

                xyz = torch.chunk(xyz, batch_split)
                indices = torch.chunk(
                    indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1),
                    batch_split,
                )

                sdf_gt = torch.chunk(sdf_gt, batch_split)
                images = torch.chunk(images, batch_split)

                image_filenames = [image_filenames[scene_per_split*j:scene_per_split*(j+1)] for j in range(batch_split)]
                for i in range(batch_split):
                    input_imgs = images[i].to(device)
                    batch_vecs = encoder(input_imgs)
                    
                    keys += list(map(extract_instance_name_from_image_name, image_filenames[0]))
                    embeddings = np.vstack((embeddings, batch_vecs.clone().cpu().numpy()))
                    
                    xyz_ = xyz[i].to(device)
                    batch_vecs = batch_vecs.unsqueeze(1).repeat(1, num_samp_per_scene, 1).view(-1, latent_size)
                    input = torch.cat([batch_vecs, xyz_], dim=1)

                    p_final, x, warping_param = decoder(input, output_warped_points=True, output_warping_param=True)
                    
                    warpping_diff = torch.abs((xyz_.reshape(-1, num_samp_per_scene, 3) - p_final.reshape(-1, num_samp_per_scene, 3))).sum(axis=-1).sum(axis=-1)
                    wrapping_diff_np = np.concatenate((wrapping_diff_np, warpping_diff.cpu().numpy()))
                if bi == args.early_stop:
                    break
        print("TSNE Processing...", end="\r")
        tsne_embedded = TSNE(n_components=2, init="pca", n_jobs=args.thread).fit_transform(embeddings)
        plt.figure(figsize=(20,20))
        sns.scatterplot(x=tsne_embedded[:,0],y=tsne_embedded[:,1], hue=keys)
        plt.legend(loc='upper left',bbox_to_anchor=(1,1))
        plt.savefig(os.path.join(saving_path, "training_embedding_distribution.png"))
        plt.clf()
        print(f"Generate plot at {os.path.join(saving_path, 'training_embedding_distribution.png')}")

        cosine_similarity_matrix = cosine_similarity(embeddings)
        difference_mtx = np.abs(difference_matrix(wrapping_diff_np))
        print("==== STAT ====")
        print(f"COS MAX = {cosine_similarity_matrix.max():.5f}")
        print(f"COS MIN = {cosine_similarity_matrix.min():.5f}")
        print(f"COS AVG = {cosine_similarity_matrix.mean():.5f}")

        cosine_similarity_matrix_triu = extract_upper_tri(cosine_similarity_matrix)
        difference_mtx_triu = extract_upper_tri(difference_mtx)

        plt.figure(figsize=(10,10))
        plt.scatter(cosine_similarity_matrix_triu, difference_mtx_triu, s=1)
        plt.xlabel("cosine similarity")
        plt.ylabel("difference in xyz")
        plt.savefig(os.path.join(saving_path, "similiarity_distribution.png"))
        print(f"Generate plot at {os.path.join(saving_path, 'similiarity_distribution.png')}")