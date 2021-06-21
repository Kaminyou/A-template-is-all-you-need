import os
import trimesh
import numpy as np
from scipy.spatial import cKDTree as KDTree
import argparse
import json

def compute_trimesh_chamfer(gt_points, gen_mesh, offset, scale, num_mesh_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.
    gt_points: trimesh.points.PointCloud of just poins, sampled from the surface (see
               compute_metrics.ply for more documentation)
    gen_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
              method (see compute_metrics.py for more)
    """

    gen_points_sampled = trimesh.sample.sample_surface(gen_mesh, num_mesh_samples)[0]

    gen_points_sampled = gen_points_sampled / scale - offset

    # only need numpy array of points
    # gt_points_np = gt_points.vertices
    gt_points_np = gt_points.vertices

    # one direction
    gen_points_kd_tree = KDTree(gen_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_np)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))

    # other direction
    gt_points_kd_tree = KDTree(gt_points_np)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(gen_points_sampled)
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer + gen_to_gt_chamfer

classes ={
    'planes':'02691156',
    'chairs':'03001627',
    'sofas':'04256520'
}

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Calculate chamber distance on the testing data")
    arg_parser.add_argument("--class_name", "-c", dest="class_name", type = str, required=True, help="planes/chairs/sofas")
    arg_parser.add_argument("--split_file", "-f", dest="split_file", type = str, default="./examples/splits", help="Spliting file folder")
    arg_parser.add_argument("--data_source", "-d", dest="data_source", type = str, required=True, help="Source folder")
    arg_parser.add_argument("--input_path", "-i", dest="input_path", type = str, required=True, help="The folder contains reconstruction ply")
    args = arg_parser.parse_args()

    # load testing split
    testing_set_split = os.path.join(args.split_file, f"sv2_{args.class_name}_test.json")
    object_class = classes[args.class_name]
    with open(testing_set_split) as f:
        split = json.load(f)

    split_list = split["ShapeNetV2"][object_class]
    split_len = len(split_list)
    print(f"Total {split_len} obj")

    chamfer_dist_list = []
    for idx, instance in enumerate(split_list):
        reconstructed_mesh_filename = os.path.join(args.input_path, f"{instance}.ply")
        ground_truth_samples_filename = os.path.join(args.data_source,"SurfaceSamples", "ShapeNetV2", object_class ,f"{instance}.ply")
        normalization_params_filename = os.path.join(args.data_source,"NormalizationParameters", "ShapeNetV2", object_class ,f"{instance}.npz")

        # check existence
        if not os.path.isfile(reconstructed_mesh_filename):
            continue
        if not os.path.isfile(ground_truth_samples_filename):
            continue
        if not os.path.isfile(normalization_params_filename):
            continue
        
        # load ply and npz
        print(f"Calculating for {instance}...              ", end="\r")
        reconstruction = trimesh.load(reconstructed_mesh_filename)
        ground_truth_points = trimesh.load(ground_truth_samples_filename)
        normalization_params = np.load(normalization_params_filename)

        chamfer_dist = compute_trimesh_chamfer(ground_truth_points, reconstruction, normalization_params["offset"], normalization_params["scale"])
        chamfer_dist_list.append(chamfer_dist)

    # STATS
    chamfer_dist_list = np.array(chamfer_dist_list)
    print("========= Chamfer distance =========                  ")
    print(f"Mean   = {chamfer_dist_list.mean():.6f}")
    print(f"Std    = {chamfer_dist_list.std():.6f}")
    print(f"Median = {np.median(chamfer_dist_list):.6f}")