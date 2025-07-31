### Evalute mesh, normal and depth 

import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser, Namespace

def read_point_cloud(path_cloud):
    assert os.path.exists(path_cloud)
    cloud = o3d.io.read_point_cloud(path_cloud)
    return cloud

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def visualize_error_map(pcd_pred, dist2, errormap_path, error_bound=0.10):
    from matplotlib import cm

    dist_score = dist2.clip(0, error_bound) / error_bound
    color_map = cm.get_cmap('Reds')
    colors = color_map(dist_score)[:, :3]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(errormap_path, pcd_pred)
    print(f'Visualize Error map of pcd to {errormap_path}\n')

def evaluate_mesh(mesh_gt, mesh_pred, 
                  threshold=.05, down_sample=.02, show_errormap=False):
    """ Borrowed from NeuralRecon and NeuRIS
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics
    """
    print(f"eval mesh using {threshold} threshold and {down_sample} down_sample")

    pcd_pred = read_point_cloud(mesh_pred)
    pcd_trgt = read_point_cloud(mesh_gt)
    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
    ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)

    precision = np.mean((dist2 < threshold).astype('float'))
    recall = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recall / (precision + recall)
    chamfer= np.mean(dist1**2)+np.mean(dist2**2)
    metrics = {
        'dist1': round(np.mean(dist2), 3),  # pred->gt
        'dist2': round(np.mean(dist1), 3),  # gt -> pred
        'precision': round(precision, 3),
        'recall': round(recall, 3),
        'fscore': round(fscore, 3),
        'chamfer': round(chamfer, 3),
    }

    print(f"fscore: {fscore:.3f}; chamfer: {chamfer:.3f}")
    if show_errormap:
        visualize_error_map(pcd_pred, dist2, mesh_pred.replace('.ply', '_error.ply'))
        
    with open(mesh_pred.replace('.ply', '.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--gt", type=str, required=True, help='path of dataset')
    parser.add_argument('--pred', type=str, required=True, default='./logs/scene0050_00/mesh_filtered.ply')
    args = parser.parse_args()

    evaluate_mesh(args.gt, args.pred, show_errormap=True)
