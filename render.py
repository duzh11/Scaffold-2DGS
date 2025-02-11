#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
import open3d as o3d
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.mesh_utils import GaussianExtractor, post_process_mesh, clean_mesh_points_outside_bbox
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, args: ArgumentParser):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        train_dir = os.path.join(dataset.model_path, 'train', "ours_{}".format(scene.loaded_iter))
        test_dir = os.path.join(dataset.model_path, 'test', "ours_{}".format(scene.loaded_iter))        
        gaussExtractor = GaussianExtractor(gaussians, render, prefilter_voxel, pipeline, bg_color=bg_color)

        if not args.skip_train:
             print("\nexport training images ...")
             os.makedirs(train_dir, exist_ok=True)
             
             render_fps, visible_count_train, per_view_dict = gaussExtractor.reconstruction(scene.getTrainCameras())
             with open(os.path.join(dataset.model_path, 'train', "ours_{}".format(scene.loaded_iter), "per_view_count.json"), 'w') as fp:
                json.dump(per_view_dict, fp, indent=True)
             
             print(f'train FPS: \033[1;35m{render_fps:.5f}\033[0m')
             gaussExtractor.export_image(train_dir, far_plane = torch.tensor(args.far_plane))

        if not args.skip_test:
             print("\nexport testing images ...")
             os.makedirs(test_dir, exist_ok=True)
             
             render_fps, visible_count_test, per_view_dict = gaussExtractor.reconstruction(scene.getTestCameras())
             with open(os.path.join(dataset.model_path, 'test', "ours_{}".format(scene.loaded_iter), "per_view_count.json"), 'w') as fp:
                json.dump(per_view_dict, fp, indent=True) 
             
             print(f'test FPS: \033[1;35m{render_fps:.5f}\033[0m')
             gaussExtractor.export_image(test_dir, far_plane = torch.tensor(args.far_plane))
        
        if not args.skip_mesh:
            print("\nexport mesh ...")
            gaussExtractor.reconstruction(scene.getTrainCameras())

            # TSDF fusion using open3d
            if args.mesh_type == 'TSDF':
                if args.scene_type == "unbounded":
                    name = 'fuse_unbounded.ply'
                    mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
                
                elif args.scene_type == "bounded":
                    # * TSDF fusion without masks
                    if args.usingmask:
                        name = 'fuse_bounded_wmask.ply'
                    else:
                        name = 'fuse_bounded.ply'
                    depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                    tsdf_voxel = (depth_trunc / args.mesh_res) if args.tsdf_voxel < 0 else args.tsdf_voxel
                    sdf_trunc = 5.0 * tsdf_voxel if args.sdf_trunc < 0 else args.sdf_trunc
                    mesh = gaussExtractor.extract_mesh_bounded(voxel_size=tsdf_voxel, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc, usingmask=args.usingmask, source_path=scene.source_path)
                
                elif args.scene_type == "bounded_handmade":
                    if args.usingmask:
                        name = 'fuse_bounded_handmade_wmask.ply'
                    else:
                        name = 'fuse_bounded_handmade.ply'
                    depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
                    mesh = gaussExtractor.extract_mesh_bounded_handmade(resolution=args.mesh_res, depth_trunc=depth_trunc)
                
                else:
                    print(f"{args.scene_type} is supported!")
            
            # Poisson reconstruction
            elif args.mesh_type == 'poisson':
                name = 'fuse_poisson.ply'
                pcd, mesh = gaussExtractor.extract_mesh_poisson(poisson_depth=args.poisson_depth, total_points=2000000)
                o3d.io.write_point_cloud(os.path.join(train_dir, "./DepthAndNormalMapsPoisson_pcd.ply"), pcd)
            
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            print("mesh saved at {}".format(os.path.join(train_dir, name)))

            # delete points outside the bounding box of the ground truth mesh
            path_mesh_gt = os.path.join(scene.source_path, 'mesh.ply')
            if os.path.exists(path_mesh_gt):
                mesh = clean_mesh_points_outside_bbox(os.path.join(train_dir, name), path_mesh_gt, scale_bbox = args.scale_factor)

            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_type", default='TSDF', type=str, help='using TSDF or poisson reconstruction or ')
    parser.add_argument("--far_plane", type=float, default = 5.0)
    # TSDF
    parser.add_argument("--tsdf_voxel", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--scene_type", default="bounded", help='Mesh: bounded / bounded_handmade/ unbounded scene')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    parser.add_argument("--usingmask", action="store_true", help='Mesh: using mask for TSDF fusion')
    parser.add_argument("--scale_factor", default=1.2, type=float, help='Mesh: clean mesh outside of the bbox')
    # poisson reconsturction
    parser.add_argument("--poisson_depth", default=10.0, type=float, help='Mesh: Poisson Octree max depth')
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args)
