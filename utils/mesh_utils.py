#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
import time
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

import torchvision
import utils.vis_utils as VISUils
from utils.camera_utils import pick_indices_at_random, get_colored_points_from_depth

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, prefilter_voxel, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.prefilter_voxel = partial(prefilter_voxel, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.rgbmaps = []
        self.alphamaps = []
        self.depthmaps = []
        self.normals = []
        self.depth_normals = []
        self.viewpoint_stack = []
        # render expected/median depthmaps
        self.expected_depthmaps = []
        self.median_depthmaps = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        t_list = []
        visible_count_list = []
        per_view_dict = {}

        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            
            ### render and compue fps
            torch.cuda.synchronize(); t0 = time.time()
            voxel_visible_mask = self.prefilter_voxel(viewpoint_cam, self.gaussians)
            render_pkg = self.render(viewpoint_cam, self.gaussians, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize(); t1 = time.time()

            t_list.append(t1-t0)

            rgb = render_pkg['render']
            visible_count = (render_pkg["radii"] > 0).sum()
            visible_count_list.append(visible_count)
            per_view_dict['{0:05d}'.format(i) + ".png"] = visible_count.item()

            ### other results
            alpha = render_pkg['rend_alpha']
            depth = render_pkg['surf_depth']
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)
            depth_normal = render_pkg['surf_normal']
            # render expected/median depthmaps
            expected_depth = render_pkg['expected_depth']
            median_depth = render_pkg['median_depth']
            
            self.rgbmaps.append(rgb.cpu())
            self.alphamaps.append(alpha.cpu())
            self.depthmaps.append(depth.cpu())
            self.normals.append(normal.cpu())
            self.depth_normals.append(depth_normal.cpu())
            # render expected/median depthmaps
            self.expected_depthmaps.append(expected_depth.cpu())
            self.median_depthmaps.append(median_depth.cpu())
        
        ### save results
        self.rgbmaps = torch.stack(self.rgbmaps, dim=0)
        self.alphamaps = torch.stack(self.alphamaps, dim=0)
        self.depthmaps = torch.stack(self.depthmaps, dim=0)
        self.normals = torch.stack(self.normals, dim=0)
        self.depth_normals = torch.stack(self.depth_normals, dim=0)
        # render expected/median depthmaps
        self.expected_depthmaps = torch.stack(self.expected_depthmaps, dim=0)
        self.median_depthmaps = torch.stack(self.median_depthmaps, dim=0)

        self.estimate_bounding_sphere()
        
        ### compute fps
        render_fps = 1.0 / (np.array(t_list[5:]).mean())

        return render_fps, visible_count_list, per_view_dict

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True, usingmask=False):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.
        
        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        alpha_thres = 0.5
        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]

            # * if require using mask and we have mask provided, use it
            if usingmask and mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0

            depth[self.alphamaps[i] < alpha_thres] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )

            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * 1
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # update weight
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def extract_mesh_poisson(self, poisson_depth=3, total_points=2000000, outlier_removal=True):
        """
        Idea: backproject depth and normal maps into 3D oriented point cloud -> Poisson
        copying from dnsplatter: https://github.com/maturk/dn-splatter
        """
        print("Running poisson reconstruction ...")
        num_frames = len(self.viewpoint_stack)
        samples_per_frame = (total_points + num_frames) // (num_frames)
        print("samples per frame: ", samples_per_frame)
        pcd_points = []
        pcd_normals = []
        pcd_colors = []
        alpha_thres = 0.5

        # get pcd
        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="poisson reconstruction progress"):
            # 2d rendering results
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i]
            normals = self.normals[i]

            # if we have mask provided, use it
            # if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
            #     depth[(viewpoint_cam.gt_alpha_mask < 0.5)] = 0
            depth[self.alphamaps[i] < alpha_thres] = 0

            # camera poses
            w2c = torch.tensor(cam_o3d.extrinsic).to(depth.device)
            c2w = w2c.inverse()
            H, W = depth.shape[1], depth.shape[2]

            # 3d points
            valid_mask = depth
            indices = pick_indices_at_random(valid_mask, samples_per_frame)
            if len(indices) == 0:
                    continue
            
            xyzs, rgbs = get_colored_points_from_depth(
                    depths=depth.permute(1,2,0),
                    rgbs=rgb.permute(1,2,0),
                    fx=cam_o3d.intrinsic.intrinsic_matrix[0,0],
                    fy=cam_o3d.intrinsic.intrinsic_matrix[1,1],
                    cx=cam_o3d.intrinsic.intrinsic_matrix[0,2],  # type: ignore
                    cy=cam_o3d.intrinsic.intrinsic_matrix[1,2],  # type: ignore
                    img_size=(W, H),
                    c2w=c2w,
                    mask=indices,
                )

            pcd_points.append(xyzs)
            pcd_colors.append(rgbs)
            normals = normals.permute(1,2,0).view(-1, 3)[indices]
            pcd_normals.append(normals)
        
        # pcd
        pcd_points = torch.cat(pcd_points, dim=0)
        pcd_colors = torch.cat(pcd_colors, dim=0)
        pcd_normals = torch.cat(pcd_normals, dim=0)

        pcd_points = pcd_points.cpu().numpy()
        pcd_colors = pcd_colors.cpu().numpy()
        pcd_normals = pcd_normals.cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        pcd.normals = o3d.utility.Vector3dVector(pcd_normals)

        if outlier_removal:
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            pcd = pcd.select_by_index(ind)

        # poisson reconstruction
        print("Poisson reconstruction... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(poisson_depth)
        )
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("remoing vertices by densities...")

        return pcd, mesh
      
    @torch.no_grad()
    def export_image(self, path, near_plane = 0.0, far_plane = 5.0):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        error_path = os.path.join(path, "errors")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(error_path, exist_ok=True)

        ### Optional: save depth/normal maps
        render_outputs = ['mask', 'expected_depth', 'median_depth', 'normal', 'depth2normal']
        render_dict = {
            'mask': self.alphamaps,
            'expected_depth': self.expected_depthmaps,
            'median_depth': self.median_depthmaps,
            'normal': self.normals,
            'depth2normal': self.depth_normals
        }

        outputs_path = []
        for output_idx in render_outputs:
            output_idx_path = os.path.join(path, f"renders_{output_idx}")
            os.makedirs(output_idx_path, exist_ok=True)
            outputs_path.append(output_idx_path)
        
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            errormap = np.abs(gt.permute(1,2,0).cpu().numpy() - self.rgbmaps[idx].permute(1,2,0).cpu().numpy())
            save_img_u8(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))

            ### Optional: save depth/normal maps
            render_dict = {
                'mask': self.alphamaps[idx],
                'expected_depth': self.expected_depthmaps[idx],
                'median_depth': self.median_depthmaps[idx],
                'normal': self.normals[idx],
                'depth2normal': self.depth_normals[idx]
            }

            for jdx, output_jdx in enumerate(render_outputs):
                render_output = render_dict[output_jdx]
                
                if 'mask' in output_jdx:
                    torchvision.utils.save_image(render_output, os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
                elif '_depth' in output_jdx:
                    np.savez(os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".npz"), np.array(render_output[0,...]))
                    render_output_map = VISUils.apply_depth_colormap(render_output[0,...,None], render_dict['mask'][0,...,None], near_plane = near_plane, far_plane = far_plane).detach()
                    torchvision.utils.save_image(render_output_map.permute(2,0,1), os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
                elif 'normal' in output_jdx:
                    np.savez(os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".npz"), np.array(render_output.permute(1,2,0)))
                    render_output_map = ((render_output+1)/2).clip(0, 1)
                    torchvision.utils.save_image(render_output_map, os.path.join(outputs_path[jdx], '{0:05d}'.format(idx) + ".png"))
                else:
                    pass
