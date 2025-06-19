import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from utils.svo_helpers.render_helpers import get_scores, eval_points

rays_dir = None

import open3d as o3d
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from skimage.measure import marching_cubes
from scipy.spatial import cKDTree

class MeshExtractor:
    def __init__(self, voxel_size=0.1):
        self.voxel_size = voxel_size  # 0.2
        self.rays_d = None
        self.depth_points = None

    @torch.no_grad()
    def downsample_points(self, points, voxel_size=0.01):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd.points)

    @torch.no_grad()
    def get_valid_points(self, frame_poses, depth_maps):
        if isinstance(frame_poses, list):
            all_points = []
            print("extracting all points")
            for i in range(0, len(frame_poses), 5):
                pose = frame_poses[i]
                depth = depth_maps[i]
                points = self.rays_d * depth.unsqueeze(-1)
                points = points.reshape(-1, 3)
                points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
                if len(all_points) == 0:
                    all_points = points.detach().cpu().numpy()
                else:
                    all_points = np.concatenate(
                        [all_points, points.detach().cpu().numpy()], 0)
            print("downsample all points")
            all_points = self.downsample_points(all_points)
            return all_points
        else:
            pose = frame_poses
            depth = depth_maps
            points = self.rays_d * depth.unsqueeze(-1)
            points = points.reshape(-1, 3)
            points = points @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
            if self.depth_points is None:
                self.depth_points = points.detach().cpu().numpy()
            else:
                self.depth_points = np.concatenate(
                    [self.depth_points, points], 0)
            self.depth_points = self.downsample_points(self.depth_points)
        return self.depth_points

    @torch.no_grad()
    def create_mesh(self, decoder, map_states, voxel_size, voxels,
                    frame_poses=None, depth_maps=None, clean_mesh=False,
                    require_color=False, offset=-10, res=8):

        sdf_grid = get_scores(decoder, map_states, voxel_size, bits=res)  # (num_voxels,res*3,4)
        sdf_grid = sdf_grid.reshape(-1, res, res, res, 1)

        voxel_centres = map_states["voxel_center_xyz"]
        verts, faces = self.marching_cubes(voxel_centres, sdf_grid)

        if clean_mesh:
            print("********** get points from frames **********")
            all_points = self.get_valid_points(frame_poses, depth_maps)
            print("********** construct kdtree **********")
            kdtree = cKDTree(all_points)
            print("********** query kdtree **********")
            point_mask = kdtree.query_ball_point(
                verts, voxel_size * 0.5, workers=12, return_length=True)
            print("********** finished querying kdtree **********")
            point_mask = point_mask > 0
            face_mask = point_mask[faces.reshape(-1)].reshape(-1, 3).any(-1)

            faces = faces[face_mask]

        if require_color and decoder is not None:
            print("********** get color from network **********")
            verts_torch = torch.from_numpy(verts).float().cuda()
            batch_points = torch.split(verts_torch, 1000)
            colors = []
            for points in batch_points:
                # voxel_pos = points // self.voxel_size
                voxel_pos = torch.div(points, self.voxel_size, rounding_mode='trunc')
                batch_voxels = voxels[:, :3].cuda()
                batch_voxels = batch_voxels.unsqueeze(
                    0).repeat(voxel_pos.shape[0], 1, 1)

                # filter outliers
                nonzeros = (batch_voxels == voxel_pos.unsqueeze(1)).all(-1)
                nonzeros = torch.where(nonzeros, torch.ones_like(
                    nonzeros).int(), -torch.ones_like(nonzeros).int())
                sorted, index = torch.sort(nonzeros, dim=-1, descending=True)
                sorted = sorted[:, 0]
                index = index[:, 0]
                valid = (sorted != -1)
                color_empty = torch.zeros_like(points)
                points = points[valid, :]
                index = index[valid]

                # get color
                if len(points) > 0:
                    color = eval_points(decoder, points).cuda()
                    color_empty[valid] = color.float()
                colors += [color_empty]
            colors = torch.cat(colors, 0)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts + offset)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        if require_color:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                colors.detach().cpu().numpy())
        mesh.compute_vertex_normals()
        return mesh

    @torch.no_grad()
    def marching_cubes(self, voxels, sdf):
        voxels = voxels[:, :3]  # (num_voxels,3)
        sdf = sdf[..., 0]  # extract sdf
        res = 1.0 / (sdf.shape[1] - 1)  # 1/(8-1)
        spacing = [res, res, res]

        num_verts = 0
        total_verts = []
        total_faces = []
        for i in range(len(voxels)):
            sdf_volume = sdf[i].detach().cpu().numpy()  # (res,res,res)
            if np.min(sdf_volume) > 0 or np.max(sdf_volume) < 0:
                continue
            try:
                verts, faces, _, _ = marching_cubes(sdf_volume, 0, spacing=spacing)
            except:
                continue
            verts -= 0.5
            verts *= self.voxel_size
            verts += voxels[i].detach().cpu().numpy()
            faces += num_verts
            num_verts += verts.shape[0]

            total_verts += [verts]
            total_faces += [faces]
        total_verts = np.concatenate(total_verts)
        total_faces = np.concatenate(total_faces)
        return total_verts, total_faces


class RGBDFrame(nn.Module):
    def __init__(self, fid, rgb, depth, K, offset=10, ref_pose=None) -> None:
        super().__init__()
        self.stamp = fid
        self.h, self.w = depth.shape
        if type(rgb) != torch.Tensor:
            rgb = torch.FloatTensor(rgb).cuda()
        if type(depth) != torch.Tensor:
            depth = torch.FloatTensor(depth).cuda()  # / 2
        self.rgb = rgb.cuda()
        self.depth = depth.cuda()
        self.K = K

        if ref_pose is not None:
            if len(ref_pose.shape) != 2:
                ref_pose = ref_pose.reshape(4, 4)
            if type(ref_pose) != torch.Tensor:  # from gt data
                self.ref_pose = torch.tensor(ref_pose, requires_grad=False, dtype=torch.float32)
                self.ref_pose[:3, 3] += offset  # Offset ensures voxel coordinates>0
            else:  # from tracked data
                self.ref_pose = ref_pose.clone().requires_grad_(False)
        else:
            self.ref_pose = None
        self.precompute()

    def get_ref_pose(self):
        return self.ref_pose

    def get_ref_translation(self):
        return self.ref_pose[:3, 3]

    def get_ref_rotation(self):
        return self.ref_pose[:3, :3]

    @torch.no_grad()
    def get_rays(self, w=None, h=None, K=None):
        w = self.w if w == None else w
        h = self.h if h == None else h
        if K is None:
            K = np.eye(3)
            K[0, 0] = self.K[0, 0] * w / self.w
            K[1, 1] = self.K[1, 1] * h / self.h
            K[0, 2] = self.K[0, 2] * w / self.w
            K[1, 2] = self.K[1, 2] * h / self.h
        ix, iy = torch.meshgrid(
            torch.arange(w), torch.arange(h), indexing='xy')
        rays_d = torch.stack(
            [(ix - K[0, 2]) / K[0, 0],
             (iy - K[1, 2]) / K[1, 1],
             torch.ones_like(ix)], -1).float()  # camera coordinate
        return rays_d

    @torch.no_grad()
    def precompute(self):
        global rays_dir
        if rays_dir is None:
            rays_dir = self.get_rays(K=self.K).cuda()
        self.rays_d = rays_dir
        self.points = self.rays_d * self.depth[..., None]
        self.valid_mask = self.depth > 0

    @torch.no_grad()
    def get_points(self):
        return self.points[self.valid_mask].reshape(-1, 3)  # [N,3]

    @torch.no_grad()
    def sample_rays(self, N_rays):
        def sample_rays_utils(mask, num_samples):
            B, H, W = mask.shape
            mask_unfold = mask.reshape(-1)
            indices = torch.rand_like(mask_unfold).topk(num_samples)[1]
            sampled_masks = (torch.zeros_like(
                mask_unfold).scatter_(-1, indices, 1).reshape(B, H, W) > 0)
            return sampled_masks

        self.sample_mask = sample_rays_utils(
            torch.ones_like(self.depth)[None, ...], N_rays)[0, ...]

class Mapping:
    def __init__(self, voxel_size = 0.01, sdf_trunc = 0.02, depth_trunc=3, num_vertexes = 200000):
        torch.classes.load_library(
            "submodules/sparse_octree/build/lib.linux-x86_64-cpython-39/svo.cpython-39-x86_64-linux-gnu.so")

        self.voxel_size = voxel_size
        self.sdf_truncation = sdf_trunc
        self.depth_truncation = depth_trunc
        self.inflate_margin_ratio = 0.1
        self.offset=10

        # initialize svo
        self.voxel_initialized = torch.zeros(num_vertexes).cuda().bool()
        self.vertex_initialized = torch.zeros(num_vertexes).cuda().bool()

        self.sdf_priors = torch.zeros(
            (num_vertexes, 1),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))

        self.sdf_weights =torch.zeros(
            (num_vertexes, 1),
            requires_grad=False, dtype=torch.int8,
            device=torch.device("cuda"))

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, int(num_vertexes), voxel_size)  # Must be a multiple of 2

    def updownsampling_voxel(self, points, indices, counts):
        summed_elements = torch.zeros(counts.shape[0], points.shape[-1]).cuda()
        summed_elements = torch.scatter_add(summed_elements, dim=0,
                                            index=indices.unsqueeze(1).repeat(1, points.shape[-1]), src=points)
        updownsample_points = summed_elements / counts.unsqueeze(-1).repeat(1, points.shape[-1])
        return updownsample_points

    def create_voxels(self, frame):
        points_raw = frame.get_points().cuda()
        pose = frame.get_ref_pose().cuda()

        points = points_raw @ pose[:3, :3].transpose(-1, -2) + pose[:3, 3]  # change to world frame (Rx)^T = x^T R^T

        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')  # Divides each element

        inflate_margin_ratio = self.inflate_margin_ratio

        voxels_raw, inverse_indices, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)

        voxels_vaild = voxels_raw[counts > 10]
        self.voxels_vaild = voxels_vaild
        offsets = torch.LongTensor([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]).to(
            voxels.device)

        updownsampling_points = self.updownsampling_voxel(points, inverse_indices, counts)
        for offset in offsets:
            offset_axis = offset.nonzero().item()
            if offset[offset_axis] > 0:
                margin_mask = updownsampling_points[:, offset_axis] % self.voxel_size > (
                        1 - inflate_margin_ratio) * self.voxel_size
            else:
                margin_mask = updownsampling_points[:,
                              offset_axis] % self.voxel_size < inflate_margin_ratio * self.voxel_size
            margin_vox = voxels_raw[margin_mask * (counts > 10)]
            voxels_vaild = torch.cat((voxels_vaild, torch.clip(margin_vox + offset, min=0)), dim=0)

        voxels_unique = torch.unique(voxels_vaild, dim=0)
        self.seen_voxel = voxels_unique
        self.current_seen_voxel = voxels_unique.shape[0]
        voxels_svo, children_svo, vertexes_svo, svo_mask, svo_idx = self.svo.insert(voxels_unique.cpu().int())
        svo_mask = svo_mask[:, 0].bool()
        voxels_svo = voxels_svo[svo_mask]
        children_svo = children_svo[svo_mask]
        vertexes_svo = vertexes_svo[svo_mask]

        self.octant_idx = svo_mask.nonzero().cuda()
        self.svo_idx = svo_idx
        self.update_grid(voxels_svo, children_svo, vertexes_svo, svo_idx)

    @torch.enable_grad()
    def update_grid(self, voxels, children, vertexes, svo_idx):

        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        children = torch.cat([children, voxels[:, -1:]], -1)

        centres = centres.cuda().float()
        children = children.cuda().int()

        map_states = {}
        map_states["voxels"] = voxels.cuda()
        map_states["voxel_vertex_idx"] = vertexes.cuda()
        map_states["voxel_center_xyz"] = centres.cuda()
        map_states["voxel_structure"] = children.cuda()
        map_states["sdf_priors"] = self.sdf_priors
        map_states["sdf_weights"] = self.sdf_weights
        map_states["svo_idx"] = svo_idx.cuda()

        self.map_states = map_states

    def back_project(self, centers, c2w, K, depth, truncation):
        """
        Back-project the center point of the voxel into the depth map.
        Transform the obtained depth in the camera coordinate system to the world
        
        Args:
            centers (tensor, num_voxels*3): voxel centers
            c2w (tensor, 4*4): camera coordinate to world coordinate.
            K (array, 3*3): camera reference
            depth (tensor, w*h): depth ground true.
            truncation (float): truncation value.
        Returns:
            initsdf (tensor,num_voxels): Each vertex of the voxel corresponds to the depth value of the depth map,
                                            if it exceeds the boundary, it will be 0
            seen_iter_mask (tensor,num_voxels): True if two points match
            (1).The voxel is mapped to the corresponding pixel in the image, and does not exceed the image boundary
            (2).The initialized sdf value should be within the cutoff distance.
        """
        H, W = depth.shape
        w2c = torch.linalg.inv(c2w.float())
        K = torch.from_numpy(K).cuda()
        ones = torch.ones_like(centers[:, 0]).reshape(-1, 1).float()
        homo_points = torch.cat([centers, ones], dim=-1).unsqueeze(-1).float()
        homo_cam_points = w2c @ homo_points  # (N,4,1) = (4,4) * (N,4,1)
        cam_points = homo_cam_points[:, :3]  # (N,3,1)
        uv = K.float() @ cam_points.float()
        z = uv[:, -1:] + 1e-8
        uv = uv[:, :2] / z  # (N,2)
        uv = uv.round()
        cur_mask_seen = (uv[:, 0] < W) & (uv[:, 0] > 0) & (uv[:, 1] < H) & (uv[:, 1] > 0)
        cur_mask_seen = (cur_mask_seen & (z[:, :, 0] > 0)).reshape(-1)  # (N_mask,1) -> (N_mask)
        uv = (uv[cur_mask_seen].int()).squeeze(-1)  # (N_mask,2)
        depth = depth.transpose(-1, -2)  # (W,H)

        initsdf = torch.zeros((centers.shape[0], 1), device=centers.device)

        voxel_depth = torch.index_select(depth, dim=0, index=uv[:, 0]).gather(dim=1, index=uv[:, 1].reshape(-1,
                                                                                                            1).long())  # (N_mask,1)

        initsdf[cur_mask_seen] = (voxel_depth - cam_points[cur_mask_seen][:, 2]) / truncation  # (N,1)
        seen_iter_mask = cur_mask_seen

        return initsdf.squeeze(-1), seen_iter_mask


    @torch.no_grad()
    def initemb_sdf(self, frame, map_states, truncation, voxel_size=None, octant_idx=None, voxel_initialized=None,
                    vertex_initialized=None):
        vertexes = map_states["voxel_vertex_idx"]
        centers = map_states["voxel_center_xyz"]

        sdf_priors = map_states["sdf_priors"]
        sdf_weights = map_states["sdf_weights"]
        
        novertexes_mask = ~(vertexes.eq(-1).any(-1)) # 代表真实几何的vertex索引

        depth = frame.depth
        K = frame.K  # (3,3)
        c2w = frame.get_ref_pose().cuda()  # (4,4)

        octant_idx = octant_idx[novertexes_mask][:, 0] # 有效的svo索引
        uninit_idx = ~ voxel_initialized[octant_idx.long()] # 待初始化体素的全局（200000个）索引
        centers = centers[novertexes_mask]  # 待初始化体素的中心点
        vertexes = vertexes[novertexes_mask, :]  # 待初始化体素的8个顶点

        """
        vert_cord relative to voxel_cord: 
                [[-1., -1., -1.],
                [-1., -1.,  1.],
                [-1.,  1., -1.],
                [-1.,  1.,  1.],
                [ 1., -1., -1.],
                [ 1., -1.,  1.],
                [ 1.,  1., -1.],
                [ 1.,  1.,  1.]]
        """
        cut_x = cut_y = cut_z = torch.linspace(-1, 1, 2)
        cut_xx, cut_yy, cut_zz = torch.meshgrid(cut_x, cut_y, cut_z, indexing='ij')
        offsets = torch.stack([cut_xx, cut_yy, cut_zz], dim=-1).int().reshape(-1, 3).to(centers.device)

        centers_vert = (centers.unsqueeze(1) + offsets * (voxel_size / 2)).reshape(-1, 3)
        initsdf_vert, seen_iter_mask_vert = self.back_project(centers_vert, c2w, K, depth, truncation)

        occ_mask = torch.ones(centers.shape[0]).to(centers.device).bool()
        occ_mask[((initsdf_vert.reshape(-1, 8) * truncation).abs() > math.sqrt(6) * voxel_size).any(-1)] = False
        occ_mask = occ_mask * (seen_iter_mask_vert.reshape(-1, 8)).all(-1) #可见的mask

        valid_vertices = vertexes[occ_mask].reshape(-1).to(torch.long)
        valid_sdf_values = initsdf_vert.reshape(-1, 8)[occ_mask].reshape(-1)

        # tsdf fusion
        if valid_vertices.numel() > 0:
            # 计算当前观测的权重（基于距离的权重）
            current_weights = 1
            
            # 计算当前观测的权重
            prev_sdf = sdf_priors[valid_vertices, 0]
            prev_weights = sdf_weights[valid_vertices, 0]
            
            # TSDF融合
            new_weights = prev_weights + current_weights
            new_weights = torch.clamp(new_weights, max=100.0)
            
            # 加权平均
            fused_sdf = (prev_sdf * prev_weights + valid_sdf_values * current_weights) / new_weights
            
            # 更新SDF值和权重
            sdf_priors[valid_vertices, 0] = fused_sdf
            sdf_weights[valid_vertices, 0] = new_weights.to(torch.int8)
            vertex_initialized[valid_vertices] = True

            # 更新体素初始化状态
            voxel_initialized[octant_idx[occ_mask].long()] = True

            map_states["sdf_priors"] = sdf_priors
            map_states["sdf_weights"] = sdf_weights

        return map_states, voxel_initialized, vertex_initialized

    def mapping_step(self, frame_id, frame):

        ######################
        self.create_voxels(frame)

        self.map_states, self.voxel_initialized, self.vertex_initialized = self.initemb_sdf(frame,
                                                                                        self.map_states,
                                                                                        self.sdf_truncation,
                                                                                        voxel_size=self.voxel_size,
                                                                                        voxel_initialized=self.voxel_initialized,
                                                                                        octant_idx=self.octant_idx,
                                                                                        vertex_initialized=self.vertex_initialized)
    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False, require_color=False, map_states=None):
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)  # remove no smallest voxel
        voxels = voxels[~index.cpu(), :]
        vertexes = vertexes[~index.cpu(), :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = vertexes.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["sdf_priors"] = self.sdf_priors

        mesh = self.mesher.create_mesh(
            None, encoder_states, self.voxel_size, voxels,
            frame_poses=None, depth_maps=None,
            clean_mesh=clean_mesh, require_color=require_color, offset=-self.offset, res=res)
        return mesh


    @torch.no_grad()
    def extract_voxels(self, map_states=None):
        vertexes = map_states["voxel_vertex_idx"]
        voxels = map_states["voxels"]

        index = vertexes.eq(-1).any(-1)
        voxels = voxels[~index.cpu(), :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
                 self.voxel_size - self.offset
        return voxels

    @torch.no_grad()
    def export_voxel_cubes(self, map_states=None, filename="voxel_cubes.ply"):
        """
        将voxel导出为立方体网格并保存为PLY文件
        """
        # 获取voxel中心点
        voxel_centers = self.extract_voxels(map_states)
        centers_np = voxel_centers.detach().cpu().numpy()
        
        print(f"Exporting {len(centers_np)} voxels as cubes...")
        
        all_vertices = []
        all_triangles = []
        vertex_count = 0
        
        half_size = self.voxel_size / 2
        
        # 立方体的8个顶点（相对于中心的偏移）
        cube_offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * half_size
        
        # 立方体的12个三角形面
        cube_faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 1], [1, 4, 5],
            [2, 6, 3], [3, 6, 7],
            [0, 3, 4], [3, 7, 4],
            [1, 5, 2], [2, 5, 6]
        ])
        
        for i, center in enumerate(centers_np):
            cube_vertices = center + cube_offsets
            all_vertices.append(cube_vertices)
            cube_triangles = cube_faces + vertex_count
            all_triangles.append(cube_triangles)
            vertex_count += 8
        
        all_vertices = np.concatenate(all_vertices, axis=0)
        all_triangles = np.concatenate(all_triangles, axis=0)
        
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
        mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        mesh.compute_vertex_normals()
        
        success = o3d.io.write_triangle_mesh(filename, mesh)
        if success:
            print(f"Voxel cubes saved to {filename}")
            print(f"Total vertices: {len(all_vertices)}")
            print(f"Total triangles: {len(all_triangles)}")
        else:
            print(f"Failed to save voxel cubes to {filename}")
        
        return mesh

    @torch.no_grad()
    def export_voxel_points(self, map_states=None, filename="voxel_points.ply"):
        """
        将voxel中心导出为点云并保存为PLY文件
        """
        # 获取voxel中心点
        voxel_centers = self.extract_voxels(map_states)
        centers_np = voxel_centers.detach().cpu().numpy()
        
        print(f"Exporting {len(centers_np)} voxel centers as point cloud...")
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers_np)
        
        # 保存点云
        success = o3d.io.write_point_cloud(filename, pcd)
        if success:
            print(f"Voxel point cloud saved to {filename}")
            print(f"Total points: {len(centers_np)}")
        else:
            print(f"Failed to save voxel point cloud to {filename}")
        
        return pcd

    @torch.no_grad()
    def export_voxel_wireframes(self, map_states=None, filename="voxel_wireframes.ply"):
        """
        将voxel导出为线框并保存为PLY文件（无颜色）
        """
        if map_states is None:
            map_states = self.map_states
        
        # 获取voxel中心点
        voxel_centers = self.extract_voxels(map_states)
        centers_np = voxel_centers.detach().cpu().numpy()
        
        print(f"Exporting {len(centers_np)} voxels as wireframes...")
        
        all_points = []
        all_lines = []
        point_count = 0
        
        half_size = self.voxel_size / 2
        
        # 立方体的8个顶点（相对于中心的偏移）
        cube_offsets = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * half_size
        
        # 立方体的12条边
        cube_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        
        for i, center in enumerate(centers_np):
            cube_vertices = center + cube_offsets
            all_points.append(cube_vertices)
            cube_lines = cube_edges + point_count
            all_lines.append(cube_lines)
            point_count += 8
        
        all_points = np.concatenate(all_points, axis=0)
        all_lines = np.concatenate(all_lines, axis=0)
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(all_lines)
        
        # 保存为PLY文件（注意：不是所有软件都支持线条的PLY格式）
        success = o3d.io.write_line_set(filename, line_set)
        if success:
            print(f"Voxel wireframes saved to {filename}")
            print(f"Total points: {len(all_points)}")
            print(f"Total lines: {len(all_lines)}")
        else:
            print(f"Failed to save voxel wireframes to {filename}")
        
        return line_set

    @torch.no_grad()
    def visualize_voxels(self, map_states=None, method='cubes', show_interactive=False):
        """
        可视化voxel的便捷函数
        
        Args:
            map_states: 地图状态
            method: 'cubes' 或 'points'
            save_file: 是否保存文件
            show_interactive: 是否显示交互式窗口
        """
        if map_states is None:
            map_states = self.map_states
        
        if method == 'cubes':
            mesh = self.export_voxel_cubes(map_states, "voxel_cubes.ply")
            result_tmp = mesh
            
        elif method == 'points':
            pcd = self.export_voxel_points(map_states, "voxel_points.ply")
            result_tmp = pcd
        
        elif method == 'wireframes':
            line_set = self.export_voxel_wireframes(map_states, "voxel_wireframes.ply")
            result_tmp = line_set

        if show_interactive:
            # 添加坐标系
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            o3d.visualization.draw_geometries([result_tmp, coord_frame],
                                            window_name="Sparse Voxel Octree", 
                                            width=1200, height=800)
            

    def run(self, cameras_o3d, rgbmaps, depthmaps):
        progress_bar = tqdm(range(0, len(cameras_o3d)), position=0)
        progress_bar.set_description("TSDF integration progress")
        
        print('TSDFfusion started!')
        for frame_id in progress_bar:
            rgb = rgbmaps[frame_id].permute(1, 2, 0)
            depth = depthmaps[frame_id].squeeze(0)
            cam_o3d = cameras_o3d[frame_id]

            frame = RGBDFrame(frame_id, rgb, depth, cam_o3d.intrinsic.intrinsic_matrix, offset=self.offset, ref_pose=np.linalg.inv(cam_o3d.extrinsic))
            if frame.ref_pose.isinf().any():
                    continue

            self.mapping_step(frame_id, frame)

        print('TSDFfusion finished!')
        
        print('extracting mesh...')
        self.mesher = MeshExtractor(self.voxel_size)
        mesh = self.extract_mesh(map_states=self.map_states)

        print('extracting voxels...')
        voxel_mesh = self.visualize_voxels(method='wireframes', show_interactive=False)
        
        return mesh
        
if __name__ == "__main__":
    #* debug in  H2mapping
    from importlib import import_module
    Dataset = import_module("src.dataset.scannet")
    data_stream = Dataset.DataLoader(data_path='../../Datasets/scannet/scene0169_00',
                                    use_gt=True,
                                    depth_scale=1000.0,
                                    crop=6,
                                    scale_factor=0,
                                    max_depth=10)
    mapper = Mapping(voxel_size=0.05, sdf_trunc=0.2, num_vertexes=500000)
    mapper.run(data_stream)
