from typing import Union
import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

from ash import HashSet, UnBoundedSparseDenseGrid, BoundedSparseDenseGrid, DotDict
import numpy as np
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c

from .data_provider import ImageDataset

def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def to_o3d_im(tensor):
    return o3d.t.geometry.Image(to_o3d(tensor))


class TSDFFusion:
    """Use ASH's hashgrid to generate differentiable sparse-dense grid from RGB + scaled monocular depth prior"""

    device = torch.device("cuda:0")

    def __init__(
        self,
        grid: Union[UnBoundedSparseDenseGrid, BoundedSparseDenseGrid],
        dilation: int,
    ):
        self.grid = grid
        self.voxel_size = self.grid.cell_size

        self.dilation = dilation
        self.trunc = self.dilation * self.voxel_size * self.grid.grid_dim

    @torch.no_grad()
    def fuse_dataset(self, dataset, step=1):
        pbar = tqdm(range(0, dataset.num_images, step))
        for i in pbar:
            pbar.set_description(f"Fuse frame {i}")
            datum = DotDict(dataset.get_image(i))
            for k, v in datum.items():
                if isinstance(v, np.ndarray):
                    datum[k] = torch.from_numpy(v.astype(np.float32)).to(self.device)
            if datum['pose'].isinf().any():
                    continue
            self.fuse_frame(datum)

    @torch.no_grad()
    def unproject_depth_to_points(
        self, depth, intrinsic, extrinsic, depth_scale, depth_max
    ):
        # Multiply back to make open3d happy
        depth_im = to_o3d_im(depth * depth_scale)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_im,
            to_o3d(intrinsic.contiguous().double()),
            to_o3d(extrinsic.contiguous().double()),
            depth_scale,
            depth_max,
        )
        if len(pcd.point["positions"]) == 0:
            warnings.warn("No points after unprojection")
            return None
        points = from_o3d(pcd.point["positions"])
        return points

    @torch.no_grad()
    def project_points_to_rgbd(
        self, points, intrinsic, extrinsic, color, depth, depth_max
    ):
        h, w, _ = color.shape
        xyz = points @ extrinsic[:3, :3].t() + extrinsic[:3, 3:].t()
        uvd = xyz @ intrinsic.t()

        # In bound validity
        d = uvd[:, 2]
        u = (uvd[:, 0] / uvd[:, 2]).round().long()
        v = (uvd[:, 1] / uvd[:, 2]).round().long()

        mask_projection = (d > 0) * (u >= 0) * (v >= 0) * (u < w) * (v < h)

        u_valid = u[mask_projection]
        v_valid = v[mask_projection]

        depth_readings = torch.zeros_like(d)
        depth_readings[mask_projection] = depth[v_valid, u_valid]

        color_readings = torch.zeros((len(d), 3), device=self.device)
        color_readings[mask_projection] = color[v_valid, u_valid, :]

        sdf = depth_readings - d
        rgb = color_readings

        mask_depth = (
            (depth_readings > 0) * (depth_readings < depth_max) * (sdf >= -self.trunc)
        )
        sdf[sdf >= self.trunc] = self.trunc

        weight = (mask_depth * mask_projection).float()

        return sdf, rgb, weight

    @torch.no_grad()
    def prune_by_weight_(self, grid_mean_weight_thr=1):
        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()

        batch_size = min(1000, len(grid_coords))

        for i in range(0, len(grid_coords), batch_size):
            grid_coords_batch = grid_coords[i : i + batch_size]
            grid_indices_batch = grid_indices[i : i + batch_size]

            weight = self.grid.embeddings[grid_indices_batch, cell_indices, 4]
            mask = weight.mean(dim=1) < grid_mean_weight_thr

            if mask.sum() > 0:
                self.grid.engine.erase(grid_coords_batch[mask].squeeze(1))
                self.grid.embeddings[grid_indices_batch[mask]] = 0
        self.grid.construct_grid_neighbor_lut_(radius=1, bidirectional=False)

    @torch.no_grad()
    def prune_by_mesh_connected_components_(self, ratio_to_largest_component=0.5):
        torch.cuda.empty_cache()
        sdf = self.grid.embeddings[..., 0].contiguous()
        weight = self.grid.embeddings[..., 4].contiguous()
        mesh = self.grid.marching_cubes(
            sdf,
            weight,
            vertices_only=False,
            color_fn=None,
            normal_fn=None,
            iso_value=0.0,
            weight_thr=1,
        )
        mesh = mesh.to_legacy()

        # Remove small connected components in mesh
        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = mesh.cluster_connected_triangles()

        triangle_clusters = np.array(triangle_clusters)
        cluster_n_triangles = np.array(cluster_n_triangles)
        largest_cluster_idx = cluster_n_triangles.argmax()
        largest_cluster_n_triangles = cluster_n_triangles[largest_cluster_idx]

        triangles_keep_mask = np.zeros_like(triangle_clusters, dtype=np.int32)
        saved_clusters = []
        for i, n_tri in enumerate(cluster_n_triangles):
            if n_tri > ratio_to_largest_component * largest_cluster_n_triangles:
                saved_clusters.append(i)
                triangles_keep_mask += triangle_clusters == i
        triangles_to_remove = triangles_keep_mask == 0
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()

        # Only keep sparse grids around surfaces
        xyz = torch.from_numpy(np.asarray(mesh.vertices)).to(self.device)

        dummy_grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=self.grid.num_embeddings,
            embedding_dim=1,  # dummy
            grid_dim=self.grid.grid_dim,
            cell_size=self.grid.cell_size,
            device=self.grid.device,
        )       

        # Use a small dilation around
        dummy_grid.spatial_init_(xyz, dilation=1, bidirectional=True)

        # Find the portion that need to be removed
        grid_coords_unpruned, _, _, _ = self.grid.items()
        grid_coords_kept, _, _, _ = dummy_grid.items()
        grid_coords_unpruned = grid_coords_unpruned.view(-1, 3)
        grid_coords_kept = grid_coords_kept.view(-1, 3)

        #* 这里必须要self.grid指定--voxel_size，即按照unbounded grid的方式来执行，否则prune出来的engine特别小（相当于全prune了）
        #* 原因在于marchingcube的时候transform_cell_to_world，而BoundedSparseDenseGrid限定grid的范围为0+，因为知道bbox_min，而UnboundedSpareseDenseGrid则没有
        hashset = HashSet(
            key_dim=3, capacity=int(len(grid_coords_unpruned) * 1.2), device=self.device
        )
        hashset.insert(grid_coords_kept)
        masks = hashset.find(grid_coords_unpruned)
        grid_coords_to_prune = grid_coords_unpruned[~masks]

        self.grid.engine.erase(grid_coords_to_prune)
        print("after pruning:", self.grid.engine.size())

    @torch.no_grad()
    def fuse_frame(self, datum):
        torch.cuda.empty_cache()
        datum.depth *= datum.depth_scale

        points = self.unproject_depth_to_points(
            datum.depth,
            datum.intrinsic,
            datum.extrinsic,
            1.0 / datum.depth_scale,  # open3d uses inverse depth scale
            datum.depth_max,
        )
        if points is None:
            return

        # Insertion
        (
            grid_coords,
            cell_coords,
            grid_indices,
            cell_indices,
        ) = self.grid.spatial_init_(points, dilation=self.dilation, bidirectional=True)
        if len(grid_indices) == 0:
            return

        cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)

        # Observation
        sdf, rgb, w = self.project_points_to_rgbd(
            cell_positions,
            datum.intrinsic,
            datum.extrinsic,
            datum.rgb,
            datum.depth,
            datum.depth_max,
        )

        # Fusion
        embedding = self.grid.embeddings[grid_indices, cell_indices]

        w_sum = embedding[..., 4:5]
        sdf_mean = embedding[..., 0:1]
        rgb_mean = embedding[..., 1:4]

        w = w.view(w_sum.shape)
        sdf = sdf.view(sdf_mean.shape)
        rgb = rgb.view(rgb_mean.shape)

        w_updated = w_sum + w
        sdf_updated = (sdf_mean * w_sum + sdf * w) / (w_updated + 1e-6)
        rgb_updated = (rgb_mean * w_sum + rgb * w) / (w_updated + 1e-6)

        embedding[..., 4:5] = w_updated
        embedding[..., 0:1] = sdf_updated
        embedding[..., 1:4] = rgb_updated
        self.grid.embeddings[grid_indices, cell_indices] = embedding

@torch.no_grad()
def draw_grids2cubes(grid_centers, voxel_size, file_name="sparse_grid_cubes.ply"):
    geometries = o3d.geometry.TriangleMesh()
    for center in grid_centers:
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        cube.translate(center.cpu().numpy() + voxel_size / 2)
        cube.paint_uniform_color([0.6, 0.8, 1.0])
        geometries += cube
    o3d.io.write_triangle_mesh(file_name, geometries)
    print("drawing grids to cubes, saved to", file_name)

@torch.no_grad()
def draw_grids2wireframes(grid_centers, voxel_size, file_name="sparse_grid_wireframes.ply"):
    all_points = []
    all_lines = []
    point_count = 0
    half_size = voxel_size / 2
    
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
    
    for center in grid_centers:
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

    o3d.io.write_line_set(file_name, line_set)
    print("drawing grids to wireframes, saved to", file_name)

@torch.no_grad()
def visualize_sparse_grid(grid, file_name="sparse_grid_cubes.ply", method="cube"):
    grid_coords, _, _, _ = grid.items()
    grid_coords = grid_coords.view(-1, 3).cpu().float()
    grid_centers = grid.transform_cell_to_world(grid_coords * grid.grid_dim)

    voxel_size = grid.cell_size * grid.grid_dim

    if method == "cube":
        draw_grids2cubes(grid_centers, voxel_size, file_name=file_name)
    elif method == "wireframe":
        draw_grids2wireframes(grid_centers, voxel_size, file_name=file_name)

@torch.no_grad()
def visualize_dense_grids(grid, file_name="dense_grid_cubes.ply", method="cube"):
    grid_coords, cell_coords, grid_indices, cell_indices = grid.items()
    grid_dim = grid.grid_dim
    cell_size = grid.cell_size

    # 获取所有cell的世界坐标
    all_cell_centers = []
    for i in range(len(grid_coords)):
        cell_xyz = grid.cell_to_world(grid_coords[i:i+1], torch.arange(grid_dim**3).view(-1,1,1).expand(-1,1,3))
        cell_xyz = cell_xyz.view(-1, 3).cpu().float()
        all_cell_centers.append(cell_xyz)
    all_cell_centers = torch.cat(all_cell_centers, dim=0)

    if method == "cube":
        draw_grids2cubes(all_cell_centers, cell_size, file_name=file_name)
    elif method == "wireframe":
        draw_grids2wireframes(all_cell_centers, cell_size, file_name=file_name)

if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num_grids", type=int, default=120000,
                        help="capacity of sparse grids")
    parser.add_argument("--voxel_size", type=float, default=-1)
    parser.add_argument("--depth_max", type=float, default=10.0, help="max depth value to truncate in meters")
    args = parser.parse_args()
    # fmt: on

    device = torch.device("cuda:0")

    if args.voxel_size > 0:
        print(
            f"Using metric voxel size {args.voxel_size}m with UnboundedSparseDenseGrid."
        )
        normalize_scene = False
        grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=args.num_grids,
            embedding_dim=5,  # 5 dims corresponds to SDF(1) + weight(1) + RGB(3)
            grid_dim=8,
            cell_size=args.voxel_size,
            device=device,
        )

    else:
        print(f"Using with BoundedSparseDenseGrid.")
        normalize_scene = True
        grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=args.num_grids,
            embedding_dim=5,
            grid_dim=8,
            sparse_grid_dim=32,
            bbox_min=-1 * torch.ones(3, device=device),
            bbox_max=torch.ones(3, device=device),
            device=device,
        )

    # Load data
    dataset = ImageDataset(
        args.path,
        depth_max=args.depth_max,
        normalize_scene=normalize_scene,
        generate_rays=True,
    )

    dilation = 1
    fuser = TSDFFusion(grid, dilation)
    fuser.fuse_dataset(dataset, step=1)
    visualize_sparse_grid(fuser.grid, file_name="ash/rgbd_fusion_examples/sparse_grid.ply", method="wireframe")
    print(f"hash map size after fusion: {fuser.grid.engine.size()}")

    # sdf_fn and normal_fn
    def color_fn(x):
        embeddings, masks = fuser.grid(x, interpolation="linear")
        return embeddings[..., 1:4].contiguous()

    def grad_fn(x):
        x.requires_grad_(True)
        embeddings, masks = fuser.grid(x, interpolation="linear")

        grad_x = torch.autograd.grad(
            outputs=embeddings[..., 0],
            inputs=x,
            grad_outputs=torch.ones_like(embeddings[..., 0], requires_grad=False),
            create_graph=True,
        )[0]
        return grad_x

    def normal_fn(x):
        return F.normalize(grad_fn(x), dim=-1).contiguous()

    sdf = fuser.grid.embeddings[..., 0].contiguous()
    weight = fuser.grid.embeddings[..., 4].contiguous()
    mesh = fuser.grid.marching_cubes(
        sdf,
        weight,
        vertices_only=False,
        color_fn=color_fn,
        normal_fn=normal_fn,
        iso_value=0.0,
        weight_thr=1,
    )
    o3d.io.write_triangle_mesh("ash/rgbd_fusion_examples/mesh.ply", mesh.to_legacy())

    fuser.prune_by_mesh_connected_components_(ratio_to_largest_component=0.5)
    visualize_sparse_grid(fuser.grid, file_name="ash/rgbd_fusion_examples/sparse_grid_pruned.ply", method="wireframe")

    # 7x7x7 gaussian filter
    fuser.grid.gaussian_filter_(size=7, sigma=0.1)

    sdf = fuser.grid.embeddings[..., 0].contiguous()
    weight = fuser.grid.embeddings[..., 4].contiguous()
    mesh_filtered = fuser.grid.marching_cubes(
        sdf,
        weight,
        vertices_only=False,
        color_fn=color_fn,
        normal_fn=normal_fn,
        iso_value=0.0,
        weight_thr=1,
    )
    o3d.io.write_triangle_mesh("ash/rgbd_fusion_examples/mesh_filtered.ply", mesh_filtered.to_legacy())
