from typing import Literal
import numpy as np

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import open3d as o3d
import open3d.core as o3c

import cv2

from pathlib import Path
from tqdm import tqdm


def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def get_image_files(
    path, folders=["image", "color"], exts=["jpg", "png", "pgm", "npy"]
):
    for folder in folders:
        for ext in exts:
            image_fnames = sorted((path / folder).glob(f"*.{ext}"))
            if len(image_fnames) > 0:
                return image_fnames
    raise ValueError(f"no images found in {path}")


def load_image(fname, im_type="image"):
    """
    Load image from file.
    """
    if fname.suffix == ".npy":
        # Versatile, could work for any image type
        data = np.load(fname).astype(np.float32)
        if data.shape[0] in [1, 3]:  # normal or depth transposed
            data = data.transpose((1, 2, 0))
        if im_type == "depth":
            data = data.squeeze()
        return data
    elif fname.suffix in [".jpg", ".jpeg", ".png", ".pgm"]:
        if im_type == "image":
            # Always normalize RGB to [0, 1]
            img = cv2.imread(str(fname), cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
            img = cv2.resize(img, (640, 480), cv2.INTER_AREA)
            return img/255.0
        elif im_type == "depth":
            # Keep depth as they are as unit is usually tricky
            return cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"unknown image type {im_type} with {fname.suffix}")
    else:
        raise ValueError(f"unknown image type {fname.suffix}")


class ImageDataset(torch.utils.data.Dataset):
    """Minimal RGBD dataset with known poses
    As a raw data container, the images can be accessed at [i] by image-wise
    - depth: (N, H, W) float32
    - color: (N, H, W, 3) float32
    (optional)
    - normal: (N, H, W) bool
    - semantic: (N, H, W, D) float32
    All in the camera coordinate systems.

    As a dataloader for ray-based rendering, the pixels can be loaded by dataloaders at [i] through pixel-wise
    - ray_o: (N * H * W, 3)
    - ray_d: (N * H * W, 3)
    - depth: (N * H * W, 1)
    - color: (N * H * W, 3)
    (optional)
    - normal: (N * H * W, 3)
    - semantic: (N * H * W, D)
    All transformed into the global coordinate systems since shuffle may result in further complexity.

    All the data are loaded to numpy in CPU memory.
    TODO: follow nerfstudio and add more flexibility with cached dataloader
    """

    # pixel value * depth_scales = depth in meters
    def __init__(
        self,
        path,
        depth_max: float = 4.0,
        normalize_scene: bool = True,
        generate_rays: bool = False,
    ):
        self.path = Path(path)
        self.depth_max = depth_max
        self.generate_rays = generate_rays

        # Load fnames
        self.image_fnames = np.stack(get_image_files(
            self.path, folders=["image", "color"], exts=["png", "jpg"]
        ))

        self.depth_fnames = np.stack(get_image_files(
            self.path, folders=["depth"], exts=["png", "pgm"]
        ))
        self.depth_scales = np.ones(len(self.depth_fnames)) * 1e-3        
        
        # Load intrinsics and poses
        self.intrinsic = np.loadtxt(self.path / "intrinsic/intrinsic_depth.txt")[:3, :3].reshape((3, 3))
        
        self.pose_frames = get_image_files(
            self.path, folders=["pose"], exts=["txt"]
        )
        self.poses_unnormalized = np.stack([np.loadtxt(i) for i in self.pose_frames])

        # filter out invalid poses
        valid_mask = ~np.isinf(self.poses_unnormalized).any(axis=(1, 2))
        invalid_indices = np.where(~valid_mask)[0]
        if valid_mask.sum()>0:
            print(f"Found {valid_mask.sum()} valid poses: {invalid_indices}")
            self.image_fnames = self.image_fnames[valid_mask]
            self.depth_fnames = self.depth_fnames[valid_mask]
            self.depth_scales = self.depth_scales[valid_mask]
            self.poses_unnormalized = self.poses_unnormalized[valid_mask]
        
        assert len(self.image_fnames) == len(
            self.depth_fnames
        ), f"{len(self.image_fnames)} != {len(self.depth_fnames)}"
        assert len(self.image_fnames) == len(
            self.poses_unnormalized
        ), f"{len(self.image_fnames)} != {len(self.poses_unnormalized)}"

        # Load images and shapes
        depth_ims = []
        rgb_ims = []
        pbar = tqdm(range(len(self.image_fnames)))
        for i in pbar:
            pbar.set_description(f"Loading {self.image_fnames[i].name}")
            depth_ims.append(load_image(self.depth_fnames[i], "depth"))
            pbar.set_description(f"Loading {self.depth_fnames[i].name}")
            rgb_ims.append(load_image(self.image_fnames[i], "image"))
            pbar.update()
        self.depth_ims = np.stack(depth_ims)
        self.rgb_ims = np.stack(rgb_ims)

        self.num_images, self.H, self.W = self.depth_ims.shape[:3]
        assert self.rgb_ims.shape == (self.num_images, self.H, self.W, 3)

        # Normalize the scene if necessary
        self.bbox_T_world = np.eye(4)
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0

        if normalize_scene:
            min_vertices = self.poses_unnormalized[:, :3, 3].min(axis=0)
            max_vertices = self.poses_unnormalized[:, :3, 3].max(axis=0)
            self.center = (min_vertices + max_vertices) / 2.0

            self.scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
            self.depth_scales *= self.scale
            self.depth_max *= self.scale
            print(f"Normalize scene with scale {self.scale} and center {self.center}m")

        poses = []
        extrinsics = []
        for pose in self.poses_unnormalized:
            pose[:3, 3] = (pose[:3, 3] - self.center) * self.scale
            poses.append(pose)
            extrinsics.append(np.linalg.inv(pose))
        self.poses = np.stack(poses)
        self.extrinsics = np.stack(extrinsics)

        # Generate rays
        if not self.generate_rays:
            return

        yy, xx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        yy = yy.flatten().astype(np.float32)
        xx = xx.flatten().astype(np.float32)

        # (H*W, 3)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)

        self.rays_d = []
        self.rays_d_norm = []
        self.rays_o = []
        pbar = tqdm(range(len(self.image_fnames)))
        for i in pbar:
            pbar.set_description(f"Generating rays for image {i}")
            P = self.intrinsic @ self.extrinsics[i, :3, :3]
            inv_P = np.linalg.inv(P)

            # Note: this rays_d is un-normalized
            rays_d = np.matmul(rays, inv_P.T)
            rays_d_norm = np.linalg.norm(rays_d, axis=-1, keepdims=True)
            rays_o = np.tile(self.poses[i, :3, 3], (self.H * self.W, 1))

            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d / rays_d_norm)
            self.rays_d_norm.append(rays_d_norm)

        self.rays_d = (
            np.concatenate(self.rays_d, axis=0).reshape(-1, 3).astype(np.float32)
        )
        self.rays_d_norm = (
            np.concatenate(self.rays_d_norm, axis=0).reshape(-1, 1).astype(np.float32)
        )
        self.rays_o = (
            np.concatenate(self.rays_o, axis=0).reshape(-1, 3).astype(np.float32)
        )

        self.depths = self.depth_ims.reshape(-1, 1).astype(np.float32)
        self.rgbs = self.rgb_ims.reshape(-1, 3).astype(np.float32)
        self.depth_scales = self.depth_scales.reshape(-1, 1).astype(np.float32)

    def get_image(self, idx):
        return {
            "rgb": self.rgb_ims[idx],
            "depth": self.depth_ims[idx],
            "depth_scale": self.depth_scales[idx],
            "depth_max": self.depth_max,
            "pose": self.poses[idx],
            "extrinsic": self.extrinsics[idx],
            "intrinsic": self.intrinsic,
        }

    def num_images(self):
        return len(self.rgb_ims)

    def __len__(self):
        if not self.generate_rays:
            raise RuntimeError(
                f"Cannot get number of rays from image-only dataset. Please use num_images() instead."
                f"If you want to use rays, please set generate_rays=True when creating the dataset."
            )

        return len(self.rays_d)

    def __getitem__(self, idx):
        if not self.generate_rays:
            raise RuntimeError(
                f"Cannot get rays {idx} from image-only dataset. Please use get_image({idx}) for images."
                f"If you want to use rays, please set generate_rays=True when creating the dataset."
            )

        view_idx = idx // (self.H * self.W)
        return {
            "view_idx": view_idx,
            "rays_o": self.rays_o[idx],  # (N, 3)
            "rays_d": self.rays_d[idx],  # (N, 3)
            "rays_d_norm": self.rays_d_norm[idx],  # (N, 1)
            "depth": self.depths[idx],  # (N, 1)
            "depth_scale": self.depth_scales[view_idx],  # (N, 1)
            "rgb": self.rgbs[idx],  # (N, 3)
        }

class Dataloader:
    def __init__(self, dataset, batch_size, shuffle, device=torch.device("cpu")):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self._generate_indices()

    def _generate_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        self.num_batches = (
            len(self.dataset) + self.batch_size - 1
        ) // self.batch_size  # Round up to the nearest integer

        self.batch_indices = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            self.batch_indices.append(indices[start:end])

        self.batch_idx = 0

    def __getitem__(self, indices):
        batch = self.dataset.__getitem__(indices)
        for k, v in batch.items():
            # TODO: use key check instead
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(self.device)
            elif isinstance(v, float):  # depth_scale
                batch[k] = torch.tensor(v).unsqueeze(-1).float().to(self.device)
            elif isinstance(v, int):  # view_idx
                batch[k] = torch.tensor(v).to(self.device)
        return batch

    def __iter__(self):
        indices = self.batch_indices[self.batch_idx]

        self.batch_idx += 1
        if self.batch_idx == self.num_batches:
            self.batch_idx = 0
            if self.shuffle:
                self._generate_indices()

        yield self.__getitem__(indices)