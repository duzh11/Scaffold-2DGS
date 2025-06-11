import numpy as np
import torch
import torch.nn as nn

rays_dir = None

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class OptimizablePose(nn.Module):
    def __init__(self, init_pose):
        super().__init__()
        assert (isinstance(init_pose, torch.FloatTensor))
        self.register_parameter('data', nn.Parameter(init_pose))
        self.data.required_grad_ = True

    def copy_from(self, pose):
        self.data = deepcopy(pose.data)

    def matrix(self):
        Rt = torch.eye(4)
        Rt[:3, :3] = self.rotation()
        Rt[:3, 3] = self.translation()
        return Rt

    def rotation(self):
        w = self.data[3:]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I + A * wx + B * wx @ wx
        return R

    def translation(self, ):
        return self.data[:3]

    @classmethod
    def log(cls, R, eps=1e-7):  # [...,3,3]
        """
        compute so(3) from the SO(3)
        Args:
            R (tensor): SO(3),rotation matrix.
        Returns:
            w (tensor,R^{3}): so(3).
        """
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        # ln(R) will explode if theta==pi
        theta = ((trace - 1) / 2).clamp(-1 + eps, 1 -
                                        eps).acos_()[..., None, None] % np.pi
        lnR = 1 / (2 * cls.taylor_A(theta) + 1e-8) * \
              (R - R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @classmethod
    def from_matrix(cls, Rt, eps=1e-8):  # [...,3,4]
        """
        Conversion of Lie Groups to Lie Algebras(SE(3)->se(3)), and add grad using torch.nn.Module.
        Args:
            cls (class): self.class
            Rt (tensor,4*4): SE(3)T, transformation matrix
        Returns:
            OptimizablePose (class): add 'data'(se(3),R^{6}) parameters incluing grad
        """
        R, u = Rt[:3, :3], Rt[:3, 3]
        w = cls.log(R)
        wx = cls.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = cls.taylor_A(theta)
        B = cls.taylor_B(theta)
        # invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
        return OptimizablePose(torch.cat([u, w], dim=-1))

    @classmethod
    def skew_symmetric(cls, w):
        """
        get symmetric matrix according vector.
        """
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    @classmethod
    def taylor_A(cls, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            if i > 0:
                denom *= (2 * i) * (2 * i + 1)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_B(cls, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 1) * (2 * i + 2)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

    @classmethod
    def taylor_C(cls, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth + 1):
            denom *= (2 * i + 2) * (2 * i + 3)
            ans = ans + (-1) ** i * x ** (2 * i) / denom
        return ans

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
            self.d_pose = OptimizablePose.from_matrix(torch.eye(4, requires_grad=False, dtype=torch.float32))
        else:
            self.ref_pose = None
        self.precompute()

    def get_d_pose(self):
        return self.d_pose.matrix()

    def get_d_translation(self):
        return self.d_pose.translation()

    def get_d_rotation(self):
        return self.d_pose.rotation()

    def get_d_pose_param(self):
        return self.d_pose.parameters()

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
    def __init__(self, args, voxel_size=0.01):
        torch.classes.load_library(
            "submodules/sparse_octree/build/lib.linux-x86_64-cpython-39/svo.cpython-39-x86_64-linux-gnu.so")

        # initialize svo
        num_vertexes = 200000
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
