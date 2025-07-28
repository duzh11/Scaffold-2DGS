import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('./')
from eval_geometry import write_image_info, visualize_depth

data_dir = '/home/zhenhua2023/Proj/3Dv_Reconstruction/GS-Reconstruction/Data'
data_name = 'ScanNetpp'
scene_lis = ['8b5caf3398']
monodepth_name = 'depths_any2'

def vis_monodepth(scene_dir, monodepth_name):
    monodepth_dir = os.path.join(scene_dir, monodepth_name)
    monodepth_vis_dir = os.path.join(scene_dir, monodepth_name+'_vis')
    os.makedirs(monodepth_vis_dir, exist_ok=True)

    # scale and shift params
    monodepths_params_path = os.path.join(scene_dir, 'sparse/0/depth_params.json')
    with open(monodepths_params_path, 'r') as f:
        monodepths_params = json.load(f)
    
    monodepth_lis = os.listdir(monodepth_dir)
    for monodepth_idx in tqdm(monodepth_lis, desc='vis monodepth'):   
        camera_id = monodepth_idx.split('.')[0]
        depth_params = monodepths_params[camera_id]

        monodepth_path = os.path.join(monodepth_dir, monodepth_idx)
        invdepthmap = cv2.imread(monodepth_path, -1).astype(np.float32) / float(2**16)

        monoinvdepth = invdepthmap * depth_params["scale"] + depth_params["offset"]
        monodepth = 1 / monoinvdepth

        monodepth_vis = visualize_depth(monodepth)
        cv2.imwrite(os.path.join(monodepth_vis_dir, monodepth_idx), monodepth_vis)

for scene in scene_lis:
    img_dir = os.path.join(data_dir, data_name, scene, 'images')
    out_dir = os.path.join(data_dir, data_name, scene, monodepth_name)

    # print(f"Generating depth for {scene}...")
    # os.system(f"python submodules/Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path {img_dir} --outdir {out_dir}")

    # print(f"Making depth scale for {scene}...")
    # os.system(f"python utils/make_depth_scale.py --base_dir {data_dir}/{data_name}/{scene} --depths_dir {out_dir}")

    vis_monodepth(os.path.join(data_dir, data_name, scene), monodepth_name)