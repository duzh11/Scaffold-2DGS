import os
import numpy as np
import json
from glob import glob
import shutil

scene_Lis = ['116456116b', '13c3e046d7', '21d970d8de']
for scene in scene_Lis:
    # exporting caminfos
    exp_dir = f"./exps/experiments_v0/scannetpp-2025-02-14_01-00-30/{scene}"
    os.system(f"python render.py --skip_train --skip_test --skip_mesh --export_caminfos -m {exp_dir}")

    target_dir = f"./torch-ash/Data/{scene}/samples"
    target_image_dir = f"{target_dir}/image"
    target_depth_dir = f"{target_dir}/depth"
    target_normal_dir = f"{target_dir}/normal"
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_depth_dir, exist_ok=True)
    os.makedirs(target_normal_dir, exist_ok=True)

    # Copy depth npz files
    exp_train_dir= f'{exp_dir}/train'

    print("copying img, depth and normal files...")
    for img_file in glob(os.path.join(f"{exp_train_dir}/ours_30000/gt", "*.png")):
        shutil.copy(img_file, target_image_dir)

    for depth_file in glob(os.path.join(f"{exp_train_dir}/ours_30000/renders_expected_depth", "*.npz")):
        shutil.copy(depth_file, target_depth_dir)

    # Copy normal npz files
    for normal_file in glob(os.path.join(f"{exp_train_dir}/ours_30000/renders_normal", "*.npz")):
        shutil.copy(normal_file, target_normal_dir)

    GT_dataset = f"../Data/ScanNetpp/{scene}"
    GT_mesh = f"{GT_dataset}/mesh.ply"
    shutil.copy(GT_mesh, os.path.join(target_dir, "../gt.ply"))
    print("copying gt mesh...")

    # Load intrinsics and extrinsics data
    intrinsics_json, extrinsics_json = f"{exp_train_dir}/intrinsics.json", f"{exp_train_dir}/extrinsics.json"

    with open(intrinsics_json, 'r') as f:
        intrinsics_data = json.load(f)

    with open(extrinsics_json, 'r') as f:
        extrinsics_data = json.load(f)

    # Create poses and intrinsics files
    poses_file = f"{target_dir}/poses.txt"
    intrinsics_file = f"{target_dir}/intrinsic.txt"

    # Get sorted camera indices (按照字典索引顺序)
    camera_indices = sorted([k for k in extrinsics_data.keys() if extrinsics_data[k]])

    # Generate poses.txt (取extrinsic的逆得到pose)
    with open(poses_file, 'w') as f:
        for cam_id in camera_indices:
            if extrinsics_data[cam_id]:  # 检查是否有数据
                # 获取4x4外参矩阵
                extrinsic_matrix = np.array(extrinsics_data[cam_id])
                
                # 计算逆矩阵得到pose
                pose_matrix = np.linalg.inv(extrinsic_matrix)
                
                # 按照poses.txt格式写入，每行写4个值
                for i in range(4):
                    for j in range(4):
                        f.write(f"{pose_matrix[i,j]}")
                        if j < 3:
                            f.write(" ")
                    f.write("\n")

    # Generate intrinsic.txt (取intrinsics.json中第一个有效值)
    first_cam_id = camera_indices[0]
    first_intrinsics = intrinsics_data[first_cam_id]

    # intrinsics_data中的格式是3x3矩阵: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    intrinsic_matrix = np.array(first_intrinsics)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1] 
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    # 按照intrinsic.txt的3x3矩阵格式写入
    with open(intrinsics_file, 'w') as f:
        f.write(f"{fx} 0.000000000000000000e+00 {cx}\n")
        f.write(f"0.000000000000000000e+00 {fy} {cy}\n") 
        f.write(f"0.000000000000000000e+00 0.000000000000000000e+00 1.000000000000000000e+00\n")

    print(f"Generated {len(camera_indices)} poses in {poses_file}")
    print(f"Generated intrinsics file: {intrinsics_file}")
    print(f"Using intrinsics from camera {first_cam_id}: fx={fx}, fy={fy}, cx={cx}, cy={cy}")





