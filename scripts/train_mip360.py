import os
from datetime import datetime
import random
os.system("ulimit -n 4096")
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '/home/zhenhua2023/Proj/3Dv_NVS/GS-NVS/Data/Mip-NeRF360'
exp_name = f'./exps/experiments_v0/Mip-NeRF360-{current_time}'
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
gpu = -1

cmd_lis = []
for scene in (mipnerf360_outdoor_scenes+mipnerf360_indoor_scenes):
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" --eval --test_iterations -1 --use_wandb --lod 0 --gpu {gpu} --port {port} --voxel_size 0.001 --update_init_factor 16 --appearance_dim 0 --ratio 1 --far_plane 20.0"
    if scene in mipnerf360_outdoor_scenes:
        train_args += " -i images_4"
    elif scene in mipnerf360_indoor_scenes:
        train_args += " -i images_2"
    cmd_lis.append("python train.py" + train_args)

    # rendering images, and mesh
    cmd_lis.append(f"python render.py" + source_args + exp_args + " --skip_train --skip_test")

    # # NVS metricd and visualization
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f train')
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f test')
    cmd_lis.append(f"python vis_outputs.py" + exp_args + ' -f train test')

cmd_lis.append(f'python summary.py --use_wandb -m ' + exp_name)

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)