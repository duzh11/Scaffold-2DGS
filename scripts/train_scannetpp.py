import os
from datetime import datetime
import random
os.system("ulimit -n 4096")
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '../../Data/ScanNetpp'
exp_name = f'../exps/experiments_v0/scannetpp-{current_time}'
scannetpp_scenes = ['8b5caf3398', '116456116b', '13c3e046d7', '0a184cf634', '578511c8a9', '21d970d8de']
gpu = -1

cmd_lis = []
for scene in scannetpp_scenes:
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" --eval --use_wandb --lod 0 --gpu {gpu} --port {port} --voxel_size 0.01 --update_init_factor 16 --appearance_dim 0 --ratio 1 -r 2 --depth_ratio 1.0 --lambda_dist 10"
    cmd_lis.append("python train.py" + train_args)

    # # rendering images
    render_args = source_args + exp_args + " --skip_train --skip_test --depth_ratio 1.0 -r 2 --eval"
    if scene in ["8b5caf3398"]:
        render_args += " --depth_trunc 5.0 --tsdf_voxel 0.01 --num_cluster 5"
    elif scene in ["21d970d8de"]:
        render_args += " --depth_trunc 5.0 --tsdf_voxel 0.01 --num_cluster 20"
    elif scene in ['578511c8a9']:
        render_args += " --depth_trunc 8.0 --tsdf_voxel 0.02 --num_cluster 1 --scale_factor 1.5"
    else:    
        render_args += " --depth_trunc 5.0 --tsdf_voxel 0.01 --num_cluster 1"
    cmd_lis.append(f"python render.py" + render_args)
    cmd_lis.append(f"python render.py" + render_args + " --usingmask")

    # # NVS metricd and visualization
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f train')
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f test')
    cmd_lis.append(f"python vis_outputs.py" + exp_args + ' -f train test')

    # evaluate mesh, depth & normal
    eval_args = source_args + exp_args + " -f train test -p tsdf"
    cmd_lis.append(f"python eval_geometry.py" + eval_args)

cmd_lis.append(f'python summary.py -m ' + exp_name + ' -s scannetpp --use_wandb')

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)