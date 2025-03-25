import os
from datetime import datetime
import random
os.system("ulimit -n 4096")
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '../../Data/TNT'
exp_name = f'../exps/experiments_v0/TNT-{current_time}'
TNT_GT = '../../Data/Official_TNT_dataset'

tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
tnt_large_scenes = ['Meetingroom', 'Courthouse']
tnt_scenes = tnt_360_scenes + tnt_large_scenes
gpu = -1

cmd_lis = []
script_dir = os.path.dirname(os.path.abspath(__file__))
for scene in tnt_scenes:
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" --depth_ratio 1.0 -r 2 --eval --test_iterations -1 --use_wandb --lod 0 --gpu {gpu} --port {port} --voxel_size 0.01 --update_init_factor 16 --appearance_dim 0 --ratio 1"
    train_args += " --far_plane 10.0"
    if scene in tnt_360_scenes:
        train_args += " --lambda_dist 100"
    elif scene in tnt_large_scenes:
        train_args += " --lambda_dist 10"
    # cmd_lis.append("python train.py" + train_args)

    # # rendering mesh
    render_args = source_args + exp_args + " --depth_ratio 1.0 -r 2 --eval --skip_train --skip_test"
    if scene in tnt_360_scenes:
        render_args += " --num_cluster 1 --tsdf_voxel 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    elif scene in tnt_large_scenes:
        render_args += " --num_cluster 1 --tsdf_voxel 0.006 --sdf_trunc 0.024 --depth_trunc 4.5"

    # cmd_lis.append(f"python render.py" + render_args)

    # # NVS metricd and visualization
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f train')
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f test')
    # cmd_lis.append(f"python vis_outputs.py" + exp_args + ' -f train test')

    # evaluate mesh, depth & normal
    # require open3d 0.9
    ply_file = f"{exp_name}/{scene}/train/ours_30000/fuse_bounded_post.ply"
    string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
        f"--dataset-dir {TNT_GT}/{scene} " + \
        f"--traj-path {data_root}/{scene}/{scene}_COLMAP_SfM.log " + \
        f"--ply-path {ply_file}"
    cmd_lis.append(string)
    
# cmd_lis.append(f'python summary.py -s tnt -m ' + exp_name)

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)