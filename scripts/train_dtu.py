import os
from datetime import datetime
import random
os.system("ulimit -n 4096")
current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
port = random.randint(10000, 30000)

data_root = '../Data/DTU'
exp_name = f'./exps/experiments_v0/DTU-{current_time}'
DTU_Official = '../Data/Offical_DTU_Dataset'

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
gpu = -1

cmd_lis = []
script_dir = os.path.dirname(os.path.abspath(__file__))
for scene in dtu_scenes:
    
    source_args = " -s " + data_root + "/" + scene
    exp_args = " -m " + exp_name+"/"+scene
    
    # training
    train_args = source_args + exp_args + f" --depth_ratio 1.0 -r 2 --test_iterations -1 --use_wandb --lod 0 --gpu {gpu} --port {port} --voxel_size 0.001 --update_init_factor 4 --appearance_dim 0 --ratio 1 --lambda_dist 1000"
    train_args += " --near_plane 1.5 --far_plane 5.0"
    cmd_lis.append("python train.py" + train_args)

    # # # rendering mesh
    render_args = source_args + exp_args + " --depth_ratio 1.0 -r 2 --num_cluster 1 --tsdf_voxel 0.002 --sdf_trunc 0.016 --depth_trunc 3.0 --skip_train --skip_test --usingmask"
    cmd_lis.append(f"python render.py" + render_args)

    # # NVS metricd and visualization
    # cmd_lis.append(f"python metrics.py" + exp_args + ' -f train')
    cmd_lis.append(f"python vis_outputs.py" + exp_args + ' -f train')

    # evaluate dtu-mesh
    scan_id = scene[4:]
    ply_file = f"{exp_name}/{scene}/train/ours_30000/"
    iteration = 30000
    string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
        f"--input_mesh {exp_name}/{scene}/train/ours_30000/fuse_bounded_wmask_post.ply " + \
        f"--scan_id {scan_id} --output_dir {exp_name}/{scene}/vis " + \
        f"--mask_dir {data_root} " + \
        f"--DTU {DTU_Official}"
    cmd_lis.append(string)
    

cmd_lis.append(f'python summary_dtu.py --use_wandb -m ' + exp_name)

# run cmd
for cmd in cmd_lis:
    print(cmd)
    os.system(cmd)