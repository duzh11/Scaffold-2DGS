import os

# scannetpp_scenes = ['8b5caf3398', '116456116b', '13c3e046d7', '21d970d8de']
scannetpp_scenes = ['116456116b']
for scene in scannetpp_scenes:
    os.system(f"python render.py --skip_mesh -m ./exps/experiments_v0/scannetpp-2025-02-14_01-00-30/{scene} --depth_ratio 1.0")