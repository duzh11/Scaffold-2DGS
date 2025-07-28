import os
import cv2
import shutil

from tqdm import tqdm

# cfg for ScanNet++
scannetpp_root = '/home/du/Proj/Dataset/ScanNetpp/scans/data'
scene_ids = ['8b5caf3398', '116456116b', '13c3e046d7', '0a184cf634', '578511c8a9', '21d970d8de']
camera_type = 'dslr'
camera_model = 'PINHOLE'

target_root = '/home/du/Proj/3Dv_Reconstruction/GS-Reconstruction/Data/ScanNetpp'

# file_type: [source_dir, target_dir]
copyfile_dict = {
    'colmap': ['colmap_undistorted', 'sparse/0'],
    'image': ['undistorted_images', 'images'], 
    'mask': ['undistorted_anon_masks', 'mask'], 
    'depth': ['render_depth_PINHOLE', 'depths'], 
    'depth_vis': ['render_depth_PINHOLE_vis', 'depths_vis'], 
    'normal': ['render_normal_PINHOLE', 'normals'], 
    'normal_vis': ['render_normal_PINHOLE_vis', 'normals_vis']
}

for scene_id in tqdm(scene_ids, desc='Processing ScanNet++'):
    scene_source_dir = os.path.join(scannetpp_root, scene_id, camera_type)
    scene_target_dir = os.path.join(target_root, scene_id)
    os.makedirs(scene_target_dir, exist_ok=True)

    # mesh, train_test_lists.json, colmap_file
    shutil.copyfile(os.path.join(scannetpp_root, scene_id, 'scans/mesh_aligned_0.05.ply'), scene_target_dir+'/mesh.ply')
    shutil.copyfile(scene_source_dir+'/train_test_lists.json', scene_target_dir+'/train_test_lists.json')
    shutil.copytree(scene_source_dir+'/'+copyfile_dict['colmap'][0], scene_target_dir+'/'+copyfile_dict['colmap'][1], dirs_exist_ok = True)

    # image+mask
    shutil.copytree(scene_source_dir+'/'+copyfile_dict['image'][0], scene_target_dir+'/'+copyfile_dict["image"][1], dirs_exist_ok = True)
    shutil.copytree(scene_source_dir+'/'+copyfile_dict["mask"][0], scene_target_dir+'/'+copyfile_dict["mask"][1], dirs_exist_ok = True)
    
    # fuse image and mask
    image_dir, mask_dir = scene_target_dir+'/'+copyfile_dict["image"][1], scene_target_dir+'/'+copyfile_dict["mask"][1]
    image_lis = os.listdir(image_dir)
    for image_name in image_lis:
        image = cv2.imread(os.path.join(image_dir, image_name))
        mask = cv2.imread(os.path.join(mask_dir, image_name.replace('.JPG', '.png')), cv2.IMREAD_UNCHANGED)

        # merge image and mask
        b, g, r = cv2.split(image)
        alpha = mask
        image_new = cv2.merge((b, g, r, alpha))
        cv2.imwrite(os.path.join(image_dir, image_name), image_new)

    # depth, depth_vis, normal, normal_vis
    shutil.copytree(scene_source_dir+'/'+copyfile_dict["depth"][0], scene_target_dir+'/'+copyfile_dict["depth"][1], dirs_exist_ok = True)
    shutil.copytree(scene_source_dir+'/'+copyfile_dict["depth_vis"][0], scene_target_dir+'/'+copyfile_dict["depth_vis"][1], dirs_exist_ok = True)
    shutil.copytree(scene_source_dir+'/'+copyfile_dict["normal"][0], scene_target_dir+'/'+copyfile_dict["normal"][1], dirs_exist_ok = True)
    shutil.copytree(scene_source_dir+'/'+copyfile_dict["normal_vis"][0], scene_target_dir+'/'+copyfile_dict["normal_vis"][1], dirs_exist_ok = True)