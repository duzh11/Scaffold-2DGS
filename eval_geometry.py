### Evalute mesh, normal and depth 

import os
import cv2
import json
import torch
import numpy as np
import open3d as o3d

from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser, Namespace


def write_image_info(image, image_info, interval_img=50, cat_mode= 'vertical'):
    H, W = image.shape[:2]
    if cat_mode == 'vertical':
        resize_image = np.concatenate([image, 255*np.ones((interval_img, W, 3))], axis=0)
        cv2.putText(resize_image, image_info,  (W//2-20, H+30), 
                    fontFace= cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, 
                    color = (0, 0, 255), 
                    thickness = 2)

    return resize_image

# concatenate images
def write_image_lis(path_img_cat, lis_imgs, use_cmap = False, interval_img = 20, cat_mode = 'horizontal', color_space = 'BGR', wirte_imgs=True):
    '''Concatenate an image list to a single image and save it to the target path
    Args:
        cat_mode: horizonal/vertical
    '''
    img_cat = []
    for i in range(len(lis_imgs)):
        img = lis_imgs[i].squeeze()
        H, W = img.shape[:2]
        
        if use_cmap:
            img = convert_gray_to_cmap(img) if img.ndim==2 else img
        else:
            img = np.stack([img]*3, axis=-1) if img.ndim==2 else img
        
        if img.max() <= 1.0:
            img *= 255

        img_cat.append(img)
        if cat_mode == 'horizontal':
            img_cat.append(255 * np.ones((H, interval_img, 3)).astype('uint8'))
        elif cat_mode == 'vertical':
            img_cat.append(255 * np.ones((interval_img, W, 3)).astype('uint8'))
    
    if cat_mode == 'horizontal':
        img_cat = np.concatenate(img_cat[:-1], axis=1)
    elif cat_mode == 'vertical':
        img_cat = np.concatenate(img_cat[:-1], axis=0)
    else:
        raise NotImplementedError
    if color_space == 'RGB':
        img_cat = cv2.cvtColor(img_cat.astype(np.uint8), cv2.COLOR_BGR2RGB)
    
    if wirte_imgs:
        cv2.imwrite(path_img_cat, img_cat)
    else: 
        return img_cat

def read_point_cloud(path_cloud):
    assert os.path.exists(path_cloud)
    cloud = o3d.io.read_point_cloud(path_cloud)
    return cloud

def nn_correspondance(verts1, verts2):
    """ for each vertex in verts2 find the nearest vertex in verts1

    Args:
        nx3 np.array's

    Returns:
        ([indices], [distances])

    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for vert in verts2:
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        indices.append(inds[0])
        distances.append(np.sqrt(dist[0]))

    return indices, distances

def visualize_error_map(pcd_pred, dist2, errormap_path, error_bound=0.10):
    from matplotlib import cm
    print('Visualize Error map of pcd...\n')

    dist_score = dist2.clip(0, error_bound) / error_bound
    color_map = cm.get_cmap('Reds')
    colors = color_map(dist_score)[:, :3]
    pcd_pred.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(errormap_path, pcd_pred)

def evaluate_mesh(model_paths, mesh_gt, pred_mesh_types=['tsdf'], 
                  threshold=.05, down_sample=.02, show_errormap=False):
    """ Borrowed from NeuralRecon and NeuRIS
    Compute Mesh metrics between prediction and target.

    Opens the Meshs and runs the metrics

    Args:
        model_paths: file path of GS model
        mesh_gt: file path of target
        pred_mesh_types: list of mesh types to evaluate
        threshold: distance threshold used to compute precision/recal
        down_sample: use voxel_downsample to uniformly sample mesh points

    Returns:
        Dict of mesh metrics
    """
    exp_paths = f'{model_paths}/train'
    mesh_pred_dict = {'tsdf': 'fuse_bounded_wmask_post.ply',
                      'poisson': 'fuse_poisson_post.ply'
                    }
    for pred_mesh_type in pred_mesh_types:
        full_dict = {}
        mesh_pred_file = mesh_pred_dict[pred_mesh_type]
        for method in os.listdir(exp_paths):
            full_dict[method] = {}
            mesh_pred = f'{exp_paths}/{method}/{mesh_pred_file}'

            pcd_pred = read_point_cloud(mesh_pred)
            pcd_trgt = read_point_cloud(mesh_gt)
            if down_sample:
                pcd_pred = pcd_pred.voxel_down_sample(down_sample)
                pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)
            verts_pred = np.asarray(pcd_pred.points)
            verts_trgt = np.asarray(pcd_trgt.points)

            ind1, dist1 = nn_correspondance(verts_pred, verts_trgt)  # para2->para1: dist1 is gt->pred
            ind2, dist2 = nn_correspondance(verts_trgt, verts_pred)
            dist1 = np.array(dist1)
            dist2 = np.array(dist2)

            precision = np.mean((dist2 < threshold).astype('float'))
            recall = np.mean((dist1 < threshold).astype('float'))
            fscore = 2 * precision * recall / (precision + recall)
            chamfer= np.mean(dist1**2)+np.mean(dist2**2)
            full_dict[method].update({'dist1': np.mean(dist2),  # pred->gt
                                'dist2': np.mean(dist1),  # gt -> pred
                                'precision': precision,
                                'recall': recall,
                                'fscore': fscore,
                                'chamfer': chamfer,
                                })

            if show_errormap:
                visualize_error_map(pcd_pred, dist2, mesh_pred.replace('.ply', '_error.ply'))
        
        with open(f'{model_paths}/result_mesh_{pred_mesh_type}.json', 'w') as f:
                json.dump(full_dict, f, indent=True)

def visualize_depth(depth, valid_mask=None, scale_factor=50):
    depth_vis = depth*scale_factor
    depth_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_vis), cv2.COLORMAP_JET)
    if valid_mask is not None:
        depth_vis[~valid_mask]=np.array([0, 0, 0])
    return depth_vis

def evaluate_depth(model_paths, gt_depth_files, min_depth=0.0, max_depth=20, train_test_flag='train', show_errormap=True):
    """ compute metrics of depth prediction
    Args:
        model_paths: file path of GS model
        gt_depth_files: files of depth_gt (.png)
        min_depth: minimum depth
        max_depth: maximum depth
        train_test_flag: 'train' or 'test'
        show_errormap: True or False
    Returns:
        depth metrics
        write to file
        visualize error map and and concat
    """
    exp_paths = f'{model_paths}/{train_test_flag}'
    full_dict = {}
    for method in os.listdir(exp_paths):
        pred_depth_files = glob(os.path.join(exp_paths, method, "renders_expected_depth", "*.npz"))
        pred_depth_files.sort()

        full_dict[method] = {}
        per_view_dict = {}
        abs_rel_lis, sq_rel_lis, rmse_lis, rmse_log_lis, a1_lis, a2_lis, a3_lis = [], [], [], [], [], [], []
        image_names = []
        for (idx, pred_depth_file) in tqdm(enumerate(pred_depth_files), desc='Evaluating depth'):
            image_names.append(pred_depth_file.split('/')[-1])
            pred_depth = np.load(pred_depth_file)['arr_0']
            pred_height, pred_width = pred_depth.shape[:2]

            gt_depth = cv2.imread(gt_depth_files[idx], cv2.IMREAD_UNCHANGED) / 1000.0
            mask = (gt_depth > 0) & (gt_depth < max_depth) & (gt_depth > min_depth)

            gt_depth = cv2.resize(gt_depth, (pred_width, pred_height))
            mask = cv2.resize(mask.astype(np.uint8), (pred_width, pred_height)).astype(bool)

            # Computation of error metrics between predicted and ground truth depths
            gt_depth_valid = gt_depth[mask]
            pred_depth_valid = pred_depth[mask]

            thresh = np.maximum((gt_depth_valid / pred_depth_valid), (pred_depth_valid / gt_depth_valid))
            a1 = (thresh < 1.25     ).mean()
            a2 = (thresh < 1.25 ** 2).mean()
            a3 = (thresh < 1.25 ** 3).mean()

            rmse = (gt_depth_valid - pred_depth_valid) ** 2
            rmse = np.sqrt(rmse.mean())

            rmse_log = (np.log(gt_depth_valid) - np.log(pred_depth_valid)) ** 2
            rmse_log = np.sqrt(rmse_log.mean())

            abs_rel = np.mean(np.abs(gt_depth_valid - pred_depth_valid) / gt_depth_valid)
            sq_rel = np.mean(((gt_depth_valid - pred_depth_valid) ** 2) / gt_depth_valid)

            # save into lists
            abs_rel_lis.append(abs_rel)
            sq_rel_lis.append(sq_rel)
            rmse_lis.append(rmse)
            rmse_log_lis.append(rmse_log)
            a1_lis.append(a1)
            a2_lis.append(a2)
            a3_lis.append(a3)

            # visualize error map
            if show_errormap:
                 errormap_path = os.path.join(exp_paths, method, "renders_expected_depth_error")
                 os.makedirs(errormap_path, exist_ok=True)

                 error_depth = np.sqrt((pred_depth - gt_depth) ** 2)
                 
                 # vislauze
                 gt_depth_vis = write_image_info(visualize_depth(gt_depth, valid_mask=mask), "gt_depth")
                 pred_depth_vis = write_image_info(visualize_depth(pred_depth), "pred_depth")
                 error_depth_vis = write_image_info(visualize_depth(error_depth, valid_mask=mask, scale_factor=500), "error_depth")
                 
                 # concating images
                 lis_imgs_cat = write_image_lis(f'{errormap_path}/{image_names[idx]}', 
                                                [gt_depth_vis, pred_depth_vis, error_depth_vis],
                                                wirte_imgs=False)
                 lis_imgs_metric = write_image_info(lis_imgs_cat,  f'rmse:  {rmse}', interval_img=50)
                 cv2.imwrite(f'{errormap_path}/{image_names[idx].replace(".npz", ".png")}', lis_imgs_metric)
        
        # save info tile
        full_dict[method].update({"abs_rel": np.mean(abs_rel_lis),
                                  "sq_rel": np.mean(sq_rel_lis),
                                  "rmse": np.mean(rmse_lis),
                                  "rmse_log": np.mean(rmse_log_lis),
                                  "a1": np.mean(a1_lis),
                                  "a2": np.mean(a2_lis),
                                  "a3": np.mean(a3_lis)})
        per_view_dict[method] = {"abs_rel": {name: result for result, name in zip(abs_rel_lis, image_names)},
                                 "sq_rel": {name: result for result, name in zip(sq_rel_lis, image_names)},
                                 "rmse": {name: result for result, name in zip(rmse_lis, image_names)},
                                 "rmse_log": {name: result for result, name in zip(rmse_log_lis, image_names)},
                                 "a1": {name: result for result, name in zip(a1_lis, image_names)},
                                 "a2": {name: result for result, name in zip(a2_lis, image_names)},
                                 "a3": {name: result for result, name in zip(a3_lis, image_names)}}

    with open(f"{model_paths}/results_{train_test_flag}_depth.json", 'w') as fp:
                json.dump(full_dict, fp, indent=True)
    with open(f"{model_paths}/per_view_{train_test_flag}_depth.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

def calculate_normal_error(pred_norm, gt_norm, valid_mask = None):
    if not torch.is_tensor(pred_norm):
        pred_norm = torch.from_numpy(pred_norm)
    if not torch.is_tensor(gt_norm):
        gt_norm = torch.from_numpy(gt_norm)
    prediction_error = torch.cosine_similarity(pred_norm, gt_norm, dim=-1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    E = torch.acos(prediction_error) * 180.0 / np.pi
    E = E.numpy()
    if valid_mask is not None:
        E[~valid_mask] = 0.0
    return E

def visualize_normal(normal, valid_mask = None):
    normal_vis = ( normal/np.linalg.norm(normal, axis=-1, keepdims=True) + 1)*0.5
    if valid_mask is not None:
        normal_vis[~valid_mask] = np.array([0, 0, 0])
    return normal_vis*255

def evaluate_normal(model_paths, gt_normal_files, train_test_flag='train', show_errormap=True):
    """ compute metrics of normal prediction
    Args:
        model_paths: file path of GS model
        gt_normal_files: files of normal_gt (.png)
        min_normal: minimum normal
        max_normal: maximum normal
        train_test_flag: 'train' or 'test'
        show_errormap: True or False
    Returns:
        normal metrics
        write to file
        visualize error map and and concat
    """
    exp_paths = f'{model_paths}/{train_test_flag}'
    full_dict = {}
    for method in os.listdir(exp_paths):
        pred_normal_files = glob(os.path.join(exp_paths, method, "renders_normal", "*.npz"))
        pred_normal_files.sort()

        full_dict[method] = {}
        per_view_dict = {}
        mean_error_lis, median_error_lis, rmse_lis, a1_lis, a2_lis, a3_lis, a4_lis, a5_lis = [], [], [], [], [], [], [], []
        image_names = []
        for (idx, pred_normal_file) in tqdm(enumerate(pred_normal_files), desc='Evaluating normal'):
            image_names.append(pred_normal_file.split('/')[-1])
            pred_normal = np.load(pred_normal_file)['arr_0']
            pred_height, pred_width = pred_normal.shape[:2]

            gt_normal = np.load(gt_normal_files[idx])['arr_0']
            mask = (np.sum(gt_normal, axis=-1)!=0)

            gt_normal = cv2.resize(gt_normal.astype(float), (pred_width, pred_height))
            mask = cv2.resize(mask.astype(np.uint8), (pred_width, pred_height)).astype(bool)

            # Computation of error metrics between predicted and ground truth normals
            normal_error = calculate_normal_error(pred_normal, gt_normal, valid_mask=mask)
            
            # save into lists
            normal_error_valid = normal_error[mask]
            mean_error_lis.append(np.mean(normal_error_valid))
            median_error_lis.append(np.median(normal_error_valid))
            rmse_lis.append(np.sqrt(np.sum(normal_error_valid * normal_error_valid) / normal_error_valid.shape[0]))
            a1_lis.append(100.0 * (np.sum(normal_error_valid < 5) / normal_error_valid.shape[0]))
            a2_lis.append(100.0 * (np.sum(normal_error_valid < 7.5) / normal_error_valid.shape[0]))
            a3_lis.append(100.0 * (np.sum(normal_error_valid < 11.25) / normal_error_valid.shape[0]))
            a4_lis.append(100.0 * (np.sum(normal_error_valid < 22.5) / normal_error_valid.shape[0]))
            a5_lis.append(100.0 * (np.sum(normal_error_valid < 30) / normal_error_valid.shape[0]))

            # visualize error map
            if show_errormap:
                 errormap_path = os.path.join(exp_paths, method, "renders_normal_error")
                 os.makedirs(errormap_path, exist_ok=True)
 
                 # vislauze
                 gt_normal_vis = write_image_info(visualize_normal(gt_normal, valid_mask=mask)[..., ::-1], "gt_normal")
                 pred_normal_vis = write_image_info(visualize_normal(pred_normal)[..., ::-1], "pred_normal")
                 error_normal_vis = write_image_info(visualize_depth(normal_error, valid_mask=mask, scale_factor=10), "error_normal")
                 
                 # concating images
                 lis_imgs_cat = write_image_lis(f'{errormap_path}/{image_names[idx]}', 
                                                [gt_normal_vis, pred_normal_vis, error_normal_vis],
                                                wirte_imgs=False)
                 lis_imgs_metric = write_image_info(lis_imgs_cat,  f'rmse:  {rmse_lis[idx]}', interval_img=50)
                 cv2.imwrite(f'{errormap_path}/{image_names[idx].replace(".npz", ".png")}', lis_imgs_metric)
        
        # save info tile
        full_dict[method].update({"mean": np.mean(mean_error_lis),
                                  "median": np.mean(median_error_lis),
                                  "rmse": np.mean(rmse_lis),
                                  "a1": np.mean(a1_lis),
                                  "a2": np.mean(a2_lis),
                                  "a3": np.mean(a3_lis),
                                  "a4": np.mean(a4_lis),
                                  "a5": np.mean(a5_lis)})
        per_view_dict[method] = {"mean": {name: result for result, name in zip(mean_error_lis, image_names)},
                                 "median": {name: result for result, name in zip(median_error_lis, image_names)},
                                 "rmse": {name: result for result, name in zip(rmse_lis, image_names)},
                                 "a1": {name: result for result, name in zip(a1_lis, image_names)},
                                 "a2": {name: result for result, name in zip(a2_lis, image_names)},
                                 "a3": {name: result for result, name in zip(a3_lis, image_names)},
                                 "a4": {name: result for result, name in zip(a4_lis, image_names)},
                                 "a5": {name: result for result, name in zip(a5_lis, image_names)}}

    with open(f"{model_paths}/results_{train_test_flag}_normal.json", 'w') as fp:
                json.dump(full_dict, fp, indent=True)
    with open(f"{model_paths}/per_view_{train_test_flag}_normal.json", 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)

if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--source_path', '-s', type=str, default='../../Data/ScanNetpp/8b5caf3398')
    parser.add_argument('--model_paths', '-m', type=str, default='../exps/full/scannetpp/8b5caf3398')
    parser.add_argument('--train_test_flag', '-f', nargs="+", type=str, default=['train'])
    parser.add_argument("--pred_mesh_types", '-p', nargs="+", type=str, default=['tsdf'])
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--skip_depth", action="store_true")
    parser.add_argument("--skip_normal", action="store_true")
    args = parser.parse_args()

    # evaluate mesh
    if not args.skip_mesh:
        mesh_gt = f'{args.source_path}/mesh.ply'
        evaluate_mesh(args.model_paths, mesh_gt, pred_mesh_types = args.pred_mesh_types, show_errormap=True)

    # evaluate depth & normal
    train_test_split = json.load(open(f'{args.source_path}/train_test_lists.json'))
    for train_test_flag in args.train_test_flag:
        # evaluate depth 
        if not args.skip_depth:
            gt_depth_files = [f'{args.source_path}/depths/{file_name.replace(".JPG", ".png")}' for file_name in train_test_split[train_test_flag]]
            gt_depth_files.sort()
            evaluate_depth(args.model_paths, gt_depth_files, train_test_flag=train_test_flag)

        # evaluate normal
        if not args.skip_normal:
            gt_normal_files = [f'{args.source_path}/normals/{file_name.replace(".JPG", ".npz")}' for file_name in train_test_split[train_test_flag]]
            gt_normal_files.sort()
            evaluate_normal(args.model_paths, gt_normal_files, train_test_flag=train_test_flag)

        
    