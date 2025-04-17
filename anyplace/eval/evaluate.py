import os, os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import threading
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation
import open3d as o3d
import copy
from pathlib import Path
import meshcat

from airobot import set_log_level
from airobot.utils.pb_util import create_pybullet_client
from anyplace.utils import util, config_util
from anyplace.utils import path_util
from anyplace.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat
from anyplace.utils.mesh_util import three_util
from anyplace.utils.anyplace.multistep_pose_regression_anyplace import policy_inference_methods_dict
from anyplace.utils.relational_policy.multistep_pose_regression_nsm import nsm_policy_inference_methods_dict
from anyplace.model.transformer.policy import (
    NSMTransformerSingleTransformationRegression, 
    NSMTransformerImplicit
)

def get_transform( t_7d ):
    t_7d = t_7d.reshape(-1, 7)
    # print("t_7d: ", t_7d)
    t = np.eye(4)
    trans = t_7d[:,0:3]
    quat = t_7d[:,3:7]
    # print("quat: ", quat)
    t[:3, :3] = Rotation.from_quat( quat ).as_matrix()
    t[:3, 3] = trans
    # print(t)
    return t

def get_7D_transform(transf):
    trans = transf[0:3,3]
    trans = trans.reshape(3)
    quat = Rotation.from_matrix( transf[0:3,0:3] ).as_quat()
    quat = quat.reshape(4)
    return np.concatenate( [trans, quat])

def numpy_2_pcd(pcd_np, color = None ):

    # pcd_np = np.array(pcd_np)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    if(color is not None):
        color_np = np.zeros(pcd_np.shape)
        color_np[:,] = color
        pcd.colors = o3d.utility.Vector3dVector(color_np)
    return pcd

def pcd_2_numpy(pcd , scaling = False):

    pcd_np = np.asarray(pcd.points)
    if(scaling == True):
        pcd_np = np.concatenate( [pcd_np, np.ones(pcd_np.shape[0],1)], axis = 1)

    return pcd_np

def pb2mc_update(
        recorder: PyBulletMeshcat, 
        mc_vis: meshcat.Visualizer, 
        stop_event: threading.Event, 
        run_event: threading.Event) -> None:
    iters = 0
    # while True:
    while not stop_event.is_set():
        run_event.wait()
        iters += 1
        try:
            recorder.add_keyframe()
            recorder.update_meshcat_current_state(mc_vis)
        except KeyError as e:
            print(f'PyBullet to Meshcat thread Exception: {e}')
            time.sleep(0.1)
        time.sleep(1/230.0)


def main(args: config_util.AttrDict) -> None:

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    infer_kwargs = {}
    #####################################################################################
    # Set up the models (feat encoder, voxel affordance, pose refinement, success)

    model_ckpt_logdir = osp.join(path_util.get_anyplace_model_weights(), args.experiment.logdir) 
    reso_grid = args.data.voxel_grid.reso_grid  # args.reso_grid
    padding_grid = args.data.voxel_grid.padding
    raster_pts = three_util.get_raster_points(reso_grid, padding=padding_grid)
    raster_pts = raster_pts.reshape(reso_grid, reso_grid, reso_grid, 3)
    raster_pts = raster_pts.transpose(2, 1, 0, 3)
    raster_pts = raster_pts.reshape(-1, 3)
    rot_grid_samples = args.data.rot_grid_samples
    rot_grid = util.generate_healpix_grid(size=rot_grid_samples) 
    args.data.rot_grid_bins = rot_grid.shape[0]
    exp_args = args.experiment

    mc_vis = None

    # Load pose refinement model, loss, and optimizer
    pose_refine_model_path = None
    pr_model = None
    if exp_args.load_pose_regression:
        # assumes model path exists
        pose_refine_model_path = args.experiment.eval.ckpt_path
        print(f'!!!!!!!!!!!!! Model path: {pose_refine_model_path}')
        pose_refine_ckpt = torch.load(pose_refine_model_path, map_location=torch.device('cpu'))

        # update config with config used during training
        config_util.update_recursive(args.model.refine_pose, config_util.recursive_attr_dict(pose_refine_ckpt['args']['model']['refine_pose']))

        # model (pr = pose refine)
        pr_type = args.model.refine_pose.type
        pr_args = config_util.copy_attr_dict(args.model[pr_type])
        if args.model.refine_pose.get('model_kwargs') is not None:
            custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
            config_util.update_recursive(pr_args, custom_pr_args)

        if pr_type == 'nsm_transformer':
            print("!!!!!!!!!!!!! AnyPlace Diffusion !!!!!!!!!!!!!")
            pr_model_cls = NSMTransformerSingleTransformationRegression
            pr_model = pr_model_cls(
                mc_vis=mc_vis, 
                feat_dim=args.model.refine_pose.feat_dim, 
            **pr_args).cuda()
        elif pr_type == 'nsm_implicit':
            print("!!!!!!!!!!!!! Using NSMTransformerImplicit !!!!!!!!!!!!!")
            pr_model_cls = NSMTransformerImplicit
            pr_model = pr_model_cls(
                mc_vis=None, 
                feat_dim=args.model.refine_pose.feat_dim, 
                is_train=False,
                **pr_args).cuda()

        pr_model.load_state_dict(pose_refine_ckpt['refine_pose_model_state_dict'])
    print("Done loading")
    # Load success classifier and optimizer
    success_model = None
    coarse_aff_model = None

    #####################################################################################
    # prepare function used for inference using set of trained models
    if args.experiment.eval.model_type == 'anyplace_diffusion_molmocrop': # Using this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        infer_relation_policy = policy_inference_methods_dict[args.experiment.eval.inference_method]
    elif args.experiment.eval.model_type == 'nsm_implicit':
        infer_relation_policy = nsm_policy_inference_methods_dict[args.experiment.eval.inference_method]


    
    # Save full name of model paths used
    if exp_args.load_pose_regression:
        args.experiment.eval.pose_refine_model_name_full = pose_refine_model_path

    if util.exists_and_true(exp_args.eval, 'multi_aff_rot'):
        infer_kwargs['multi_aff_rot'] = True

    eval_dataset_path = args.experiment.eval.eval_dataset_path
    eval_dataset_list = sorted( [os.path.join(eval_dataset_path, f) for f in os.listdir(eval_dataset_path) if os.path.isfile(os.path.join(eval_dataset_path, f))] )

    print("eval_dataset_list: ", eval_dataset_list)
    num_clons = exp_args.eval.init_k_val 
    for data_path in eval_dataset_list:
        # print("data: ", data_path)
        print(f'!!!!!!!!! Running inference for {data_path} !!!!!!!!')
        # print(str(data_path)[-6:-4])

        # print(f"./anyplace_result/test{str(data_path)[-6:-4]}.npy",)
        exp_args.num_iterations = 1

        use_molmo_crop = False   # this is to prevent cropping the base object like pegs in insertion eval
        episode = np.load(data_path , allow_pickle=True)
        episode = episode.item()
        
        base_obj_pc = numpy_2_pcd(episode['pcd'][0]).transform( get_transform(episode['world_frame_pose'][0]) )
        base_obj_pc = pcd_2_numpy( base_obj_pc )

        grasp_obj_pc = numpy_2_pcd(episode['pcd'][1]).transform( get_transform(episode['world_frame_pose'][1]) )
        grasp_obj_pc = pcd_2_numpy( grasp_obj_pc )

        # Todo change to world frame

        for iteration in range(exp_args.start_iteration, exp_args.num_iterations):

            crop_base_obj_pc = base_obj_pc
            num_clons = 100  # for object that does not need cropping, do 100 iterations

            if crop_base_obj_pc.shape[0] < 10:
                continue
 
            
            scene_extents = args.data.coarse_aff.scene_extents 
            scene_scale = 1 / np.max(scene_extents)

            args.data.coarse_aff.scene_scale = scene_scale 
            infer_kwargs['gt_child_cent'] = None
            infer_kwargs['export_viz'] = args.export_viz
            infer_kwargs['export_viz_dirname'] = args.export_viz_dirname
            infer_kwargs['export_viz_relative_trans_guess'] = None
            infer_kwargs['iteration'] = iteration
            
            multi_mesh_dict = None

            #base_obj_pc  = crop_base_obj_pc
            parent_pcd = base_obj_pc
            child_pcd_guess = grasp_obj_pc
            relative_trans_preds = infer_relation_policy(
                mc_vis, 
                parent_pcd, child_pcd_guess, 
                coarse_aff_model,
                pr_model, 
                success_model,
                scene_mean=args.data.coarse_aff.scene_mean, scene_scale=args.data.coarse_aff.scene_scale, 
                grid_pts=raster_pts, rot_grid=rot_grid, 
                viz=False, n_iters=exp_args.eval.n_refine_iters, 
                no_parent_crop=(not exp_args.parent_crop),
                return_top=(not exp_args.eval.return_rand), with_coll=exp_args.eval.with_coll, 
                run_affordance=exp_args.eval.run_affordance, init_k_val=num_clons,
                no_sc_score=exp_args.eval.no_success_classifier, 
                init_parent_mean=exp_args.eval.init_parent_mean_pos, init_orig_ori=exp_args.eval.init_orig_ori,
                refine_anneal=exp_args.eval.refine_anneal,
                mesh_dict=multi_mesh_dict,
                add_per_iter_noise=exp_args.eval.add_per_iter_noise,
                per_iter_noise_kwargs=exp_args.eval.per_iter_noise_kwargs,
                variable_size_crop=exp_args.eval.variable_size_crop,
                timestep_emb_decay_factor=exp_args.eval.timestep_emb_decay_factor,
                remove_redundant_pose = args.experiment.eval.remove_redundant_pose,
                **infer_kwargs)

            # save data
            data = {"relative": relative_trans_preds}
            # print("output: ", relative_trans_preds.shape)

            # save data
            result = copy.deepcopy(episode)
            print("relative_trans_preds: ", )
            result['anyplace_result'] = relative_trans_preds
            save_data_dir = "./anyplace_result"
            OUTPUT_DIR = Path(save_data_dir)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            np.save(f"./anyplace_result/test{str(data_path)[-6:-4]}.npy", result, allow_pickle = True)



if __name__ == "__main__":
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_fname', type=str, required=True, help='Name of config file')
    parser.add_argument('-d', '--debug', action='store_true', help='If True, run in debug mode')
    parser.add_argument('-dd', '--debug_data', action='store_true', help='If True, run data loader in debug mode')
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-l', '--local_dataset_dir', type=str, default=None, help='If the data is saved on a local drive, pass the root of that directory here')
    parser.add_argument('-ex', '--export_viz', action='store_true', help='If True, save data for post-processed visualization')
    parser.add_argument('--export_viz_dirname', type=str, default='rpdiff_export_viz')
    parser.add_argument('-nm', '--new_meshcat', action='store_true')

    args = parser.parse_args()
    eval_args = config_util.load_config(osp.join(path_util.get_eval_config_dir(), args.config_fname), demo_train_eval='eval')
    eval_args['debug'] = args.debug
    eval_args['debug_data'] = args.debug_data
    eval_args['port_vis'] = args.port_vis
    eval_args['seed'] = args.seed
    eval_args['local_dataset_dir'] = args.local_dataset_dir
    
    # other runtime options
    eval_args['export_viz'] = args.export_viz
    eval_args['export_viz_dirname'] = args.export_viz_dirname
    
    # if we want to override the port setting for meshcat and directly start our own
    eval_args['new_meshcat'] = args.new_meshcat
    eval_args = config_util.recursive_attr_dict(eval_args)

    main(eval_args)
