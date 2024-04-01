# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

import numpy as np
import torch
import trimesh
import warnings

warnings.filterwarnings("ignore",
                        message="^The given NumPy array is not writable,")  # ignore manopth warning

from manopth.manolayer import ManoLayer
from manopth.rodrigues_layer import batch_rodrigues
from scipy.spatial.transform import Rotation as R
from trimesh.viewer import SceneViewer

import contactopt.arguments as arguments
import contactopt.util as util
from contactopt.hand_object import HandObject
from contactopt.run_contactopt import run_contactopt


def vis_hand_obj(hand_full_pose_pca, hand_trans, obj_mesh, v_template):
    mano_model_layer = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False, center_idx=None)
    mano_model_layer.v_template = v_template

    verts, joints = mano_model_layer(torch.tensor(hand_full_pose_pca), th_trans=torch.tensor(hand_trans))
    faces = mano_model_layer.th_faces.detach().cpu()

    hand_mesh = trimesh.Trimesh(vertices=(verts[0].numpy() / 1000), faces=faces, process=False)
    hand_mesh.visual.face_colors = [244, 191, 175, 255]

    scene = trimesh.Scene([obj_mesh, hand_mesh])
    viewer = SceneViewer(scene=scene, resolution=(800, 800), flags={'wireframe': True, 'axis': True})


def vis_output(fp, out_dict):
    # in
    seq_dict = np.load(fp, allow_pickle=True).item()
    n_samples = seq_dict['global_rotation'].shape[0]

    in_obj_fp = seq_dict['obj_mesh_fp']
    print(in_obj_fp)

    in_global_rotation = torch.tensor(seq_dict['global_rotation'])
    in_wrist_trans = torch.tensor(seq_dict["transl"])
    in_obj_trans = torch.tensor(seq_dict["obj_transl"].copy())
    in_obj_rot = torch.tensor(seq_dict["obj_rot"].copy())
    in_obj_rot_mats = batch_rodrigues(in_obj_rot.view(-1, 3)).view([-1, 3, 3])

    # out
    out_wrist_rot = torch.tensor(out_dict["out_pose_pca_15_flat_hand_false"][:, :3])
    out_wrist_rot_mats = batch_rodrigues(out_wrist_rot).view(out_wrist_rot.shape[0], 3, 3)

    r_hand_pose_pca = torch.tensor(out_dict["out_pose_pca_15_flat_hand_false"][:, 3:])
    n_comps = r_hand_pose_pca.shape[1]

    out_wrist_transl = out_dict["out_mTc"][:, :3, 3]
    out_in_wrist_transl_diff = out_wrist_transl - in_wrist_trans.numpy()

    out_obj_rot = torch.tensor(out_dict["obj_rot_rotvec"])
    out_obj_rot_mats = batch_rodrigues(out_obj_rot.view(-1, 3)).view([-1, 3, 3])

    # vis
    obj_mesh = trimesh.load(in_obj_fp)
    obj_vertices = torch.matmul(torch.tensor(obj_mesh.vertices), in_obj_rot_mats) + in_obj_trans.unsqueeze(dim=1)

    obj_verts = util.apply_rot(out_obj_rot_mats[0, :, :].unsqueeze(0),
                               obj_vertices[0].clone().unsqueeze(0),
                               around_centroid=True).squeeze(0)
    assert (torch.allclose(obj_verts[:], obj_vertices[0]))  # no obj rotation!?
    obj_mesh.vertices = obj_verts

    # obj_mesh.vertices = util.apply_rot(torch.tensor(out_dict["out_mTc"][0, :3, :3]).T.unsqueeze(0),
    #                                    torch.tensor(obj_mesh.vertices).to(dtype=torch.float32).unsqueeze(0),
    #                                    around_centroid=False).squeeze(0)
    obj_mesh.vertices -= out_in_wrist_transl_diff[0]

    wrist_rot_mat = torch.tensor(out_dict["out_mTc"][:, :3, :3]).bmm(out_wrist_rot_mats)
    wrist_rot_rotvec = torch.tensor(R.from_matrix(wrist_rot_mat).as_rotvec())

    hand_full_pose_pca = torch.hstack([wrist_rot_rotvec, r_hand_pose_pca]).to(dtype=torch.float32)

    vis_hand_obj(hand_full_pose_pca, hand_trans=in_wrist_trans, obj_mesh=obj_mesh, v_template=None)


def create_demo_dataset(in_pose_fp, vis=False):
    fp = in_pose_fp

    seq_dict = np.load(fp, allow_pickle=True).item()
    n_samples = seq_dict['global_rotation'].shape[0]

    in_obj_fp = seq_dict['obj_mesh_fp']
    print(in_obj_fp)

    mano_model_layer = ManoLayer(flat_hand_mean=False, ncomps=15, use_pca=True)
    mano_model_layer.th_v_template = seq_dict['v_template']

    mano_beta = mano_model_layer.th_betas.numpy().squeeze()  # use default shape parameters

    global_rotation = seq_dict['global_rotation']
    hand_full_pose_rotvec = seq_dict["full_pose_rotvec"]
    hand_trans = seq_dict["transl"]
    obj_trans = seq_dict["obj_transl"]
    obj_rot = seq_dict["obj_rot"]

    hand_full_pose_rotvec = np.hstack([global_rotation, hand_full_pose_rotvec])

    mano_betas = np.repeat(mano_beta.reshape(1, -1), n_samples, axis=0)

    # Axis angle to pca pose
    hand_full_pose_pca = util.fit_pca_to_axang_tensor(mano_pose=hand_full_pose_rotvec, mano_beta=mano_betas)
    hand_full_pose_pca = np.hstack([global_rotation, seq_dict['hand_pose_pca_flat_hand_false'][:, :15]])

    obj_mesh = trimesh.load(in_obj_fp)

    # Rotate and translate object
    obj_rot = torch.tensor(obj_rot)
    rot_mats = batch_rodrigues(obj_rot.view(-1, 3)).view([-1, 3, 3])

    obj_vertices = torch.matmul(torch.tensor(obj_mesh.vertices), rot_mats) + torch.tensor(obj_trans).unsqueeze(dim=1)

    obj_mesh.vertices = obj_vertices[0]

    if vis:
        vis_hand_obj(hand_full_pose_pca, hand_trans, obj_mesh, v_template=seq_dict["v_template"])

    dataset_list = []
    for j in range(n_samples):
        # Initialize the HandObject class with the given mano parameters and object mesh.
        # Note that pose must be represented using the 15-dimensional PCA space
        ho_pred = HandObject()
        ho_pred.load_from_mano_params(hand_beta=mano_beta.tolist(), hand_pose=hand_full_pose_pca[j], hand_trans=hand_trans[j],
                                      obj_faces=obj_mesh.faces, obj_verts=obj_vertices[j])

        # To make the dataloader happy, we need a "ground truth" H/O set.
        # However, since this isn't used for this demo, just copy the ho_pred object.
        ho_gt = HandObject()
        ho_gt.load_from_ho(ho_pred)
        ho_gt.hand_contact = ho_pred.hand_contact

        new_sample = dict()
        new_sample['ho_aug'] = ho_pred
        new_sample['ho_gt'] = ho_gt

        # Select the random object vertices which will be sampled
        new_sample['obj_sampled_idx'] = np.random.randint(0, len(ho_gt.obj_verts), util.SAMPLE_VERTS_NUM)

        # Calculate hand and object features. The network uses these for improved performance.
        new_sample['hand_feats_aug'], new_sample['obj_feats_aug'] = ho_pred.generate_pointnet_features(new_sample['obj_sampled_idx'])

        dataset_list.append(new_sample)

    return dataset_list, seq_dict['sequence_cleaned_fp']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alignment networks training')
    parser.add_argument('--in_pose_fp', type=str)
    args = parser.parse_args()
    import sys

    if sys.argv[1] == "--in_pose_fp":
        del sys.argv[1:3]

    in_pose_fp = args.in_pose_fp
    vis_on = False

    dataset, out_fp = create_demo_dataset(in_pose_fp, vis=vis_on)

    args = arguments.run_contactopt_parse_args()

    # TODO add to args
    no_rot_and_trans = False

    defaults = {'lr': 0.01,
                'n_iter': 250,
                'w_cont_hand': 2.5,
                'sharpen_thresh': -1,
                'ncomps': 15,
                'w_cont_asym': 2,
                'w_opt_trans': 0.0 if no_rot_and_trans else 0.1,  # 0.3
                'w_opt_rot': 0 if no_rot_and_trans else 1,
                'w_opt_pose': 1.0,
                'caps_rad': 0.005,  # 0.001
                'cont_method': 0,
                'caps_top': 0.0005,
                'caps_bot': -0.001,
                'w_pen_cost': 800,  # 320
                'pen_it': 0,
                'rand_re': 0,  # default: 8
                'rand_re_trans': 0.0,  # default: 0.02
                'rand_re_rot': 0,  # default: 5
                'w_obj_rot': 0,
                'vis_method': 1}

    for k in defaults.keys():
        if vars(args)[k] is None:
            vars(args)[k] = defaults[k]

    args.test_dataset = dataset
    args.split = 'user'

    args.batch_size = 128

    out_dict = run_contactopt(args)

    print("Saving results with n-frames:", out_dict['out_pose'].shape[0])

    obj_rot = out_dict["obj_rot"]

    obj_rot_rotvec = R.from_matrix(obj_rot).as_rotvec()

    out_dict['out_pose_pca_15_flat_hand_false'] = out_dict["out_pose"]
    out_dict["obj_rot_rotvec"] = obj_rot_rotvec

    del out_dict["out_pose"]
    del out_dict["obj_rot"]

    mano_model = ManoLayer(mano_root='mano/models', use_pca=True, ncomps=15, side='right', flat_hand_mean=False)

    th_pose_coeffs = torch.tensor(out_dict['out_pose_pca_15_flat_hand_false'])

    th_hand_pose_coeffs = th_pose_coeffs[:, 3:3 + 15]
    # PCA components --> axis angles
    th_full_hand_pose = th_hand_pose_coeffs.mm(mano_model.th_selected_comps)

    # Concatenate global rot with pose
    th_full_pose = torch.cat([
        th_pose_coeffs[:, :3],
        mano_model.th_hands_mean + th_full_hand_pose
    ], 1)

    out_dict['full_hand_pose_rotvec'] = th_full_pose.detach().numpy()

    if vis_on:
        vis_output(in_pose_fp, out_dict)

    out_dir = os.path.dirname(out_fp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    np.save(file=out_fp, arr=out_dict)

    exit(1)
