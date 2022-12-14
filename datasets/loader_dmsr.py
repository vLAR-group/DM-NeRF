import os
import h5py
import json
import torch
import imageio
import numpy as np
from tools.pose_generator import pose_spherical

np.random.seed(0)


class rgb_processor:
    def __init__(self, args):
        super(rgb_processor, self).__init__()
        self.basedir = args.datadir
        self.testskip = args.testskip
        self.args = args

    def load_rgb(self):
        splits = ['train', 'test']
        all_rgb = []
        all_pose = []
        counts = [0]
        angle_x = None
        for s in splits:
            poses = []
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            fname = os.path.join(self.basedir, s, 'rgbs')

            imagefile = [os.path.join(fname, f) for f in sorted(os.listdir(fname))]

            rgbs = [imageio.imread(f) for f in imagefile]

            posefile = os.path.join(self.basedir, s, 'transforms.json')
            with open(posefile, 'r') as read_pose:
                meta = json.load(read_pose)

            angle_x = meta['camera_angle_x']
            for frame in meta['frames'][::skip]:
                poses.append(frame['transform_matrix'])
            poses = np.array(poses).astype(np.float32)

            index = np.arange(0, len(rgbs), skip)
            rgbs = np.array(rgbs)[index]
            rgbs = (rgbs / 255.).astype(np.float32)  # keep all 3 channels (RGB)
            counts.append(counts[-1] + rgbs.shape[0])
            all_rgb.append(rgbs)
            all_pose.append(poses)

        all_rgb = np.concatenate(all_rgb, 0)
        all_pose = np.concatenate(all_pose, 0)

        if all_pose.shape[-1] == 16:
            all_pose = all_pose.reshape((all_pose.shape[0], 4, 4))

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(2)]

        if self.args.mesh or self.args.mani_demo:
            if self.args.mani_type == 'rigid':
                objs_info_fname = os.path.join(self.basedir, 'mani', 'objs_info_rigid.json')
            else:
                objs_info_fname = os.path.join(self.basedir, 'mani', 'objs_info_deform.json')
            with open(objs_info_fname, 'r') as f_obj_info:
                objs_info = json.load(f_obj_info)
            f_obj_info.close()
            objs = objs_info["objects"]
            view_id = objs_info["view_id"]
            ins_map = objs_info["ins_map"]
        else:
            objs, view_id, ins_map = None, None, None

        return all_rgb, all_pose, i_split, angle_x, objs, view_id, ins_map


class ins_processor:
    def __init__(self, base_path, testskip=1):
        super(ins_processor, self).__init__()
        self.base_path = base_path
        self.testskip = testskip

        # load operation
        self.gt_labels, self.ins_rgbs = self.load_semantic_instance()
        self.ins_num = len(self.ins_rgbs)
        self.unique_labels = np.unique(self.gt_labels)

    def load_semantic_instance(self):
        splits = ['train', 'test']
        all_ins = []
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip

            ins_path = os.path.join(self.base_path, s, 'semantic_instance')
            ins_files = [os.path.join(ins_path, f) for f in sorted(os.listdir(ins_path))]
            gt_labels = np.array([imageio.imread(f) for f in ins_files])

            index = np.arange(0, len(gt_labels), skip)
            gt_labels = gt_labels[index]
            all_ins.append(gt_labels)

        gt_labels = np.concatenate(all_ins, 0)
        f = os.path.join(self.base_path, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        return gt_labels, ins_rgbs


def load_data(args):
    # load color image RGB
    imgs, poses, i_split, camera_angle_x, objs, view_id, ins_map = rgb_processor(args).load_rgb()
    # add another load class which assigns to semantic labels
    if args.is_train:
        view_poses = None
    else:
        if view_id is not None:
            view_poses = np.repeat(poses[view_id][np.newaxis, ...], args.views, axis=0)
        else:
            view_poses = torch.stack(
                [pose_spherical(angle, -65.0, 7.0) for angle in np.linspace(0, 180, args.views)], 0)
            # print(view_poses.shape)

    imgs = imgs[..., :3]
    # load instance labels
    ins_info = ins_processor(args.datadir, testskip=args.testskip)
    gt_labels = ins_info.gt_labels
    ins_rgbs = ins_info.ins_rgbs

    H, W = imgs[0].shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([[focal, 0, W * 0.5], [0, -focal, H * 0.5], [0, 0, -1]])
    hwk = [int(H), int(W), K]

    return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_info.ins_num, objs, view_poses, ins_map
