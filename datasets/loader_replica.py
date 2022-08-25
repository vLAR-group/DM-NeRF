import os
import h5py
import json
import torch
import imageio
import numpy as np

np.random.seed(0)

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def png_i(f):
    return imageio.imread(f)


class processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_rgb(self):
        # testskip operation

        objs_info_fname = os.path.join(self.basedir, 'objs_info.json')
        with open(objs_info_fname, 'r') as f_obj_info:
            objs_info = json.load(f_obj_info)
        f_obj_info.close()
        objs = objs_info["objects"]
        view_id = objs_info["view_id"]
        ins_map = objs_info["ins_map"]

        _, _, dataset_name, scene_name = self.basedir.split('/')
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)
        selected_test_ids = np.array(self.test_ids)[skip_idx]
        gt_id = self.train_ids[view_id]
        # gt_id = selected_test_ids[view_id[scene_name]]

        # load poses
        traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
        Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        poses = Ts_full[self.train_ids]

        color_f = os.path.join(self.basedir, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()

        return objs, view_id, ins_map, poses, ins_rgbs


class rgb_processor:
    def __init__(self, basedir, train_ids, test_ids, testskip=1):
        super(rgb_processor, self).__init__()
        self.basedir = basedir
        self.testskip = testskip
        self.train_ids = train_ids
        self.test_ids = test_ids

    def load_rgb(self):
        # testskip operation
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)

        # load poses
        traj_file = os.path.join(self.basedir, 'traj_w_c.txt')
        Ts_full = np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
        train_poses = Ts_full[self.train_ids]
        test_poses = Ts_full[self.test_ids]
        test_poses = test_poses[skip_idx]
        poses = np.concatenate([train_poses, test_poses], axis=0)

        # load rgbs
        rgb_basedir = os.path.join(self.basedir, 'rgb')
        train_rgbs = [png_i(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.train_ids]
        test_rgbs = [png_i(os.path.join(rgb_basedir, f'rgb_{idx}.png')) for idx in self.test_ids]
        train_rgbs = np.array(train_rgbs)
        test_rgbs = np.array(test_rgbs)[skip_idx]
        rgbs = np.concatenate([train_rgbs, test_rgbs], axis=0)
        rgbs = (rgbs / 255.).astype(np.float32)

        i_train = np.arange(0, len(self.train_ids), 1)
        i_test = np.arange(len(self.train_ids), len(self.train_ids) + len(skip_idx), 1)
        i_split = [i_train, i_test]

        return rgbs, poses, i_split


class ins_processor:
    def __init__(self, base_path, train_ids, test_ids, weakly_mode, weakly_value, testskip=1):
        super(ins_processor, self).__init__()
        self.weakly_mode = weakly_mode
        self.weakly_value = weakly_value
        self.base_path = base_path
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.testskip = testskip

        # load operation
        self.gt_labels, self.ins_rgbs = self.load_semantic_instance()
        self.ins_num = len(self.ins_rgbs)

    def load_semantic_instance(self):
        skip_idx = np.arange(0, len(self.test_ids), self.testskip)
        ins_base_path = os.path.join(self.base_path, 'semantic_instance')
        train_sem_ins = [png_i(os.path.join(ins_base_path, f'semantic_instance_{idx}.png')) for idx in self.train_ids]
        train_sem_ins = np.array(train_sem_ins).astype(np.float32)
        test_sem_ins = [png_i(os.path.join(ins_base_path, f'semantic_instance_{idx}.png')) for idx in self.test_ids]
        test_sem_ins = np.array(test_sem_ins)[skip_idx].astype(np.float32)

        gt_labels = np.concatenate([train_sem_ins, test_sem_ins], 0)
        color_f = os.path.join(self.base_path, 'ins_rgb.hdf5')
        with h5py.File(color_f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        return gt_labels, ins_rgbs


def load_mani_poses(args):
    load_path = os.path.join(args.datadir, 'transformation_matrix.json')
    with open(load_path, 'r') as rf:
        obj_trans = json.load(rf)
    rf.close()
    return obj_trans


def load_data(args):
    # load color image RGB
    total_num = 900
    step = 5
    train_ids = list(range(0, total_num, step))
    test_ids = [x + step // 2 for x in train_ids]
    if args.editor_demo:
        objs, view_id, ins_map, poses, ins_rgbs = processor(args.datadir, train_ids, test_ids,
                                                            testskip=args.testskip).load_rgb()

        if view_id is not None:
            view_poses = np.repeat(poses[view_id][np.newaxis, ...], args.views, axis=0)
        else:
            view_poses = torch.stack(
                [pose_spherical(angle, -65.0, 7.0) for angle in np.linspace(-180, 180, args.views)], 0)

        ins_num = len(ins_rgbs)
        H, W = int(480), int(640)
        focal = W / 2.0
        K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])
        hwk = [int(H), int(W), K]

        return objs, view_poses, ins_map, poses, hwk, ins_rgbs, ins_num
    else:
        imgs, poses, i_split = rgb_processor(args.datadir, train_ids, test_ids, testskip=args.testskip).load_rgb()
        # add another load class which assigns to semantic labels

        # load instance labels
        ins_info = ins_processor(args.datadir, train_ids, test_ids, None, None, testskip=args.testskip)
        gt_labels = ins_info.gt_labels
        ins_rgbs = ins_info.ins_rgbs

        H, W = imgs[0].shape[:2]

        focal = W / 2.0
        K = np.array([[focal, 0, (W - 1) * 0.5], [0, focal, (H - 1) * 0.5], [0, 0, 1]])
        hwk = [int(H), int(W), K]

        return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_info.ins_num
