import os
import h5py
import json
import torch
import imageio
import numpy as np

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


np.random.seed(0)


def png_i(f):
    return imageio.imread(f)


class processor:
    def __init__(self, basedir, mani_mode, testskip=1):
        super(processor, self).__init__()
        self.basedir = basedir
        self.mani_mode = mani_mode
        self.testskip = testskip
        # self.rgbs, self.pose, self.split = self.load_rgb()

    def load_gts(self):
        poses = []

        fname = os.path.join(self.basedir, f'indoor_{self.mani_mode}_test', 'rgbs')
        imagefile = [os.path.join(fname, f) for f in sorted(os.listdir(fname))]
        rgbs = [png_i(f) for f in imagefile]

        posefile = os.path.join(self.basedir, f'transforms.json')
        with open(posefile, 'r') as read_pose:
            meta = json.load(read_pose)

        angle_x = meta['camera_angle_x']
        for frame in meta['frames'][::self.testskip]:
            poses.append(frame['transform_matrix'])
        poses = np.array(poses).astype(np.float32)

        index = np.arange(0, len(rgbs), self.testskip)
        rgbs = np.array(rgbs)[index]
        rgbs = (rgbs / 255.).astype(np.float32)  # keep all 3 channels (RGB)

        ins_path = os.path.join(self.basedir, f'indoor_{self.mani_mode}_test', 'semantic_instance')
        ins_files = [os.path.join(ins_path, f) for f in sorted(os.listdir(ins_path))]
        gt_labels = np.array([png_i(f) for f in ins_files])
        gt_labels = gt_labels[index]

        f = os.path.join(self.basedir, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()

        return rgbs[..., :3], poses, gt_labels, ins_rgbs, angle_x


class ins_processor:
    def __init__(self, base_path, weakly_mode, weakly_value, testskip=1):
        super(ins_processor, self).__init__()
        self.weakly_mode = weakly_mode
        self.weakly_value = weakly_value
        self.base_path = base_path
        self.testskip = testskip

        # load operation
        self.gt_labels, self.ins_rgbs = self.load_semantic_instance()
        self.ins_num = len(self.ins_rgbs)
        # self.unique_labels = np.unique(self.gt_labels)

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
            gt_labels = np.array([png_i(f) for f in ins_files])

            index = np.arange(0, len(gt_labels), skip)
            gt_labels = gt_labels[index]
            all_ins.append(gt_labels)

        gt_labels = np.concatenate(all_ins, 0)
        f = os.path.join(self.base_path, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        return gt_labels, ins_rgbs


def load_mani_poses(args):
    load_path = os.path.join(args.datadir, 'transformation_matrix.json')
    with open(load_path, 'r') as rf:
        mani_poses = json.load(rf)
    rf.close()
    transformations = mani_poses['transformations']
    return transformations


def load_data(args):
    # load color image RGB
    imgs, poses, gt_labels, ins_rgbs, camera_angle_x = processor(args.datadir, args.mani_mode,
                                                                 testskip=args.testskip).load_gts()
    ins_num = len(ins_rgbs)
    H, W = imgs[0].shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([[focal, 0, W * 0.5], [0, -focal, H * 0.5], [0, 0, -1]])
    hwk = [int(H), int(W), K]

    return imgs, poses, hwk, gt_labels, ins_rgbs, ins_num
