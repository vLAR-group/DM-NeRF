import os
import h5py
import json
import torch
import imageio
import numpy as np
from tools.pose_generator import pose_spherical

np.random.seed(0)


class processor:
    def __init__(self, basedir, mani_mode, testskip=1):
        super(processor, self).__init__()
        self.basedir = basedir
        self.mani_mode = mani_mode
        self.testskip = testskip

    def load_gts(self):
        poses = []

        fname = os.path.join(self.basedir, 'mani', self.mani_mode, 'rgbs')
        imagefile = [os.path.join(fname, f) for f in sorted(os.listdir(fname))]
        rgbs = [imageio.imread(f) for f in imagefile]

        posefile = os.path.join(self.basedir, 'mani', 'transforms.json')
        with open(posefile, 'r') as read_pose:
            meta = json.load(read_pose)

        angle_x = meta['camera_angle_x']
        for frame in meta['frames'][::self.testskip]:
            poses.append(frame['transform_matrix'])
        poses = np.array(poses).astype(np.float32)

        index = np.arange(0, len(rgbs), self.testskip)
        rgbs = np.array(rgbs)[index]
        rgbs = (rgbs / 255.).astype(np.float32)  # keep all 3 channels (RGB)

        ins_path = os.path.join(self.basedir, 'mani', self.mani_mode, 'semantic_instance')
        ins_files = [os.path.join(ins_path, f) for f in sorted(os.listdir(ins_path))]
        gt_labels = np.array([imageio.imread(f) for f in ins_files])
        gt_labels = gt_labels[index]

        f = os.path.join(self.basedir, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()

        return rgbs[..., :3], poses, gt_labels, ins_rgbs, angle_x


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
