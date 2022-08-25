import os
import h5py
import cv2
import json
import imageio
import numpy as np


def load_mani_poses(args):
    load_path = os.path.join(args.datadir, 'transformation_matrix.json')
    with open(load_path, 'r') as rf:
        obj_trans = json.load(rf)
    rf.close()
    return obj_trans


def crop_data(H, W, crop_size):
    # center_w, center_h = H // 2, W // 2
    crop_mask = np.zeros(shape=(H, W))
    new_w, new_h = crop_size
    margin_h = (H - new_h) // 2
    margin_w = (W - new_w) // 2
    crop_mask[margin_h: (H - margin_h), margin_w: (W - margin_w)] = 1
    return crop_mask.astype(np.int8)


def resize(data, H=480, W=640):
    imgs_half_res = None
    if len(data.shape) == 3:
        imgs_half_res = np.zeros((data.shape[0], H, W))
    elif len(data.shape) == 4:
        imgs_half_res = np.zeros((data.shape[0], H, W, 3))
    for i, data_i in enumerate(data):
        imgs_half_res[i] = cv2.resize(data_i, (W, H), interpolation=cv2.INTER_NEAREST)
    data = imgs_half_res
    return data


class img_processor:

    def __init__(self, data_dir, testskip=1, resize=True):
        super(img_processor, self).__init__()
        self.data_dir = data_dir
        self.testskip = testskip
        self.resize = resize
        self.images = None
        self.poses = None
        self.depths = None
        self.i_split = None

    @staticmethod
    def img_i(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    @staticmethod
    def txt_i(f):
        txt = np.loadtxt(f, delimiter=' ')
        return txt

    def load_rgb(self):
        splits = ['train', 'test']
        all_rgb = []
        all_pose = []
        counts = [0]
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip
            indices = np.loadtxt(os.path.join(self.data_dir, f'{s}_split_idx.txt')).astype(np.int16)

            file_train = os.path.join(self.data_dir, s)

            rgb_fnames = [os.path.join(file_train, f'{s}_images', f'{index}.jpg') for index in indices]
            rgbs = [self.img_i(f) for f in rgb_fnames]
            pose_fnames = [os.path.join(file_train, f'{s}_pose', f'{index}.txt') for index in indices]
            poses = [self.txt_i(f) for f in pose_fnames]

            index = np.arange(0, len(poses), skip)

            rgbs = np.array(rgbs)[index]
            poses = np.array(poses)[index]
            rgbs = (rgbs / 255.).astype(np.float32)
            if self.resize:
                rgbs = resize(rgbs)
            counts.append(counts[-1] + rgbs.shape[0])
            all_rgb.append(rgbs)
            all_pose.append(poses)

        all_rgb = np.concatenate(all_rgb, 0)
        all_pose = np.concatenate(all_pose, 0)

        i_split = [np.arange(counts[i], counts[i + 1]) for i in range(2)]
        if not self.resize:
            intrinsic_f = os.path.join(self.data_dir, 'intrinsic', 'intrinsic_color.txt')
            intrinsic = np.loadtxt(intrinsic_f, delimiter=' ')
        else:
            intrinsic_f = os.path.join(self.data_dir, 'intrinsic', 'intrinsic_depth.txt')
            intrinsic = np.loadtxt(intrinsic_f, delimiter=' ')
        print("data load finish")
        return all_rgb, all_pose, i_split, intrinsic


class ins_processor:
    def __init__(self, data_dir, testskip=1, resize=True, weakly_value=1.):
        super(ins_processor, self).__init__()
        self.data_dir = data_dir
        self.testskip = testskip
        self.resize = resize
        self.weakly_value = weakly_value

    def ins_npz_i(self, f):
        npz = np.load(f)
        ins_map = npz.f.ins_2d_label_id
        return ins_map

    def ins_png_i(self, f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    def load_semantic_instance(self):
        splits = ['train', 'test']
        all_ins = []
        for s in splits:
            if s == 'train' or self.testskip == 0:
                skip = 1
            else:
                skip = self.testskip
            indices = np.loadtxt(os.path.join(self.data_dir, f'{s}_split_idx.txt')).astype(np.int16)
            file_train = os.path.join(self.data_dir, s)

            ins_fnames = [os.path.join(file_train, f'{s}_ins', f'{index}.npz') for index in indices]
            gt_labels = np.array([self.ins_npz_i(f) for f in ins_fnames])
            index = np.arange(0, len(gt_labels), skip)
            gt_labels = gt_labels[index]
            if self.resize:
                gt_labels = resize(gt_labels)
            all_ins.append(gt_labels)

        gt_labels = np.concatenate(all_ins, 0).astype(np.int8)
        f = os.path.join(self.data_dir, 'ins_rgb.hdf5')
        with h5py.File(f, 'r') as f:
            ins_rgbs = f['datasets'][:]
        f.close()
        unique_labels = np.unique(gt_labels)
        ins_num = len(unique_labels) - 1
        ins_rgbs = ins_rgbs[:ins_num]
        gt_labels[gt_labels == -1] = ins_num
        return gt_labels, ins_rgbs, ins_num

    def selected_pixels(self, full_ins, ins_num, crop_mask):

        def weakly_img():
            """select ins label regard img as unit"""
            crop_mask_temp = crop_mask.reshape(-1)
            ins[crop_mask_temp == 0] = ins_num
            all_ins_indices = np.where(ins != ins_num)[0]
            select_amount = int(len(all_ins_indices) * self.weakly_value)
            select_indies = np.random.choice(len(all_ins_indices), size=[select_amount],
                                             replace=False)  # generate select index
            ins_hws = all_ins_indices[select_indies]
            return ins_hws

        # begin weakly
        N, H, W = full_ins.shape
        full_ins = np.reshape(full_ins, [N, -1])  # (N, H*W)
        all_ins_hws = []

        for i in range(N):
            ins = full_ins[i]
            unique_labels, label_amounts = np.unique(ins, return_counts=True)
            # need a parameter
            hws = weakly_img()
            all_ins_hws.append(hws)

        return all_ins_hws


def load_data(args):
    imgs, poses, i_split, intrinsic = img_processor(args.datadir,
                                                    args.testskip,
                                                    resize=args.resize).load_rgb()

    decompose_processor = ins_processor(args.datadir,
                                        testskip=args.testskip,
                                        resize=args.resize,
                                        weakly_value=args.weakly_value)
    gt_labels, ins_rgbs, ins_num = decompose_processor.load_semantic_instance()
    crop_size = [args.crop_width, args.crop_height]

    H, W = imgs[0].shape[:2]
    hwk = [int(H), int(W), intrinsic]
    crop_mask = crop_data(H, W, crop_size)
    ins_indices = ins_processor.selected_pixels(gt_labels, ins_num, crop_mask)

    return imgs, poses, hwk, i_split, gt_labels, ins_rgbs, ins_num, ins_indices, crop_mask
