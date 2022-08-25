import torch

from datasets.loader_scannet import *
from networks.tester import render_test
from config import create_nerf, initial
from networks.manipulator import manipulator_demo


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():
        if args.render:
            print('Rendering......')
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', iteration))
            os.makedirs(testsavedir, exist_ok=True)
            mathed_file = os.path.join(testsavedir, 'matching_log.txt')
            render_test(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk, args,
                        gt_imgs=images, gt_labels=instances, ins_rgbs=ins_colors, savedir=testsavedir,
                        matched_file=mathed_file, crop_mask=crop_mask)
            print('Rendering Done', testsavedir)
        elif args.mani_demo:
            print('Manipulating......')
            """this operations list can re-design"""
            trans_dicts = load_mani_poses(args)
            ori_pose = poses[args.ori_pose]
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_demo_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator_demo(position_embedder, view_embedder, model_fine, ori_pose, hwk,
                             trans_dicts=trans_dicts, save_dir=testsavedir, ins_rgbs=ins_colors, args=args)
            print('Manipulating Done', testsavedir)
    return


if __name__ == '__main__':

    args = initial()
    # load data
    images, poses, hwk, i_split, instances, ins_colors, args.ins_num, ins_indices, crop_mask = load_data(args)
    print('Loaded blender', images.shape, hwk, args.datadir)

    H, W, K = hwk
    i_train, i_test = i_split

    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    iteration = 0
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, args.log_time, '250000.tar')]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        if torch.cuda.is_available():
            ckpt = torch.load(ckpt_path)
        else:
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        iteration = ckpt['iteration']
        # Load model
        model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
        model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    images = torch.Tensor(images[i_test])
    instances = torch.Tensor(instances[i_test]).type(torch.int16)
    poses = torch.Tensor(poses[i_test])
    args.perturb = False

    test()
