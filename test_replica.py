from datasets.loader_replica import *
from config import create_nerf, initial
from networks.tester import render_test
from networks.manipulator import manipulator_demo


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():
        print('Rendering......')
        testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                   'render_test_{:06d}'.format(iteration))
        os.makedirs(testsavedir, exist_ok=True)
        mathed_file = os.path.join(testsavedir, 'matching_log.txt')
        render_test(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk, args,
                    gt_imgs=images, gt_labels=instances, ins_rgbs=ins_colors, savedir=testsavedir,
                    matched_file=mathed_file)
        print('Rendering Done', testsavedir)
    return


if __name__ == '__main__':

    args = initial()
    # load data
    images, poses, hwk, i_split, instances, ins_colors, args.ins_num = load_data(args)
    print('Load data from', args.datadir)

    H, W, K = hwk
    i_train, i_test = i_split
    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    ckpt_path = os.path.join(args.basedir, args.expname, args.log_time, args.test_model)
    print('Reloading from', ckpt_path)
    ckpt = torch.load(ckpt_path)
    iteration = ckpt['iteration']
    # Load model
    model_coarse.load_state_dict(ckpt['network_coarse_state_dict'])
    model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    images = torch.Tensor(images[i_test])
    instances = torch.Tensor(instances[i_test]).type(torch.int16)
    poses = torch.Tensor(poses[i_test])
    args.perturb = False

    test()
