import os
import time
import torch
import configargparse

from networks.dm_nerf import get_embedder, DM_NeRF


def config_parser():
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True, type=str, default='train_configs/replica/office_0.txt',
                        help='train_configs file path')
    parser.add_argument("--expname", type=str, default='office_0',
                        help='experiment name')
    parser.add_argument("--log_time", default=None,
                        help="save as time level")
    parser.add_argument("--basedir", type=str, default='./logs',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/replica/office_0',
                        help='input data directory')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    # 32*32*4
    parser.add_argument("--N_train", type=int, default=4096,
                        help='batch size (number of random rays per gradient step)')

    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 250
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--N_test", type=int, default=1024 * 2,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--is_train", type=bool, default=True,
                        help='train or test')
    # semantic instance network options

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # 0
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--render", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--test_model", type=str, default='000000.tar',
                        help='where to store ckpts and logs')

    # datasets options
    parser.add_argument("--testskip", type=int, default=10,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    parser.add_argument("--resize", action='store_true',
                        help='will resize image and instance map shape of ScanNet dataset')
    parser.add_argument("--near", type=float,
                        help='set the nearest depth')
    parser.add_argument("--far", type=float,
                        help='set the farest depth')
    parser.add_argument("--crop_width", type=int,
                        help='set the width of crop')
    parser.add_argument("--crop_height", type=int,
                        help='set the height of crop')

    # logging/saving options
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_save", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=50000,
                        help='frequency of testset saving')

    # semantic instance options
    parser.add_argument("--penalize", action='store_true',
                        help="aim to penalize unlabeled rays to air")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="move the center of Gaussian Distribution to the depth - tolerance")
    parser.add_argument("--deta_w", type=float, default=None,
                        help="contorl the Amplitude of Gaussian Distribution")

    # visualizer hyper-parameter
    parser.add_argument("--target_label", type=int, default=None,
                        help='sign the instance you want to move')

    parser.add_argument("--center_index", type=int, default=None,
                        help='sign the instance center wich you want to move')

    parser.add_argument("--ori_pose", type=int, default=None,
                        help='sign the instance center')
    # manipulation parameter
    parser.add_argument("--mani_demo", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--mani_eval", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--mani_mode", type=str, default='rotation',
                        help='select operation mode includes translation, rotation, scale, multi')
    parser.add_argument("--mani_type", type=str, default='rigid',
                        help='select operation type includes rigid, deform')
    parser.add_argument("--views", type=int, default=720,
                        help="the amount of generated view")
    parser.add_argument("--translation", type=bool, default=False,
                        help='do the shift operation')
    parser.add_argument("--rotation", type=bool, default=False,
                        help='do the rotation operation')
    parser.add_argument("--scale", type=bool, default=False,
                        help='do the scale down or scale up operation')

    # 3D color mesh extraction
    parser.add_argument("--mesh", action='store_true',
                        help='generate a 3d point cloud file')
    return parser


def create_nerf(args):
    """
        Instantiate NeRF's MLP model.
    """
    position_embedder, input_ch_pos = get_embedder(args.multires, args.i_embed)
    view_embedder, input_ch_view = get_embedder(args.multires_views, args.i_embed)
    model_coarse = \
        DM_NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4], args.ins_num).to(args.device)

    model_fine = \
        DM_NeRF(args.netdepth, args.netwidth, input_ch_pos, input_ch_view, [4], args.ins_num).to(args.device)
    print(model_fine)
    return position_embedder, view_embedder, model_coarse, model_fine, args


def initial():
    parser = config_parser()
    args = parser.parse_args()

    # get log time
    if args.log_time is None:
        args.log_time = time.strftime("%Y%m%d%H%M", time.localtime())

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")

    log_dir = os.path.join(args.basedir, args.expname, args.log_time)
    print('Logs in', log_dir)
    os.makedirs(log_dir, exist_ok=True)
    f = os.path.join(log_dir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(log_dir, 'configs.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    return args
