import os
import torch
import trimesh

from networks import manipulator
from tools import pose_generator
from datasets import loader_dmsr_mani, loader_dmsr
from networks.tester import render_test
from config import create_nerf, initial
from tools.mesh_generator import mesh_main


def test():
    model_coarse.eval()
    model_fine.eval()
    args.is_train = False
    with torch.no_grad():
        if args.render:
            print('Rendering......')
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'render_{}_{:06d}'
                                       .format('test' if args.render_test else 'path', iteration))
            os.makedirs(testsavedir, exist_ok=True)
            mathed_file = os.path.join(testsavedir, 'matching_log.txt')
            i_train, i_test = i_split
            in_images = torch.Tensor(images[i_test])
            in_instances = torch.Tensor(instances[i_test]).type(torch.int16)
            in_poses = torch.Tensor(poses[i_test])
            render_test(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk, args,
                        gt_imgs=in_images, gt_labels=in_instances, ins_rgbs=ins_colors, savedir=testsavedir,
                        matched_file=mathed_file)
            print('Rendering Done', testsavedir)

        elif args.mani_eval:
            print('Manipulating......')
            """this operations list can re-design"""
            in_images = torch.Tensor(images)
            in_instances = torch.Tensor(instances).type(torch.int8)
            in_poses = torch.Tensor(poses)
            pose_generator.generate_poses_eval(args)
            trans_dicts = loader_dmsr_mani.load_mani_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_eval_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.manipulator_eval(position_embedder, view_embedder, model_coarse, model_fine, in_poses, hwk,
                                         trans_dicts=trans_dicts, save_dir=testsavedir, ins_rgbs=ins_colors, args=args,
                                         gt_rgbs=in_images, gt_labels=in_instances)
            print('Manipulating Done', testsavedir)

        elif args.mani_demo:
            print('Manipulating......')
            """this operations list can re-design"""
            print('Loaded blender', hwk, args.datadir)
            int_view_poses = torch.Tensor(view_poses)
            pose_generator.generate_poses_demo(objs, args)
            obj_trans = loader_dmsr.load_mani_poses(args)
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time,
                                       'mani_demo_{:06d}'.format(iteration))
            os.makedirs(testsavedir, exist_ok=True)
            manipulator.manipulator_demo(position_embedder, view_embedder, model_coarse, model_fine, poses, hwk,
                                         save_dir=testsavedir, ins_rgbs=ins_colors, args=args, objs=objs,
                                         objs_trans=obj_trans, view_poses=int_view_poses, ins_map=ins_map)
            print('Manipulating Done', testsavedir)

        elif args.mesh:
            print("Meshing......")
            mesh_file = os.path.join(args.datadir, "mesh.ply")
            assert os.path.exists(mesh_file)
            trimesh_scene = trimesh.load(mesh_file, process=False)
            meshsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'mesh_{:06d}'.format(iteration))
            os.makedirs(meshsavedir, exist_ok=True)
            mesh_main(position_embedder, view_embedder, model_coarse, model_fine,
                      args, trimesh_scene, ins_colors, meshsavedir, ins_map)
            print('Meshing Done', meshsavedir)
    return


if __name__ == '__main__':

    args = initial()

    # load data
    if args.mani_eval:
        images, poses, hwk, instances, ins_colors, args.ins_num = loader_dmsr_mani.load_data(args)
    else:
        images, poses, hwk, i_split, instances, ins_colors, args.ins_num, objs, view_poses, ins_map = loader_dmsr.load_data(args)
    print('Loaded blender', images.shape, hwk, args.datadir)

    args.perturb = False
    H, W, K = hwk

    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    iteration = 0
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(args.basedir, args.expname, args.log_time, "200000.tar")]
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

    test()
