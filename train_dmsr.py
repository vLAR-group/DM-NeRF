import os
import torch
import numpy as np

from networks.render import dm_nerf
from config import initial, create_nerf
from networks.tester import render_test
from datasets.loader_dmsr import load_data
from networks.penalizer import ins_penalizer
from networks.helpers import get_select_full, z_val_sample
from networks.evaluator import ins_criterion, img2mse, mse2psnr

np.random.seed(0)
torch.cuda.manual_seed(3)


def train():
    model_fine.train()
    model_coarse.train()
    N_iters = 500000 + 1

    z_val_coarse = z_val_sample(args.N_train, args.near, args.far, args.N_samples)
    args.N_ins = None
    for i in range(0, N_iters):
        img_i = np.random.choice(i_train)
        gt_rgb = images[img_i].to(args.device)
        pose = poses[img_i, :3, :4].to(args.device)
        gt_label = gt_labels[img_i].to(args.device)

        target_c, target_i, batch_rays = get_select_full(gt_rgb, pose, K, gt_label, args.N_train)

        all_info = dm_nerf(batch_rays, position_embedder, view_embedder, model_coarse, model_fine, z_val_coarse, args)

        # coarse losses
        rgb_loss_coarse = img2mse(all_info['rgb_coarse'], target_c)
        psnr_coarse = mse2psnr(rgb_loss_coarse)

        ins_loss_coarse, valid_ce_coarse, invalid_ce_coarse, valid_siou_coarse = \
            ins_criterion(all_info['ins_coarse'], target_i, args.ins_num)

        # fine losses
        rgb_loss_fine = img2mse(all_info['rgb_fine'], target_c)
        psnr_fine = mse2psnr(rgb_loss_fine)
        ins_loss_fine, valid_ce_fine, invalid_ce_fine, valid_siou_fine = \
            ins_criterion(all_info['ins_fine'], target_i, args.ins_num)

        # without penalize loss
        ins_loss = ins_loss_fine + ins_loss_coarse
        rgb_loss = rgb_loss_fine + rgb_loss_coarse
        total_loss = ins_loss + rgb_loss

        # use penalize
        if args.penalize:
            emptiness_coarse = ins_penalizer(all_info['raw_coarse'], all_info['z_vals_coarse'],
                                            all_info['depth_coarse'], batch_rays[1], args)
            emptiness_fine = ins_penalizer(all_info['raw_fine'], all_info['z_vals_fine'],
                                          all_info['depth_fine'], batch_rays[1], args)

            emptiness_loss = emptiness_fine + emptiness_coarse
            total_loss = total_loss + emptiness_loss
        # optimizing
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # losses decay
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** ((i) / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ###################################

        if i % args.i_print == 0:
            print(f"[TRAIN] Iter: {i} PSNR: {psnr_fine.item()} Total_Loss: {total_loss.item()} RGB_Loss: {rgb_loss.item()} Ins_Loss: {ins_loss.item()}")

        if i % args.i_save == 0:
            path = os.path.join(args.basedir, args.expname, args.log_time, '{:06d}.tar'.format(i))
            save_model = {
                'iteration': i,
                'network_coarse_state_dict': model_coarse.state_dict(),
                'network_fine_state_dict': model_fine.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(save_model, path)

        if i % args.i_test == 0:
            model_coarse.eval()
            model_fine.eval()
            args.is_train = False
            selected_indices = np.random.choice(len(i_test), size=[10], replace=False)
            selected_i_test = i_test[selected_indices]
            testsavedir = os.path.join(args.basedir, args.expname, args.log_time, 'testset_{:06d}'.format(i))
            matched_file = os.path.join(testsavedir, 'matching_log.txt')
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_poses = torch.Tensor(poses[selected_i_test].to(args.device))
                test_imgs = images[selected_i_test]
                test_gt_labels = gt_labels[selected_i_test].to(args.device)
                render_test(position_embedder, view_embedder, model_coarse, model_fine, test_poses, hwk, args,
                            gt_imgs=test_imgs, gt_labels=test_gt_labels, ins_rgbs=ins_rgbs, savedir=testsavedir,
                            matched_file=matched_file)
            print('Training model saved!')
            args.is_train = True
            model_coarse.train()
            model_fine.train()


if __name__ == '__main__':

    args = initial()
    # load data
    images, poses, hwk, i_split, gt_labels, ins_rgbs, args.ins_num, objs, view_poses, ins_map = load_data(args)
    print('Load data from', args.datadir)

    i_train, i_test = i_split
    H, W, K = hwk

    # Create nerf model
    position_embedder, view_embedder, model_coarse, model_fine, args = create_nerf(args)

    # Create optimizer
    grad_vars = list(model_coarse.parameters()) + list(model_fine.parameters())
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # move data to gpu
    images = torch.Tensor(images).cpu()
    gt_labels = torch.Tensor(gt_labels).type(torch.int16).cpu()
    poses = torch.Tensor(poses).cpu()

    train()
