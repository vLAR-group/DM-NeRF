import torch
import os
import imageio
import time
import json
import lpips
import cv2
import numpy as np
import torch.nn.functional as F

from skimage import metrics
from networks.evaluator import to8b
from networks.evaluator import ins_eval
from networks.helpers import get_rays_k, sample_pdf
from tools.visualizer import render_label2img, render_gt_label2img


def exchanger(ori_raw, tar_raws, ori_raw_pred, tar_raw_preds, move_labels):
    ori_pred_ins = ori_raw[..., 4:]
    ori_pred_ins = torch.sigmoid(ori_pred_ins)
    ori_pred_label = torch.argmax(ori_pred_ins, dim=-1)  # 0-32

    ori_accum_ins = ori_raw_pred[..., :-1]
    ori_accum_ins = torch.sigmoid(ori_accum_ins)
    ori_accum_label = torch.argmax(ori_accum_ins, dim=-1)  # 0-32
    ori_accum_label = ori_accum_label[:, None].repeat(1, ori_pred_label.shape[-1])

    for idx, move_label in enumerate(move_labels):
        tar_raw = tar_raws[idx]
        tar_raw_pre = tar_raw_preds[idx]
        ######################################################

        ori_is_move = ori_pred_label == move_label
        ori_acc_not_move = ori_accum_label != move_label
        ori_occludes = ori_acc_not_move * ori_is_move
        ori_pred_label[ori_occludes == True] = ori_accum_label[ori_occludes == True]

        ######################################################

        ori_not_move = ori_pred_label != move_label
        ori_accum_move = ori_accum_label == move_label
        fillings = ori_accum_move * ori_not_move
        # ori_pred_label[ccc == True] = move_label

        tar_pred_ins = tar_raw[..., 4:]
        tar_pred_ins = torch.sigmoid(tar_pred_ins)
        tar_pred_label = torch.argmax(tar_pred_ins, dim=-1)  # 0-32
        tar_pred_label_temp = tar_pred_label

        tar_accum_ins = tar_raw_pre[..., :-1]
        tar_accum_ins = torch.sigmoid(tar_accum_ins)
        tar_accum_label = torch.argmax(tar_accum_ins, dim=-1)  # 0-32
        tar_accum_label = tar_accum_label[:, None].repeat(1, tar_pred_label.shape[-1])

        ######################################################

        tar_is_move = tar_pred_label == move_label
        tar_accum_not_move = tar_accum_label != move_label
        tar_occludes = tar_accum_not_move * tar_is_move
        tar_pred_label[tar_occludes == True] = tar_accum_label[tar_occludes == True]

        ######################################################

        operation_mask = torch.zeros_like(ori_pred_label)
        ori_move_mask, tar_move_mask = torch.zeros_like(ori_pred_label), torch.zeros_like(tar_pred_label)
        ori_move_mask[ori_pred_label == move_label] = -2
        tar_move_mask[tar_pred_label == move_label] = 1

        reduced_mask = tar_move_mask - ori_move_mask

        operation_mask[reduced_mask == 0] = -1
        operation_mask[reduced_mask == 1] = 1
        operation_mask[reduced_mask == 2] = 0
        operation_mask[reduced_mask == 3] = 1
        '''-1 means not exchange, 0 means eliminate, 1 means exchange'''
        ######################################################
        ori_raw[fillings] = tar_raw[fillings]
        ######################################################

        ori_raw[operation_mask == 1] = tar_raw[operation_mask == 1]
        ori_raw[operation_mask == 0] = ori_raw[operation_mask == 0] * 0

    return ori_raw, tar_raws, ori_pred_label, tar_pred_label_temp


def manipulator_render(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    ins_labels = raw[..., 4:]
    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    ins_map = torch.sum(weights[..., None] * ins_labels, -2)  # [N_rays, 16]
    ins_map = torch.sigmoid(ins_map)
    depth_map = torch.sum(weights * z_vals, -1)

    return rgb_map, weights, depth_map, ins_map


def manipulator_nerf(rays, position_embedder, view_embedder, model, N_samples=None, near=None, far=None, z_vals=None):
    rays_o, rays_d = rays
    viewdirs = rays_d
    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # flatten

    N_rays, c = rays_d.shape

    if z_vals is None:
        near_, far_ = near * torch.ones(size=(N_rays, 1)), far * torch.ones(size=(N_rays, 1))  # N_rays,
        t_vals = torch.linspace(0., 1., steps=N_samples)
        z_vals = near_ * (1. - t_vals) + far_ * t_vals
        z_vals = z_vals.expand([N_rays, N_samples])

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    embedded_pos = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded_pos, embedded_dirs], -1)

    raw = model(embedded)
    raw = torch.reshape(raw, list(pts.shape[:-1]) + [raw.shape[-1]])  # B,N_sample,rgb+density+instance

    return raw, z_vals


def manipulator(position_embedder, view_embedder, model_coarse, model_fine, ori_rays, f_tar_rays, args):
    # extract parameter
    N_samples, N_importance, near, far = args.N_samples, args.N_importance, args.near, args.far
    ori_raw, ori_z_vals = manipulator_nerf(ori_rays, position_embedder, view_embedder,
                                           model_coarse, N_samples, near, far)

    # ori
    _, ori_weights, _, _ = manipulator_render(ori_raw, ori_z_vals, ori_rays[1])

    # sample 128
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points
    ori_z_vals_full, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples], dim=-1), dim=-1)
    ori_raw_full, _ = manipulator_nerf(ori_rays, position_embedder, view_embedder, model_fine,
                                       N_samples, near, far, z_vals=ori_z_vals_full)
    _, _, _, ori_ins_accum = manipulator_render(ori_raw_full, ori_z_vals_full, ori_rays[1])

    tar_raws, tar_rgbs, f_tar_weights, tar_instances, f_tar_z_vals, f_tar_z_samples, tar_ins_accums, = [], [], [], [], [], [], []
    for idx, tar_rays in enumerate(f_tar_rays):
        # sample 64
        tar_raw, tar_z_vals = manipulator_nerf(tar_rays, position_embedder, view_embedder,
                                               model_coarse, N_samples, near, far)
        tar_raws.append(tar_raw)
        f_tar_z_vals.append(tar_z_vals)

        # tar
        tar_rgb, tar_weights, tar_depth, tar_ins = manipulator_render(tar_raw, tar_z_vals, tar_rays[1])
        tar_rgbs.append(tar_rgb)
        f_tar_weights.append(tar_weights)
        tar_instances.append(tar_ins)

        # sample 128
        tar_z_vals_mid = .5 * (tar_z_vals[..., 1:] + tar_z_vals[..., :-1])
        tar_z_samples = sample_pdf(tar_z_vals_mid, tar_weights[..., 1:-1], N_importance)
        tar_z_vals_full, _ = torch.sort(torch.cat([tar_z_vals, tar_z_samples], dim=-1), dim=-1)
        tar_raw_full, _ = manipulator_nerf(tar_rays, position_embedder, view_embedder,
                                           model_fine, z_vals=tar_z_vals_full)
        _, _, _, tar_ins_accum = manipulator_render(tar_raw_full, tar_z_vals_full, tar_rays[1])
        f_tar_z_samples.append(tar_z_samples)
        tar_ins_accums.append(tar_ins_accum)

    # exchange
    ori_raw, tar_raw, _, tar_pred_label = exchanger(ori_raw, tar_raws, ori_ins_accum, tar_ins_accums, args.target_labels)

    """step2"""
    # calculate weights
    ori_rgb, ori_weights, ori_depth, ori_ins = manipulator_render(ori_raw, ori_z_vals, ori_rays[1])

    # resample 128 points respectively
    ori_z_vals_mid = .5 * (ori_z_vals[..., 1:] + ori_z_vals[..., :-1])
    ori_z_samples = sample_pdf(ori_z_vals_mid, ori_weights[..., 1:-1], N_importance)  # interpolate 128 points

    # cat all the points 64+128+(128)
    f_tar_z_samples = torch.cat(f_tar_z_samples, dim=-1)
    ori_z_vals, _ = torch.sort(torch.cat([ori_z_vals, ori_z_samples, f_tar_z_samples], dim=-1), dim=-1)
    for idx, tar_rays in enumerate(f_tar_rays):
        tar_z_vals = f_tar_z_vals[idx]
        ori_raw, ori_z_vals = manipulator_nerf(ori_rays, position_embedder, view_embedder,
                                               model_fine, z_vals=ori_z_vals)
        tar_z_vals, _ = torch.sort(torch.cat([tar_z_vals, ori_z_samples, f_tar_z_samples], dim=-1), dim=-1)
        tar_raw, tar_z_vals = manipulator_nerf(tar_rays, position_embedder, view_embedder,
                                               model_fine, z_vals=tar_z_vals)
        tar_raws[idx] = tar_raw

    ori_raw, tar_raws, _, _ = exchanger(ori_raw, tar_raws, ori_ins_accum, tar_ins_accums, args.target_labels)
    # final render a rgb and ins map
    final_rgb, final_weights, final_depth, final_ins = manipulator_render(ori_raw, ori_z_vals, ori_rays[1])

    return final_rgb, final_ins, tar_rgb, tar_ins_accum


def manipulator_eval(position_embedder, view_embedder, model_coarse, model_fine, ori_poses,
                     hwk, trans_dicts, save_dir, ins_rgbs, args, gt_rgbs=None, gt_labels=None):
    """move_object must between 1 to args.class_number"""
    _, _, dataset_name, scene_name = args.datadir.split('/')
    H, W, K = hwk
    if gt_rgbs is not None:
        gt_rgbs_cpu = gt_rgbs.cpu().numpy()
        gt_rgbs_gpu = gt_rgbs.to(args.device)
        lpips_vgg = lpips.LPIPS(net='vgg').to(args.device)
        gt_ins = torch.zeros(size=(H, W, args.ins_num))

    ori_rgbs, ins_imgs, psnrs, ssims, lpipses, aps = [], [], [], [], [], []

    gt_color_dict_path = './data/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    color_dict = gt_color_dict[dataset_name][scene_name]
    full_map = {}

    trans_dict = trans_dicts['transformations'][0]
    trans = torch.Tensor(trans_dict['transformation'])
    save_dir = os.path.join(save_dir, trans_dict["mode"])
    os.makedirs(save_dir, exist_ok=True)

    args.target_labels = [args.target_label]
    # original
    for i, ori_pose in enumerate(ori_poses):
        time_0 = time.time()
        ori_rays_o, ori_rays_d = get_rays_k(H, W, K, torch.Tensor(ori_pose))
        ori_rays_o = torch.reshape(ori_rays_o, [-1, 3]).float()
        ori_rays_d = torch.reshape(ori_rays_d, [-1, 3]).float()

        tar_pose = trans @ ori_pose
        tar_rays_o, tar_rays_d = get_rays_k(H, W, K, torch.Tensor(tar_pose))
        tar_rays_o = torch.reshape(tar_rays_o, [-1, 3]).float()
        tar_rays_d = torch.reshape(tar_rays_d, [-1, 3]).float()
        full_rgb, full_ins, full_tar_rgb = None, None, None

        """doing editor"""
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
            # original view rays render
            ori_rays_io = ori_rays_o[step:step + N_test]  # (chuck, 3)
            ori_rays_id = ori_rays_d[step:step + N_test]  # (chuck, 3)
            ori_batch_rays = torch.stack([ori_rays_io, ori_rays_id], dim=0)
            # target view rays render
            tar_rays_io = tar_rays_o[step:step + N_test]  # (chuck, 3)
            tar_rays_id = tar_rays_d[step:step + N_test]  # (chuck, 3)
            tar_batch_rays = torch.stack([tar_rays_io, tar_rays_id], dim=0)
            tar_batch_rays = tar_batch_rays[None, ...]
            # edit render
            ori_rgb, ins, tar_rgb, tar_ins = manipulator(position_embedder, view_embedder, model_coarse, model_fine,
                                                         ori_batch_rays, tar_batch_rays, args)
            # all_info = ins_nerf(tar_batch_rays, position_embedder, view_embedder, model_fine, model_coarse, args)
            if full_rgb is None and full_ins is None:
                full_rgb, full_ins, full_tar_rgb, full_tar_ins = ori_rgb, ins, tar_rgb, tar_ins
            else:
                full_rgb = torch.cat((full_rgb, ori_rgb), dim=0)
                full_ins = torch.cat((full_ins, ins), dim=0)
                full_tar_rgb = torch.cat((full_tar_rgb, tar_rgb), dim=0)
                full_tar_ins = torch.cat((full_tar_ins, tar_ins), dim=0)

        """editor finish"""
        ori_rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])
        ins = full_ins.reshape([H, W, full_ins.shape[-1]])

        """calculating step for our dataset"""
        if gt_rgbs is not None:
            print('=' * 50, i, '=' * 50)
            psnr = metrics.peak_signal_noise_ratio(ori_rgb.cpu().numpy(), gt_rgbs_cpu[i], data_range=1)
            ssim = metrics.structural_similarity(ori_rgb.cpu().numpy(), gt_rgbs_cpu[i], multichannel=True, data_range=1)
            lpips_i = lpips_vgg(ori_rgb.permute(2, 0, 1).unsqueeze(0), gt_rgbs_gpu[i].permute(2, 0, 1).unsqueeze(0))
            psnrs.append(psnr)
            ssims.append(ssim)
            lpipses.append(lpips_i.item())
            print(f"PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips_i.item()}")

            # calculate ap
            gt_label = gt_labels[i]
            valid_gt_labels = torch.unique(gt_label)
            valid_gt_num = len(valid_gt_labels)
            gt_ins[..., :valid_gt_num] = F.one_hot(gt_label.long())[..., valid_gt_labels.long()]
            gt_label_np = valid_gt_labels.cpu().numpy()
            if valid_gt_num > 0:
                # mask = (gt_label < args.ins_num).type(torch.float32)
                pred_label, ap, pred_matched_order = ins_eval(ins[..., :-1].cpu(), gt_ins, valid_gt_num, args.ins_num)
            else:
                pred_label = -1 * torch.ones([H, W])
                ap = torch.tensor([1.0])

            ins_map = {}
            for idx, pred_label_replica in enumerate(pred_matched_order):
                if pred_label_replica != -1:
                    ins_map[str(pred_label_replica)] = int(gt_label_np[idx])

            full_map[i] = ins_map

            aps.append(ap)
            print(f"APs: {ap}")

        # get predicted rgb
        ori_rgb_s = ori_rgb.cpu().numpy()
        ori_rgb_s = ori_rgb_s.reshape([H, W, 3])
        ori_rgb_s = to8b(ori_rgb_s)

        # get target rgb
        tar_rgb = full_tar_rgb.reshape([H, W, full_tar_rgb.shape[-1]])
        tar_rgb = tar_rgb.cpu().numpy()
        tar_rgb = tar_rgb.reshape([H, W, 3])
        tar_rgb = to8b(tar_rgb)

        # get predicted ins color
        label = torch.argmax(ins, dim=-1)
        label = label.reshape([H, W])
        ins_img = render_label2img(label, ins_rgbs, color_dict, ins_map)

        # get target ins color
        gt_img = to8b(gt_rgbs[i].cpu().numpy())
        gt_ins_img = render_gt_label2img(gt_label, ins_rgbs, color_dict)

        # save images
        ori_rgbs.append(ori_rgb_s)
        ins_imgs.append(ins_img)
        img_file = os.path.join(save_dir, f'{i}_rgb.png')
        imageio.imwrite(img_file, ori_rgb_s)
        ins_file = os.path.join(save_dir, f'{i}_ins.png')
        cv2.imwrite(ins_file, ins_img)
        gt_img_file = os.path.join(save_dir, f'{i}_rgb_gt.png')
        imageio.imwrite(gt_img_file, gt_img)
        gt_ins_file = os.path.join(save_dir, f'{i}_ins_gt.png')
        cv2.imwrite(gt_ins_file, gt_ins_img)
        time_1 = time.time()
        # print(f'IMAGE[{i}] TIME: {np.round(time_1 - time_0, 6)} second')

    """save all results"""
    if gt_rgbs is not None:

        map_result_file = os.path.join(save_dir, 'matching_log.json')
        with open(map_result_file, 'w') as f:
            json.dump(full_map, f)

        aps = np.array(aps)
        output = np.stack([psnrs, ssims, lpipses, aps[:, 0], aps[:, 1], aps[:, 2], aps[:, 3], aps[:, 4], aps[:, 5]])
        output = output.transpose([1, 0])
        out_ap = np.mean(aps, axis=0)
        mean_output = np.array([np.nanmean(psnrs), np.nanmean(ssims), np.nanmean(lpipses), out_ap[0],
                                out_ap[1], out_ap[2], out_ap[3], out_ap[4], out_ap[5]])
        mean_output = mean_output.reshape([1, 9])
        output = np.concatenate([output, mean_output], 0)
        test_result_file = os.path.join(save_dir, 'test_results.txt')
        np.savetxt(fname=test_result_file, X=output, fmt='%.6f', delimiter=' ')
        print('=' * 49, 'Avg', '=' * 49)
        print('PSNR: {:.4f}, SSIM: {:.4f},  LPIPS: {:.4f} '.format(np.mean(psnrs), np.mean(ssims), np.mean(lpipses)))
        print('AP50: {:.4f}, AP75: {:.4f}, AP80: {:.4f}, AP85: {:.4f}, AP90: {:.4f}, AP95: {:.4f}'
              .format(out_ap[0], out_ap[1], out_ap[2], out_ap[3], out_ap[4], out_ap[5]))
    return


def manipulator_demo(position_embedder, view_embedder, model_coarse, model_fine, ori_poses,
                     hwk, objs_trans, save_dir, ins_rgbs, objs, view_poses, ins_map, args):

    _, _, dataset_name, scene_name = args.datadir.split('/')
    H, W, K = hwk

    gt_color_dict_path = './data/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    color_dict = gt_color_dict[dataset_name][scene_name]

    save_dir = os.path.join(save_dir, args.mani_type)
    os.makedirs(save_dir, exist_ok=True)

    # original
    deform_v = np.concatenate((np.linspace(0, 0.18, 2), np.linspace(0.18, 0, 2),
                               np.linspace(0, -0.18, 2), np.linspace(-0.18, 0, 2)))

    for i, ori_pose in enumerate(view_poses):
        # operate objects at same time
        time_0 = time.time()

        ori_rays_o, ori_rays_d = get_rays_k(H, W, K, torch.Tensor(ori_pose))
        ori_rays_o = torch.reshape(ori_rays_o, [-1, 3]).float()
        ori_rays_d = torch.reshape(ori_rays_d, [-1, 3]).float()

        tar_rays_os, tar_rays_ds, target_labels = [], [], []
        for obj in objs:
            obj_name = obj['obj_name']
            target_labels.append(obj['tar_id'])
            mani_mode = obj['mani_mode']
            if mani_mode == 'deform':
                v_1 = np.linspace(1, H, H)
                deform_func = obj['deform_func']
                if deform_func == 'sin':
                    """deform sin"""
                    v_1 = ((8 * np.pi) / 400) * v_1
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = np.sin(v_1) * deform_v[i]
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif deform_func == 'ex':
                    """deform e^x"""
                    v_1 = np.exp(-1 * v_1 / 50)
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif deform_func == 'linear':
                    """"deform linear"""
                    v_1 = (v_1 - 200) / 215
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif deform_func == 'abs_linear':
                    """"deform linear"""
                    v_1 = np.abs(v_1 - 200) / 200
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                elif deform_func == 'ln':
                    """deform ln"""
                    v_1 = v_1 / 200
                    v_1 = np.repeat(v_1[:, np.newaxis], W, axis=-1)
                    v_1 = np.log(v_1)
                    v_1 = torch.from_numpy(v_1.reshape(-1)).to(args.device)
                tar_rays_o = ori_rays_o.clone()
                tar_rays_o[:, 0] = tar_rays_o[:, 0] + v_1
                tar_rays_d = ori_rays_d.clone()
            else:
                trans = torch.Tensor(objs_trans[obj_name][i]['transformation'])
                tar_pose = trans @ ori_pose
                tar_rays_o, tar_rays_d = get_rays_k(H, W, K, torch.Tensor(tar_pose))
            tar_rays_os.append(tar_rays_o)
            tar_rays_ds.append(tar_rays_d)

        args.target_labels = target_labels
        tar_rays_os = torch.stack(tar_rays_os)
        tar_rays_ds = torch.stack(tar_rays_ds)
        tar_rays_os = torch.reshape(tar_rays_os, [len(objs), -1, 3]).float()
        tar_rays_ds = torch.reshape(tar_rays_ds, [len(objs), -1, 3]).float()
        full_rgb, full_ins, full_tar_rgb = None, None, None

        """doing editor"""
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
            # original view rays render
            ori_rays_io = ori_rays_o[step:step + N_test]  # (chuck, 3)
            ori_rays_id = ori_rays_d[step:step + N_test]  # (chuck, 3)
            ori_batch_rays = torch.stack([ori_rays_io, ori_rays_id], dim=0)
            # target view rays render
            tar_rays_ios = tar_rays_os[:, step:step + N_test]  # (chuck, 3)
            tar_rays_ids = tar_rays_ds[:, step:step + N_test]  # (chuck, 3)
            tar_batch_rays = torch.stack([tar_rays_ios, tar_rays_ids], dim=1)
            # edit render
            ori_rgb, ins, tar_rgb, tar_ins = manipulator(position_embedder, view_embedder, model_coarse, model_fine,
                                                         ori_batch_rays, tar_batch_rays, args)
            if full_rgb is None and full_ins is None:
                full_rgb, full_ins, full_tar_rgb, full_tar_ins = ori_rgb, ins, tar_rgb, tar_ins
            else:
                full_rgb = torch.cat((full_rgb, ori_rgb), dim=0)
                full_ins = torch.cat((full_ins, ins), dim=0)
                full_tar_rgb = torch.cat((full_tar_rgb, tar_rgb), dim=0)
                full_tar_ins = torch.cat((full_tar_ins, tar_ins), dim=0)
        ori_rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])
        ins = full_ins.reshape([H, W, full_ins.shape[-1]])

        """editor finish"""
        # get predicted rgb
        ori_rgb_s = ori_rgb.cpu().numpy()
        ori_rgb_s = ori_rgb_s.reshape([H, W, 3])
        ori_rgb_s = to8b(ori_rgb_s)

        # get predicted ins color
        label = torch.argmax(ins, dim=-1)
        label = label.reshape([H, W])
        ins_img = render_label2img(label, ins_rgbs, color_dict, ins_map)

        # save images
        img_file = os.path.join(save_dir, f'{i}_rgb.png')
        imageio.imwrite(img_file, ori_rgb_s)
        ins_file = os.path.join(save_dir, f'{i}_ins.png')
        cv2.imwrite(ins_file, ins_img)

        gt_ins_file = os.path.join(save_dir, f'{i}_ins_pred_mask.png')
        imageio.imwrite(gt_ins_file, np.array(label.cpu().numpy(), dtype=np.uint8))
        time_1 = time.time()
        print(f"Image{i}: {time_1 - time_0}")
    return
