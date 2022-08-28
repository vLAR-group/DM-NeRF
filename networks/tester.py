import os
import cv2
import time
import lpips
import torch
import imageio
import numpy as np
import torch.nn.functional as F

from skimage import metrics
from networks.render import dm_nerf
from networks.evaluator import to8b, ins_eval
from networks.helpers import get_rays_k, z_val_sample
from tools.visualizer import render_label2img, render_gt_label2img


def render_test(position_embedder, view_embedder, model_coarse, model_fine, render_poses, hwk, args, gt_imgs=None,
                gt_labels=None, ins_rgbs=None, savedir=None, matched_file=None, crop_mask=None):
    data_info = args.datadir.split('/')
    dataset_name = data_info[2]
    scene_name = data_info[-1]
    H, W, K = hwk
    cropped_imgs, cropped_labels, psnrs, ssims, lpipses, aps = [], [], [], [], [], []

    if crop_mask is not None:
        crop_mask = crop_mask.reshape(-1)
        for index in range(len(gt_imgs)):
            gt_img = gt_imgs[index].reshape(-1, 3)
            gt_img = gt_img[crop_mask == 1]
            gt_img = gt_img.reshape([args.crop_height, args.crop_width, gt_img.shape[-1]])
            cropped_imgs.append(gt_img)

            gt_label = gt_labels[index].reshape(-1)
            gt_label = gt_label[crop_mask == 1]
            gt_label = gt_label.reshape([args.crop_height, args.crop_width])
            cropped_labels.append(gt_label)
            # vis_crop(gt_img)

        gt_imgs = torch.stack(cropped_imgs, 0)
        gt_labels = torch.stack(cropped_labels, 0)
    gt_imgs_cpu = gt_imgs.cpu().numpy()
    gt_imgs_gpu = gt_imgs.to(args.device)
    lpips_vgg = lpips.LPIPS(net="vgg").to(args.device)
    gt_ins = torch.zeros(size=(gt_labels.shape[1], gt_labels.shape[2], args.ins_num)).cpu()
    if matched_file is not None:
        if os.path.exists(matched_file):
            os.remove(matched_file)

    import json
    gt_color_dict_path = './data/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    color_dict = gt_color_dict[dataset_name][scene_name]
    full_map = {}

    for i, c2w in enumerate(render_poses):
        print('=' * 50, i, '=' * 50)
        t = time.time()
        z_val_coarse = z_val_sample(args.N_test, args.near, args.far, args.N_samples)
        rays_o, rays_d = get_rays_k(H, W, K, torch.Tensor(c2w))
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        full_rgb, full_ins, full_disp = None, None, None
        for step in range(0, H * W, args.N_test):
            N_test = args.N_test
            if step + N_test > H * W:
                N_test = H * W - step
                z_val_coarse = z_val_sample(N_test, args.near, args.far, args.N_samples)
            rays_io = rays_o[step:step + N_test]  # (chuck, 3)
            rays_id = rays_d[step:step + N_test]  # (chuck, 3)
            batch_rays = torch.stack([rays_io, rays_id], dim=0)
            all_info = dm_nerf(batch_rays, position_embedder, view_embedder,
                               model_coarse, model_fine, z_val_coarse, args)
            if full_rgb is None and full_ins is None:
                full_rgb, full_ins = all_info['rgb_fine'], all_info['ins_fine']
            else:
                full_rgb = torch.cat((full_rgb, all_info['rgb_fine']), dim=0)
                full_ins = torch.cat((full_ins, all_info['ins_fine']), dim=0)
        if crop_mask is not None:
            rgb = full_rgb[crop_mask == 1]
            ins = full_ins[crop_mask == 1]
            rgb = rgb.reshape([args.crop_height, args.crop_width, rgb.shape[-1]])
            ins = ins.reshape([args.crop_height, args.crop_width, ins.shape[-1]])
        else:
            rgb = full_rgb.reshape([H, W, full_rgb.shape[-1]])
            ins = full_ins.reshape([H, W, full_ins.shape[-1]])

        if gt_imgs is not None:
            # rgb image evaluation part
            psnr = metrics.peak_signal_noise_ratio(rgb.cpu().numpy(), gt_imgs_cpu[i], data_range=1)
            ssim = metrics.structural_similarity(rgb.cpu().numpy(), gt_imgs_cpu[i], multichannel=True, data_range=1)
            lpips_i = lpips_vgg(rgb.permute(2, 0, 1).unsqueeze(0), gt_imgs_gpu[i].permute(2, 0, 1).unsqueeze(0))
            psnrs.append(psnr)
            ssims.append(ssim)
            lpipses.append(lpips_i.item())
            print(f"PSNR: {psnr} SSIM: {ssim} LPIPS: {lpips_i.item()}")

            gt_label = gt_labels[i].cpu()
            if crop_mask is not None:
                valid_gt_labels = torch.unique(gt_label)[:-1]
                valid_gt_num = len(valid_gt_labels)
                gt_ins[..., :valid_gt_num] = F.one_hot(gt_label.long())[..., valid_gt_labels.long()]
                gt_label_np = valid_gt_labels.cpu().numpy()
                if valid_gt_num > 0:
                    mask = (gt_label < args.ins_num).type(torch.float32)
                    pred_label, ap, pred_matched_order = ins_eval(ins.cpu(), gt_ins, valid_gt_num, args.ins_num, mask)
                else:
                    pred_label = -1 * torch.ones([H, W])
                    ap = torch.tensor([1.0])
            else:
                valid_gt_labels = torch.unique(gt_label)
                valid_gt_num = len(valid_gt_labels)
                gt_ins[..., :valid_gt_num] = F.one_hot(gt_label.long())[..., valid_gt_labels.long()]
                gt_label_np = valid_gt_labels.cpu().numpy()  # change name
                if valid_gt_num > 0:
                    pred_label, ap, pred_matched_order = ins_eval(ins.cpu(), gt_ins, valid_gt_num, args.ins_num)
                else:
                    pred_label = -1 * torch.ones([H, W])
                    ap = torch.tensor([1.0])
            # get ins_map
            ins_map = {}
            for idx, pred_label_replica in enumerate(pred_matched_order):
                if pred_label_replica != -1:
                    ins_map[str(pred_label_replica)] = int(gt_label_np[idx])

            full_map[i] = ins_map

            aps.append(ap)
            print(f"APs: {ap}")

        if savedir is not None:
            rgb8 = to8b(rgb.cpu().numpy())
            ins_img = render_label2img(pred_label, ins_rgbs, color_dict, ins_map)
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            fileins = os.path.join(savedir, f"instance_{str(i).zfill(3)}.png")
            cv2.imwrite(fileins, ins_img)
            imageio.imwrite(filename, rgb8)

            gt_ins_img = render_gt_label2img(gt_label, ins_rgbs, color_dict)
            gt_img_file = os.path.join(savedir, f'{i}_ins_gt.png')
            cv2.imwrite(gt_img_file, gt_ins_img)
            gt_ins_file = os.path.join(savedir, f'{i}_ins_gt_mask.png')
            imageio.imwrite(gt_ins_file, np.array(gt_label.cpu().numpy(), dtype=np.uint8))

    if gt_imgs is not None:
        map_result_file = os.path.join(savedir, 'matching_log.json')
        with open(map_result_file, 'w') as f:
            json.dump(full_map, f)

        aps = np.array(aps)
        output = np.stack([psnrs, ssims, lpipses, aps[:, 0], aps[:, 1], aps[:, 2], aps[:, 3], aps[:, 4], aps[:, 5]])
        output = output.transpose([1, 0])
        out_ap = np.mean(aps, axis=0)
        mean_output = np.array([np.mean(psnrs), np.mean(ssims), np.mean(lpipses),
                                out_ap[0], out_ap[1], out_ap[2], out_ap[3], out_ap[4], out_ap[5]])
        mean_output = mean_output.reshape([1, 9])
        output = np.concatenate([output, mean_output], 0)
        test_result_file = os.path.join(savedir, 'test_results.txt')
        np.savetxt(fname=test_result_file, X=output, fmt='%.6f', delimiter=' ')
        print('=' * 49, 'Avg', '=' * 49)
        print('PSNR: {:.4f}, SSIM: {:.4f},  LPIPS: {:.4f} '.format(np.mean(psnrs), np.mean(ssims), np.mean(lpipses)))
        print('AP50: {:.4f}, AP75: {:.4f}, AP80: {:.4f}, AP85: {:.4f}, AP90: {:.4f}, AP95: {:.4f}'
              .format(out_ap[0], out_ap[1], out_ap[2], out_ap[3], out_ap[4], out_ap[5]))
