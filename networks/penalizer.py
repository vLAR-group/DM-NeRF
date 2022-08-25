import torch
import numpy as np


def emptiness_penalizer(raw, z_vals, depths, rays_d, tolerance, deta_w):
    # initial gaussian distribution algorithm
    Gaussian_Distribution = lambda delta_dist, deta_h, deta_w: torch.exp(
        -(delta_dist ** 2) / (2 * (deta_w ** 2))) / (deta_h * torch.sqrt(torch.Tensor([2 * np.pi]))) + 1e-8

    delta_H, delta_W = torch.Tensor([0.4]), torch.Tensor([deta_w])

    # calculate distances between two points
    norm = torch.norm(rays_d[..., None, :], dim=-1)
    depths_before = depths - tolerance  # (N_rays, 1)
    depths_after = depths + tolerance  # (N_rays, 1)
    dists_before = depths_before * norm  # (N_rays, 1)
    dists_after = depths_after * norm  # (N_rays, 1)
    depth_dist = depths * norm
    p_dists = z_vals * norm

    # calculate weights
    delta_dist = depth_dist - p_dists  # (N_rays, N_points)
    penalize_weights = Gaussian_Distribution(delta_dist, delta_H, delta_W)
    penalize_weights_air = 1 - penalize_weights

    # prepare mask
    mask_before = (p_dists < dists_before).type(torch.float32)  # (N_rays, N_points)
    mask_after = (p_dists > dists_after).type(torch.float32)  # (N_rays, N_points)
    mask_middle = 1 - (mask_after + mask_before)  # (N_rays, N_points)

    # if depth is at the begin or end the mask will be zero, so plus a value closes to zero
    pred_ins = raw[..., 4:]  # (N_rays, N_point, ins_number+1)
    pred_ins = torch.sigmoid(pred_ins)

    # calculate begin part of penalize losses and penalize all ins num + 1 postions

    gt_labeles = torch.zeros_like(pred_ins)
    gt_labeles[..., -1] = 1
    loss_before = -gt_labeles * torch.log(pred_ins + 1e-8) - (1 - gt_labeles) * torch.log(1 - pred_ins + 1e-8)
    masked_penalize_weights_air = penalize_weights_air * mask_before
    loss_before = loss_before * masked_penalize_weights_air[..., None]  # (N_rays, N_points, ins + 1)
    loss_before = torch.sum(loss_before) / (
            pred_ins.shape[-1] * torch.maximum(torch.sum(mask_before), torch.tensor([1e-8])))  # one value

    # calculate middle part
    penalized_ins_middle = pred_ins[..., -1]
    gt_labeled_middle = torch.zeros_like(penalized_ins_middle)
    loss_middle = -gt_labeled_middle * torch.log(penalized_ins_middle + 1e-8) - (1 - gt_labeled_middle) * torch.log(
        1 - penalized_ins_middle + 1e-8)
    masked_penalize_weights = penalize_weights * mask_middle
    loss_middle = loss_middle * masked_penalize_weights
    loss_middle = torch.sum(loss_middle) / torch.maximum(torch.sum(mask_middle), torch.tensor([1e-8]))  # one value
    penalize_loss = loss_before + loss_middle

    return penalize_loss


def ins_penalizer(raw, z_vals, depth, rays_d, args):
    depth = depth[..., None].detach()
    valid_penalize_loss = emptiness_penalizer(raw, z_vals, depth, rays_d, args.tolerance, args.deta_w)

    return valid_penalize_loss
