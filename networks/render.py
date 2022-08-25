import torch
import torch.nn.functional as F
from networks.helpers import sample_pdf


def render_train(raw, z_vals, rays_d):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    ins_labels = raw[..., 4:]
    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]

    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, -1)

    weights_ins = weights.clone()
    weights_ins = weights_ins.detach()
    ins_map = torch.sum(weights_ins[..., None] * ins_labels, -2)  # [N_rays, Class_number]
    ins_map = torch.sigmoid(ins_map)
    ins_map = ins_map[..., :-1]

    return rgb_map, weights, depth_map, ins_map


def dm_nerf(rays, position_embedder, view_embedder, model_coarse, model_fine, z_vals_coarse, args):
    # extract parameter
    # split rays
    rays_o, rays_d = rays

    # provide ray directions as input
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    if args.perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
        upper = torch.cat([mids, z_vals_coarse[..., -1:]], -1)
        lower = torch.cat([z_vals_coarse[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals_coarse.shape)
        z_vals_coarse = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_coarse[..., :, None]  # [N_rays, N_samples, 3]

    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])

    # coarse prediction part
    embedded_pos = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded_pos, embedded_dirs], -1)

    raw_coarse = model_coarse(embedded)
    raw_coarse = torch.reshape(raw_coarse, list(pts.shape[:-1]) + [raw_coarse.shape[-1]])
    # render part
    rgb_coarse, weights_coarse, depth_coarse, ins_coarse = render_train(raw_coarse, z_vals_coarse, rays_d)
    # N_importance
    # fine points sampling
    z_vals_mid = .5 * (z_vals_coarse[..., 1:] + z_vals_coarse[..., :-1])
    z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], args.N_importance, det=(args.perturb == 0.))
    z_samples = z_samples.detach()

    z_vals_fine, _ = torch.sort(torch.cat([z_vals_coarse, z_samples], -1), -1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_fine[..., :,
                                                        None]  # [N_rays, N_samples + N_importance, 3]

    pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])
    # fine prediction part
    embedded = position_embedder.embed(pts_flat)
    input_dirs = viewdirs[:, None].expand(pts.shape)
    input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
    embedded_dirs = view_embedder.embed(input_dirs_flat)
    embedded = torch.cat([embedded, embedded_dirs], -1)

    raw_fine = model_fine(embedded)
    raw_fine = torch.reshape(raw_fine, list(pts.shape[:-1]) + [raw_fine.shape[-1]])

    # fine render part
    rgb_fine, weights_fine, depth_fine, ins_fine = render_train(raw_fine, z_vals_fine, rays_d)

    if args.is_train and args.N_ins is not None:
        ins_fine = ins_fine[-args.N_ins:]
        ins_coarse = ins_coarse[-args.N_ins:]
    # return prediction of coarse and fine
    all_info = {'rgb_fine': rgb_fine, 'ins_fine': ins_fine, 'z_vals_fine': z_vals_fine, 'raw_fine': raw_fine,
                'raw_coarse': raw_coarse, 'rgb_coarse': rgb_coarse, 'ins_coarse': ins_coarse,
                'z_vals_coarse': z_vals_coarse, 'depth_fine': depth_fine, 'depth_coarse': depth_coarse}

    return all_info
