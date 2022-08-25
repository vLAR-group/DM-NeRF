import torch
import numpy as np

# shift helper
# Rotation matrix tools
r_x = lambda roll: np.array([
    [1, 0, 0, 0],
    [0, np.cos(roll), -np.sin(roll), 0],
    [0, np.sin(roll), np.cos(roll), 0],
    [0, 0, 0, 1]])
r_y = lambda pitch: np.array([
    [np.cos(pitch), 0, np.sin(pitch), 0],
    [0, 1, 0, 0],
    [-np.sin(pitch), 0, np.cos(pitch), 0],
    [0, 0, 0, 1]])
r_z = lambda yaw: np.array([
    [np.cos(yaw), -np.sin(yaw), 0, 0],
    [np.sin(yaw), np.cos(yaw), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]])


# Ray helper
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - (W - 1) * .5) / focal, (j - (H - 1) * .5) / focal, torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


# for scannet image process
def get_rays_k(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - K[0, 2]) / K[0, 0], (j - K[1, 2]) / K[1, 1], K[2, 2] * torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_select_crop(rgb, pose, K, ins_target, ins_index, crop_mask, N_train):
    N_ins = int(N_train * 0.3)

    if N_ins > len(ins_index):
        N_ins = len(ins_index)
    N_rgb = N_train - N_ins

    crop_mask = crop_mask.reshape(-1)
    H, W, C = rgb.shape
    rays_o, rays_d = get_rays_k(H, W, K, pose)
    loc_h, loc_w = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    loc_h, loc_w = torch.reshape(loc_h, [-1]).long(), torch.reshape(loc_w, [-1]).long()  # (H * W)
    crop_indices = np.where(crop_mask == 1)[0]

    labeled_selected_idx = np.random.choice(ins_index.shape[0], size=[N_ins], replace=False)  # (N_ins,)
    labeled_idx = ins_index[labeled_selected_idx]
    labeled_h, labeled_w = loc_h[labeled_idx], loc_w[labeled_idx]

    unlabeled_idx = np.array(list(set(crop_indices) - set(labeled_idx)))
    unlabeled_selected_idx = np.random.choice(len(unlabeled_idx), size=[N_rgb], replace=False)
    unlabeled_idx = crop_indices[unlabeled_selected_idx]
    unlabeled_h, unlabeled_w = loc_h[unlabeled_idx], loc_w[unlabeled_idx]

    selected_h, selected_w = torch.cat((unlabeled_h, labeled_h), dim=0), torch.cat((unlabeled_w, labeled_w), dim=0)

    rays_o = rays_o[selected_h, selected_w]  # (N_rgb, 3)
    rays_d = rays_d[selected_h, selected_w]  # (N_rgb, 3)
    batch_rays = torch.stack([rays_o, rays_d], 0)  # (N_rgb+N_ins, 3)
    target_c = rgb[selected_h, selected_w]  # (N_rgb, 3)
    target_i = ins_target[labeled_h, labeled_w]  # (N_ins, 3)

    return target_c, target_i, batch_rays, N_ins


# for other rooms
def get_select_full(rgb, pose, K, ins_target, N_train):
    H, W, C = rgb.shape
    rays_o, rays_d = get_rays_k(H, W, K, pose)
    loc_h, loc_w = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    loc_h, loc_w = torch.reshape(loc_h, [-1]).long(), torch.reshape(loc_w, [-1]).long()
    selected_index = np.random.choice(loc_h.shape[0], size=[N_train], replace=False)
    selected_h, selected_w = loc_h[selected_index], loc_w[selected_index]
    rays_o = rays_o[selected_h, selected_w]
    rays_d = rays_d[selected_h, selected_w]
    batch_rays = torch.stack([rays_o, rays_d], 0)  # (N_train, 3)
    target_c = rgb[selected_h, selected_w]  # (N_train, 3)
    target_i = ins_target[selected_h, selected_w]  # (N_train, 3)
    return target_c, target_i, batch_rays


def z_val_sample(N_rays, near, far, N_samples):
    near, far = near * torch.ones(size=(N_rays, 1)), far * torch.ones(size=(N_rays, 1))
    t_vals = torch.linspace(0., 1., steps=N_samples)
    z_vals_coarse = near + t_vals * (far - near)
    z_vals_coarse = z_vals_coarse.expand([N_rays, N_samples])
    return z_vals_coarse


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
