import os
import json
import trimesh
import torch.nn.functional as F
import skimage.measure as ski_measure

from tools.visualizer import *
from networks.render import dm_nerf
from networks.helpers import z_val_sample


def mesh_main(position_embedder, view_embedder, model_coarse, model_fine, args, trimesh_scene, ins_rgbs, save_dir,
              ins_map=None):
    _, _, dataset_name, scene_name = args.datadir.split('/')
    gt_color_dict_path = './data/color_dict.json'
    gt_color_dict = json.load(open(gt_color_dict_path, 'r'))
    color_dict = gt_color_dict[dataset_name][scene_name]

    level = 0.45  # level = 0
    threshold = 0.2
    grid_dim = 256

    to_origin_transform, extents = trimesh.bounds.oriented_bounds(trimesh_scene)
    T_extent_to_scene = np.linalg.inv(to_origin_transform)
    scene_transform = T_extent_to_scene
    scene_extents = np.array([1.9, 7.0, 7.])
    grid_query_pts, scene_scale = grid_within_bound([-1.0, 1.0], scene_extents, scene_transform, grid_dim=grid_dim)
    grid_query_pts = grid_query_pts[:, :, [0, 2, 1]]
    grid_query_pts[:, :, 1] = grid_query_pts[:, :, 1] * -1
    query = grid_query_pts.numpy()[:, 0, :]
    grid_query_pts = grid_query_pts.cuda().reshape(-1, 3)  # Num_rays, 1, 3-xyz

    N = grid_query_pts.shape[0]
    raw = None

    for step in range(0, N, args.N_test):
        N_test = args.N_test
        if step + N_test > N:
            N_test = N - step
        in_pcd = grid_query_pts[step:step + N_test]
        embedded = position_embedder.embed(in_pcd)
        viewdirs = torch.zeros_like(in_pcd)
        embedded_dirs = view_embedder.embed(viewdirs)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        raw_fine = model_fine(embedded)
        if raw is None:
            raw = raw_fine
        else:
            raw = torch.cat((raw, raw_fine), dim=0)

    alpha = raw[..., 3]
    raw = raw.cpu().numpy()

    def occupancy_activation(alpha, distances):
        occ = 1.0 - torch.exp(-F.relu(alpha) * distances)
        # notice we apply RELU to raw sigma before computing alpha
        return occ

    voxel_size = (args.far - args.near) / args.N_importance  # or self.N_importance
    occ = occupancy_activation(alpha, voxel_size)
    print("Compute Occupancy Grids")
    occ = occ.reshape(grid_dim, grid_dim, grid_dim)
    occupancy_grid = occ.detach().cpu().numpy()

    print('fraction occupied:', (occupancy_grid > threshold).mean())
    print('Max Occ: {}, Min Occ: {}, Mean Occ: {}'.format(occupancy_grid.max(), occupancy_grid.min(),
                                                          occupancy_grid.mean()))
    vertices, faces, vertex_normals, _ = ski_measure.marching_cubes(occupancy_grid, level=level,
                                                                    gradient_direction='ascent')

    dim = occupancy_grid.shape[0]
    vertices = vertices / (dim - 1)
    mesh = trimesh.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces)

    # Transform to [-1, 1] range
    mesh_canonical = mesh.copy()

    vertices_ = np.array(mesh_canonical.vertices).reshape([-1, 3]).astype(np.float32)

    mesh_canonical.apply_translation([-0.5, -0.5, -0.5])
    mesh_canonical.apply_scale(2)

    scene_scale = scene_extents / 2.0
    # Transform to scene coordinates
    mesh_canonical.apply_scale(scene_scale)
    mesh_canonical.apply_transform(scene_transform)
    # mesh.show()

    exported = trimesh.exchange.export.export_mesh(mesh_canonical,
                                                   os.path.join(save_dir, 'mesh.ply'))
    print("Saving Marching Cubes mesh to mesh.ply !")

    o3d_mesh = trimesh_to_open3d(mesh)
    o3d_mesh_canonical = trimesh_to_open3d(mesh_canonical)

    print('Removing noise ...')
    print(f'Original Mesh has {len(o3d_mesh_canonical.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical.triangles) / 1e6:.2f} M faces.')
    o3d_mesh_canonical_clean = clean_mesh(o3d_mesh_canonical, keep_single_cluster=False, min_num_cluster=400)

    vertices_ = np.array(o3d_mesh_canonical_clean.vertices).reshape([-1, 3]).astype(np.float32)
    triangles = np.asarray(o3d_mesh_canonical_clean.triangles)  # (n, 3) int
    N_vertices = vertices_.shape[0]
    print(f'Denoised Mesh has {len(o3d_mesh_canonical_clean.vertices) / 1e6:.2f} M vertices and {len(o3d_mesh_canonical_clean.triangles) / 1e6:.2f} M faces.')

    selected_mesh = o3d_mesh_canonical_clean
    rays_d = - torch.FloatTensor(
        np.asarray(selected_mesh.vertex_normals))  # use negative normal directions as ray marching directions
    rays_d = rays_d[:, [0, 2, 1]]
    rays_d[:, 1] = rays_d[:, 1] * -1

    vertices_ = vertices_[:, [0, 2, 1]]
    vertices_[:, 1] = vertices_[:, 1] * -1
    rays_o = torch.FloatTensor(vertices_) - rays_d * 0.03 * args.near

    print(np.max(vertices_, axis=0), np.min(vertices_, axis=0))

    full_ins = None
    N = rays_o.shape[0]
    z_val_coarse = z_val_sample(args.N_test, 0.01, 15, args.N_samples)
    with torch.no_grad():
        for step in range(0, N, args.N_test):
            N_test = args.N_test
            if step + N_test > N:
                N_test = N - step
                z_val_coarse = z_val_sample(N_test, 0.01, 15, args.N_samples)
            rays_io = rays_o[step:step + N_test]  # (chuck, 3)
            rays_id = rays_d[step:step + N_test]  # (chuck, 3)
            batch_rays = torch.stack([rays_io, rays_id], dim=0)
            batch_rays = batch_rays.to(args.device)
            all_info = dm_nerf(batch_rays, position_embedder, view_embedder,
                               model_coarse, model_fine, z_val_coarse, args)
            if full_ins is None:
                full_ins = all_info['ins_fine']
            else:
                full_ins = torch.cat((full_ins, all_info['ins_fine']), dim=0)
    pred_label = torch.argmax(full_ins, dim=-1)
    ins_color = render_label2world(pred_label, ins_rgbs, color_dict, ins_map)

    o3d_mesh_canonical_clean.vertex_colors = o3d.utility.Vector3dVector(ins_color[:, [2, 1, 0]] / 255.0)
    o3d.io.write_triangle_mesh(
        os.path.join(save_dir, 'color_mesh.ply'),
        o3d_mesh_canonical_clean)
    print("Saving Marching Cubes mesh to color_mesh.ply")
