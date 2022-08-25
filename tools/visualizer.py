import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def ins2img(predicted_onehot, rgbs):
    predicted_labels = torch.argmax(predicted_onehot, dim=-1)
    predicted_labels = predicted_labels.cpu()
    unique_labels = torch.unique(predicted_labels)
    rgb = [0, 0, 0]
    h, w = predicted_labels.shape
    ra_in_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        if label == 0:
            ra_in_im_t[predicted_labels == label] = rgb
        else:
            ra_in_im_t[predicted_labels == label] = rgbs[label]
    return ra_in_im_t.astype(np.uint8)


# vis instance map after shift
def manipulator_label2img(predicted_labels, rgbs):
    predicted_labels = predicted_labels.cpu()
    unique_labels = torch.unique(predicted_labels)
    rgb = [0, 0, 0]
    sh = predicted_labels.shape
    ra_in_im_t = np.zeros(shape=(sh[0], sh[1], 3))
    for index, label in enumerate(unique_labels):
        if label == 32:
            ra_in_im_t[predicted_labels == label] = rgb
        else:
            ra_in_im_t[predicted_labels == label] = rgbs[label]
    return ra_in_im_t.astype(np.uint8)


# vis instance map after matching
def matching_label2img(predicted_labels, rgbs):
    unique_labels = torch.unique(predicted_labels).long()
    predicted_labels = predicted_labels.cpu()
    unique_labels = unique_labels.cpu()
    rgb = [0, 0, 0]
    unmatched_rgb = [255, 255, 255]
    h, w = predicted_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        if label == -1:
            ra_se_im_t[predicted_labels == label] = rgb
        elif label == -2:
            ra_se_im_t[predicted_labels == label] = unmatched_rgb
        else:
            ra_se_im_t[predicted_labels == label] = rgbs[label]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t


def render_gt_label2img(gt_labels, rgbs, color_dict):
    unique_labels = torch.unique(gt_labels)
    gt_labels = gt_labels.cpu()
    unique_labels = unique_labels.cpu()
    h, w = gt_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        label_cpu = str(int(label.cpu()))
        gt_keys = color_dict.keys()
        if label_cpu in gt_keys:
            ra_se_im_t[gt_labels == label] = rgbs[color_dict[str(label_cpu)]]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t


# vis instance at testing phrase
def render_label2img(predicted_labels, rgbs, color_dict, ins_map):
    unique_labels = torch.unique(predicted_labels)
    predicted_labels = predicted_labels.cpu()
    unique_labels = unique_labels.cpu()
    h, w = predicted_labels.shape
    ra_se_im_t = np.zeros(shape=(h, w, 3))
    for index, label in enumerate(unique_labels):
        label_cpu = str(int(label.cpu()))
        gt_keys = ins_map.keys()
        if label_cpu in gt_keys:
            gt_label_cpu = ins_map[label_cpu]
            ra_se_im_t[predicted_labels == label] = rgbs[color_dict[str(gt_label_cpu)]]
    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t


# get all instance rgbs and corresponding labels
def show_instance_rgb(ins_rgbs, save_rgbs_file):
    show_boxes = np.zeros(shape=[len(ins_rgbs), 8, 8, 3])
    y_ax = 4
    x_ax = int((len(ins_rgbs)) // y_ax)
    fig, ax = plt.subplots(x_ax, y_ax, figsize=(8, 8))
    fontdict = {'fontsize': 6}
    for i in range(len(ins_rgbs)):
        rgb = ins_rgbs[i]
        show_boxes[i, ..., 0:3] = rgb
        x_index, y_index = i // y_ax, i % y_ax
        ax[x_index][y_index].imshow(show_boxes[i].astype(np.uint8))
        ax[x_index][y_index].set_title(f'Label:{i}: [{str(rgb[0])},{str(rgb[1])},{str(rgb[2])}]',
                                       fontdict=fontdict)
        ax[x_index][y_index].grid(False)
        ax[x_index][y_index].axis('off')
    plt.savefig(save_rgbs_file)
    # plt.show()
    return


# 3d mesh part
def make_3D_grid(occ_range, dim, transform=None, scale=None):
    t = torch.linspace(occ_range[0], occ_range[1], steps=dim)
    grid = torch.meshgrid(t, t, t)
    grid_3d_norm = torch.cat(
        (grid[0][..., None],
         grid[1][..., None],
         grid[2][..., None]), dim=3
    )

    if scale is not None:
        grid_3d = grid_3d_norm.cpu() * scale
    if transform is not None:
        R1 = transform[None, None, None, 0, :3]
        R2 = transform[None, None, None, 1, :3]
        R3 = transform[None, None, None, 2, :3]

        grid1 = (R1 * grid_3d).sum(-1, keepdim=True)
        grid2 = (R2 * grid_3d).sum(-1, keepdim=True)
        grid3 = (R3 * grid_3d).sum(-1, keepdim=True)
        grid_3d = torch.cat([grid1, grid2, grid3], dim=-1)

        trans = transform[None, None, None, :3, 3]
        grid_3d = grid_3d + trans

    return grid_3d


def grid_within_bound(occ_range, extents, transform, grid_dim):
    range_dist = occ_range[1] - occ_range[0]
    bounds_tranform_np = transform

    bounds_tranform = torch.from_numpy(bounds_tranform_np).float()
    scene_scale_np = extents / (range_dist * 1.0)
    scene_scale = torch.from_numpy(scene_scale_np).float()

    # todo: only make grid once, then only transform!
    grid_pc = make_3D_grid(
        occ_range,
        grid_dim,
        transform=bounds_tranform,
        scale=scene_scale,
    )
    grid_pc = grid_pc.view(-1, 1, 3)

    return grid_pc, scene_scale


def trimesh_to_open3d(src):
    dst = o3d.geometry.TriangleMesh()
    dst.vertices = o3d.utility.Vector3dVector(src.vertices)
    dst.triangles = o3d.utility.Vector3iVector(src.faces)
    vertex_colors = src.visual.vertex_colors[:, :3].astype(np.float) / 255.0
    dst.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    dst.compute_vertex_normals()

    return dst


def clean_mesh(o3d_mesh, keep_single_cluster=False, min_num_cluster=200):
    import copy

    o3d_mesh_clean = copy.deepcopy(o3d_mesh)
    # http://www.open3d.org/docs/release/tutorial/geometry/mesh.html?highlight=cluster_connected_triangles
    triangle_clusters, cluster_n_triangles, cluster_area = o3d_mesh_clean.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    if keep_single_cluster:
        # keep the largest cluster.!
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with largest cluster kept")
    else:
        # remove small clusters
        triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_num_cluster
        o3d_mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        o3d_mesh_clean.remove_unreferenced_vertices()
        print("Show mesh with small clusters removed")

    return o3d_mesh_clean


def render_label2rgb(predicted_labels, rgbs):
    predicted_labels = predicted_labels
    unique_labels = np.unique(predicted_labels)
    N = predicted_labels.shape[0]
    print(N)
    ra_in_im_t = np.zeros(shape=(N, 3))
    for index, label in enumerate(unique_labels):
        ra_in_im_t[predicted_labels == label] = rgbs[label]
    return ra_in_im_t.astype(np.uint8)


def render_label2world(predicted_labels, rgbs, color_dict, ins_map):
    unique_labels = torch.unique(predicted_labels)
    predicted_labels = predicted_labels.cpu()
    unique_labels = unique_labels.cpu()
    N = predicted_labels.shape[0]
    ra_se_im_t = np.zeros(shape=(N, 3))
    for index, label in enumerate(unique_labels):
        # label_cpu = str(int(label.cpu()))
        label_cpu = str(int(label))
        gt_keys = ins_map.keys()
        if label_cpu in gt_keys:
            gt_label_cpu = ins_map[label_cpu]
            ra_se_im_t[predicted_labels == label] = rgbs[color_dict[str(gt_label_cpu)]]

    ra_se_im_t = ra_se_im_t.astype(np.uint8)
    return ra_se_im_t
