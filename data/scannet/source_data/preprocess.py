import os
import cv2
import csv
import copy
import glob
import shutil
import random
import pickle
import imageio
import zipfile
import datetime
import numpy as np
import scipy.misc, scipy.stats, scipy.io


from PIL import Image
from split import Split
from plyfile import PlyData, PlyElement
from SensorData_py3 import SensorData


class Data_configs:
    sem_names_all_nyu40 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                           'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
                           'floor mat',
                           'clothes', 'ceiling', 'books', 'refrigerator', 'television', 'paper', 'towel',
                           'shower curtain', 'box', 'whiteboard',
                           'person', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
                           'otherfurniture', 'otherprop']
    sem_ids_all_nyu40 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    sem_names_train_cls19 = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'bookshelf', 'counter', 'desk', 'shelves',
                             'dresser', 'pillow',
                             'refrigerator', 'television', 'box', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub']
    sem_ids_train_cls19 = [3, 4, 5, 6, 7, 9, 11, 13, 14, 16, 17, 23, 24, 28, 31, 32, 33, 35, 36]


##################################################

all_scans_folder = './scans/'
unzip_scans_folder = './selected_scenes/'
label_map_file = './selected_scenes/scannetv2-labels.combined.tsv'

ins_num_per_img = []


###################################################
def get_full_3d_mesh(scene_name=''):
    ## low res xyz-rgb mesh    --> scene0000_00_vh_clean_2.ply
    ## high res xyz-rgb mesh   --> scene0000_00_clean.ply

    ####### 3d mesh, low resolution
    scene_full_3d_mesh = PlyData.read(all_scans_folder + scene_name + '/' + scene_name + '_vh_clean_2.ply')
    scene_full_3d_pc = np.asarray((scene_full_3d_mesh['vertex'].data).tolist(), dtype=np.float32).reshape([-1, 7])
    scene_full_3d_face = np.asarray((scene_full_3d_mesh['face'].data).tolist(), dtype=np.int32).reshape([-1, 3])

    ####### 3d mesh, high resolution
    # scene_full_3d_mesh = PlyData.read(raw_scans_folder+scene_name+'/'+scene_name+'_vh_clean.ply')
    # scene_full_3d_pc = np.asarray((scene_full_3d_mesh['vertex'].data).tolist(), dtype=np.float32).reshape([-1, 10])
    # scene_full_3d_pc = np.concatenate([scene_full_3d_pc[:,0:3],scene_full_3d_pc[:,6:9]], axis=-1)
    # scene_full_3d_face = np.asarray((scene_full_3d_mesh['face'].data).tolist(), dtype=np.int32).reshape([-1, 3])

    ####### visualization
    # from helper_data_plot import Plot as Plot
    # Plot.draw_pc(scene_full_3d_pc[:,0:6])

    return scene_full_3d_pc, scene_full_3d_face


def unzip_raw_2d_files(raw_scans_folder, unzip_scans_folder):
    scene_names = sorted(os.listdir(raw_scans_folder))
    for scene in scene_names:
        if len(scene) < len('scene0000_00'): continue

        print(scene)
        if os.path.exists(unzip_scans_folder + scene):
            print('deleted and recreate scene folder');
            shutil.rmtree(unzip_scans_folder + scene)

        ### extract and save 3D data
        scene_full_3d_pc, scene_full_3d_face = get_full_3d_mesh(scene)
        out_sub_folder = unzip_scans_folder + scene + '/mesh/'
        if not os.path.isdir(out_sub_folder):
            os.makedirs(out_sub_folder)
        np.savez_compressed(out_sub_folder + 'scene_full_3d_pc.npz', scene_full_3d_pc=scene_full_3d_pc)
        np.savez_compressed(out_sub_folder + 'scene_full_3d_face.npz', scene_full_3d_face=scene_full_3d_face)

        ###  extract and save 2D data
        sensor_data_file = raw_scans_folder + scene + '/' + scene + '.sens'
        sensor_data = SensorData(sensor_data_file)
        if not os.path.exists(unzip_scans_folder + scene + '/color'):
            sensor_data.export_color_images(unzip_scans_folder + scene + '/color')
        if not os.path.exists(unzip_scans_folder + scene + '/depth'):
            sensor_data.export_depth_images(unzip_scans_folder + scene + '/depth')
        if not os.path.exists(unzip_scans_folder + scene + '/pose'):
            sensor_data.export_poses(unzip_scans_folder + scene + '/pose')
        if not os.path.exists(unzip_scans_folder + scene + '/intrinsic'):
            sensor_data.export_intrinsics(unzip_scans_folder + scene + '/intrinsic')

        print('unzip done:', scene)


###################################################

def represents_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    # if ints convert
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k): v for k, v in mapping.items()}
    return mapping


def map_sem_nyuID(image, label_mapping):
    mapped = np.copy(image)
    keys = np.unique(image)
    for k in keys:
        if k not in label_mapping: continue
        mapped[image == k] = label_mapping[k]
    return mapped


def map_sem_id(image, sem_ids_train):
    mapped_id = np.zeros((image.shape[0], image.shape[1]), dtype=np.int16) - 1
    for sem_i in sem_ids_train:
        new_id = sem_ids_train.index(sem_i)
        mapped_id[image == sem_i] = new_id
    return mapped_id


def map_ins_id(ins_image_in, sem_id):
    ins_image = copy.deepcopy(ins_image_in)
    ins_image[sem_id == -1] = -1  # filter the invalid pixels
    ins_ids = list(set(np.unique(ins_image)) - set([-1]))

    ins_num = len(ins_ids)
    ins_num_per_img.append(ins_num)

    ins_image_new = np.zeros(ins_image.shape, dtype=np.int16) - 1
    for new_id, ins_i in enumerate(ins_ids):
        sem_tp = np.unique(sem_id[ins_image == ins_i])
        if len(sem_tp) > 1:
            print('one ins has more than >1 sem, error');
            exit()
        if sem_tp[0] not in range(len(Data_configs.sem_ids_train_cls19)):
            print('the sem of ins is incorrect');
            exit()
        ins_image_new[ins_image == ins_i] = new_id

    # ins_ids = list(set(np.unique(ins_image_new)))
    return ins_image_new


def preprocess_imgs(scene_f):
    print('process folder:', scene_f)
    sem_mapping_dic = read_label_mapping(label_map_file, label_from='id', label_to='nyu40id')

    out_sem_f_id = scene_f + '/label-filt-cls' + str(len(Data_configs.sem_ids_train_cls19)) + '/'
    if os.path.exists(out_sem_f_id): print('deleted and recreate sem id'); shutil.rmtree(out_sem_f_id)
    os.makedirs(out_sem_f_id)

    out_ins_f_id = scene_f + '/instance-filt-cls' + str(len(Data_configs.sem_ids_train_cls19)) + '/'
    if os.path.exists(out_ins_f_id): print('deleted and recreate ins id'); shutil.rmtree(out_ins_f_id)
    os.makedirs(out_ins_f_id)

    total_imgs = sorted(glob.glob(scene_f + '/color/*.jpg'))
    for i in range(len(total_imgs)):
        sem_f = scene_f + '/label-filt/' + str(i) + '.png'
        ins_f = scene_f + '/instance-filt/' + str(i) + '.png'

        ## proj sem
        sem_2d_label_rawID = np.asarray(imageio.imread(sem_f), dtype=np.int16)
        sem_2d_label_nyuID = map_sem_nyuID(sem_2d_label_rawID, sem_mapping_dic)
        sem_2d_label_id = map_sem_id(sem_2d_label_nyuID, Data_configs.sem_ids_train_cls19)
        np.savez_compressed(out_sem_f_id + str(i) + '.npz', sem_2d_label_id=sem_2d_label_id)

        ## proj ins
        ins_2d_label_rawID = np.asarray(imageio.imread(ins_f), dtype=np.int16)
        ins_2d_label_id = map_ins_id(ins_2d_label_rawID, sem_2d_label_id)
        np.savez_compressed(out_ins_f_id + str(i) + '.npz', ins_2d_label_id=ins_2d_label_id)

        # from matplotlib import pyplot as plt
        # fig_input = plt.figure()
        # fig_input.add_subplot(2, 2, 1); plt.imshow(helper_util.visualize_instance_image(sem_2d_label_rawID ))
        # fig_input.add_subplot(2, 2, 2); plt.imshow(helper_util.visualize_instance_image(sem_2d_label_id ))
        # fig_input.add_subplot(2, 2, 3); plt.imshow(helper_util.visualize_instance_image(ins_2d_label_rawID ))
        # fig_input.add_subplot(2, 2, 4); plt.imshow(helper_util.visualize_instance_image(ins_2d_label_id ))

    return 0


################################################
if __name__ == '__main__':

    unzip_raw_2d_files(raw_scans_folder=all_scans_folder, unzip_scans_folder=unzip_scans_folder)
    scene_names = sorted(glob.glob(unzip_scans_folder + '*_*'))
    for scene_f in scene_names:
        preprocess_imgs(scene_f)
