from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd
import glob
from os.path import join, exists, dirname, abspath
import pickle, yaml, os, sys
from helper_ply import read_ply
from helper_tool import Plot


# sudo apt-get install ros-melodic-velodyne-*
# roslaunch velodyne_pointcloud 32e_points.launch pcap:=/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/Raw-001-HDL32.pcap
# rosrun pcl_ros pointcloud_to_pcd input:=/velodyne_points
# roslaunch velodyne_pointcloud 32e_points.launch pcap:=/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/Raw-001-HDL32.pcap  gps_time:=/false  pcap_time:=/true (issue still)
# roslaunch velodyne_pointcloud 32e_points.launch pcap:=/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/Raw-001-HDL32.pcap pcap_time:=true read_once:=true

def draw_pc_sem_ins(pc_xyz, pc_sem_ins, plot_colors=None):
    """
    pc_xyz: 3D coordinates of point clouds
    pc_sem_ins: semantic or instance labels
    plot_colors: custom color list
    """
    if plot_colors is not None:
        ins_colors = plot_colors
    else:
        ins_colors = Plot.random_colors(len(np.unique(pc_sem_ins)) + 1, seed=2)

    ##############################
    sem_ins_labels = np.unique(pc_sem_ins)
    sem_ins_bbox = []
    Y_colors = np.zeros((pc_sem_ins.shape[0], 3))
    for id, semins in enumerate(sem_ins_labels):
        valid_ind = np.argwhere(pc_sem_ins == semins)[:, 0]
        if semins <= -1:
            tp = [0, 0, 0]
        else:
            if plot_colors is not None:
                tp = ins_colors[semins]
            else:
                tp = ins_colors[id]

        Y_colors[valid_ind] = tp

        ### bbox
        valid_xyz = pc_xyz[valid_ind]

        xmin = np.min(valid_xyz[:, 0]);
        xmax = np.max(valid_xyz[:, 0])
        ymin = np.min(valid_xyz[:, 1]);
        ymax = np.max(valid_xyz[:, 1])
        zmin = np.min(valid_xyz[:, 2]);
        zmax = np.max(valid_xyz[:, 2])
        sem_ins_bbox.append(
            [[xmin, ymin, zmin], [xmax, ymax, zmax], [min(tp[0], 1.), min(tp[1], 1.), min(tp[2], 1.)]])

    Y_semins = np.concatenate([pc_xyz[:, 0:3], Y_colors], axis=-1)
    Plot.draw_pc(Y_semins)
    return Y_semins

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)

def getBin(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # get xyz
    # print(points)
    # print(points.shape)
    return points

def showLabel(labelPath):

    BASE_DIR = dirname(abspath(__file__))
    data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    label=load_label_kitti(labelPath,remap_lut) # [18 16 16 ...  9  9  9]  20662
    return label

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import read_ply
from helper_tool import Plot

if __name__ == '__main__':
    # base_dir = './data/semantic_kitti/dataset/sequences_0.06/26/'
    base_dir = './data/semantic_kitti/dataset/sequences/30/'
    # base_dir = './data/semantic_kitti/dataset/sequences_0.06/08/'
    # (88109, 3)
    # (123389,)

    # (123389, 3)
    # (123389,)

    data_path = glob.glob(os.path.join(base_dir, 'velodyne', '*.bin'))
    # data_path = glob.glob(os.path.join(base_dir, 'velodyne','*.npy'))
    # base_dir = './data/semantic_kitti/dataset/sequences_0.06/04/'
    # data_path = glob.glob(os.path.join(base_dir, 'velodyne', '*.npy'))
    data_path = np.sort(data_path)

    for file_name in data_path:
        print(file_name)
        points_data = getBin(file_name)
        # points_data = np.load(file_name)
        filename = file_name.split('/')[-1].split('.')[0] + '.label'
        # filename = file_name.split('/')[-1].split('.')[0] + '.npy'
        print(filename)
        labels = os.path.join(base_dir, 'predictions', filename)
        # labels =os.path.join(base_dir,'labels',filename)
        labels=showLabel(labels)
        # labels = np.load(labels)
        print(labels)

        print(points_data.shape)
        print(labels.shape)
        print('--------------')

        element = 6  # 6: 'person'
        count_person = list(labels).count(element)
        print(count_person)
        if count_person>0:
            Plot.draw_pc_sem_ins(points_data, labels)

        # if points_data.shape[0]!=labels.shape[0]:
        #     continue
        # Plot.draw_pc_sem_ins(points_data, labels[labels.shape[0]-points_data.shape[0]:])
        # Plot.draw_pc_sem_ins(points_data, labels)


