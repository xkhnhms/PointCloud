import numpy as np
import pickle, yaml, os, sys
from os.path import join, exists, dirname, abspath

def read_pcd(pcd_path):
    lines = []
    num_points = None
    # with open(pcd_path, 'r', encoding='utf-8', errors='ignore') as f:
    with open(pcd_path, 'r') as f:
        # data = f.readlines()
        # print(data)
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        x, y, z, _,i,_ = list(map(float, line.split()))
        # points.append((np.array([x, y, z, 1.0]), i))
        points.append((np.array([x, y, z,i])))

    return np.asarray(points)

def read_pcd2(pcd_path):
    lines = []
    num_points = None

    with open(pcd_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        x, y, z,i ,_,_= list(map(float, line.split()))
        # points.append((np.array([x, y, z, 1.0]), i))
        points.append((np.array([x, y, z,i])))

    return np.asarray(points)

def saveText(points,filename):
    np.savetxt(filename, points, fmt='%f', delimiter=' ')

def getNpy(filename):
    data=np.load(filename)
    print(data)
    print(data.shape)
    return data

def getBin(filename):
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:, 0:3]  # get xyz
    # print(points)
    # print(points.shape)
    return points

def load_label_kitti(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def eliminateNan(list_a):
    # list_a = [elem if not np.isnan(elem) else None for elem in list_a]
    print(np.isnan(list_a).any(axis=1))
    while None in list_a:
        list_a.remove(None)
    return list_a


def saveBin(data,pc_path):
    data.tofile(pc_path)

def pcd2bin(filepath):
    pc_path = join(filepath, 'road06')
    pc_path_out = join(filepath, 'velodyne')
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None

    seq_list = np.sort(os.listdir(pc_path))

    for i in range(len(seq_list)):
        points = read_pcd(os.path.join(pc_path, seq_list[i]))
        points = points.astype(np.float32)
        filename = seq_list[i].split('.')[0] + '.bin'
        saveBinData = os.path.join(pc_path_out, filename)
        saveBin(points, saveBinData)

def showLabel(labelPath):

    BASE_DIR = dirname(abspath(__file__))
    data_config = os.path.join(BASE_DIR, 'semantic-kitti.yaml')
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

    label=load_label_kitti(labelPath,remap_lut) # [18 16 16 ...  9  9  9]  20662
    print(label)
    print(label.size)
    return label
    # saveText(label,"./road06_330Label.txt")


if __name__ == "__main__":

    # pc_path='./data/outdoor02/'
    pc_path = '/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/pcds'

    result_filepath='./data/semantic_kitti/dataset/sequences/30/'
    pc_path_out = join(result_filepath, 'velodyne')
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None

    #
    seq_list = np.sort(os.listdir(pc_path))

    for i in range(len(seq_list)):

        points=read_pcd2(os.path.join(pc_path,seq_list[i]))
        # points = getBin(os.path.join(pc_path, seq_list[i]))
        points=points.astype(np.float32)
        points=eliminateNan(points)
        points=points[~np.isnan(points).any(axis=1)]
        print(points)
        print(points.shape)
        filename=seq_list[i].split('.')[0]+'.bin'
        saveBinData = os.path.join(pc_path_out, filename)
        saveBin(points, saveBinData)
    #
    #
    #     filename2 = seq_list[i].split('.')[0] + '.txt'
    #     saveBinData2=os.path.join(pc_path_out2,filename2)
    #     # print(saveBinData)
    #     saveText(points, saveBinData2)
    #
    # points=getBin('../data4/velodyne/road06_1.bin')
    # print(points)
    # print(type(points))

    # points = read_pcd2('./autoware-210416.pcd')
    # points = points.astype(np.float32)
    # filename = './autoware-210416.bin'
    # saveBin(points, filename)


    # data=np.load("./003535.npy")
    # print(data)
    # print(data.shape)

    #
    # saveText(data,"./003535.txt")

    # scan = np.fromfile("./road06_9.bin", dtype=np.float32) # (120662, 4)
    # scan = scan.reshape((-1, 4))
    # # points = scan[:, 0:3]  # get xyz
    # points = scan# get xyz
    # print(points)
    # print(points.shape)
    # saveText(points,"./00630.txt")
    #
    # label=showLabel('./road06_9.label') # [18 16 16 ...  9  9  9]  20662
    # print(label)
    # print(label.size)
    # saveText(label,"./road06_330Label.txt")
