#coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from plyfile import PlyData

# 读取pcd数据
def read_pcd(pcd_path):
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
        x, y, z, i = list(map(float, line.split()))
        #这里没有把i放进去，也是为了后面 x, y, z 做矩阵变换的时候方面
        #但这个理解我选择保留， 因为可能把i放进去也不影响代码的简易程度
        # points.append((np.array([x, y, z, 1.0]), i))
        points.append((np.array([x, y, z])))

    return np.asarray(points)

# 读取.ply文件
def read_ply(pcd_path):
    plydata = PlyData.read(pcd_path)
    # with open(pcd_path, 'rb') as f:
    #     plydata = PlyData.read(f)
    # print(plydata)
    return plydata

def saveText(points):
    np.savetxt("./test.txt", points, fmt='%f', delimiter=' ')

def printPcdPointsInfo(points):
    print(points)
    print(type(points))
    print(len(points))
    print(points.shape)

def printPlyPointsInfo(plyData):
    print(plyData.elements[0].name)
    print(plyData.elements[0].data)
    print(type(plyData.elements[0].data))
    print(plyData['vertex']['z'])

def plotPointBar(dictData):
    l_unique_data=dictData.keys()
    l_num=dictData.values()
    plt.bar(l_unique_data, l_num, width=0.1)
    plt.show()

def getDims(points):

    dim1 ,dim2,dim3= [],[],[]

    for i in range(len(points)):
        # print(points[i][-1])
        dim1.append(points[i][0])
        dim2.append(points[i][1])
        dim3.append(points[i][-1])

    return dim1,dim2,dim3

def getNumCount(dim):

    data={}
    data_pd = pd.Series(dim)

    l_unique_data = list(data_pd.value_counts().index)  # 该数据中的唯一值
    l_num = list(data_pd.value_counts())  # 唯一值出现的次数
    # l_unique_data = str(data_pd.value_counts().index)  # 该数据中的唯一值
    # l_num = str(data_pd.value_counts())  # 唯一值出现的次数
    print(l_unique_data)
    print(l_num)
    # print(len(data_pd))

    plt.bar(l_unique_data, l_num, width=0.001)
    plt.show()

    # data.update({l_unique_data:l_num})
    return data

if __name__ == "__main__":

    '''
    # 1 处理PCD文件
    pcd_path='./autoware-210416.pcd'
    # pcd_path = './table_1_4.ply'
    points=read_pcd(pcd_path)

    print(len(points))
    print(points)

    dim1,dim2,dim3=getDims(points)

    print(min(dim3))
    print(mean(dim3))

    for i in range(len(points)):
        # print(points[i][-1])
        if points[i][-1]>mean(dim3):
            # print(points[i])
            del points[i]
            # points = np.delete(points,i, axis = 0)

    printPointsInfo(points)
    '''

    #处理PLY文件
    ply_path = './data/table_1_4.ply'
    plyData=read_ply(ply_path)
    plydata=plyData.elements[0].data
    # print(plydata)
    # print(plydata[0])
    # print(plydata[0][-1])
    # print(type(plydata[0][-1]))

    dim1, dim2, dim3 = getDims(plydata)
    # print(dim3)
    # print(len(dim3))
    # print(type(dim3))
    dim3=getNumCount(dim3)
    print(dim3.keys())
    print(dim3.values())
    # plotPointBar(dim3)








