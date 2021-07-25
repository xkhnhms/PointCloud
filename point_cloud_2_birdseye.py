import os
from os.path import join, exists
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                         POINT_CLOUD_2_BIRDSEYE
# ==============================================================================
def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-80., 80.),  # left-most to right-most
                           fwd_range = (-80., 80.), # back-most to forward-most
                           height_range=(-10., 10.),  # bottom-most to upper-most
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    im[y_img, x_img] = pixel_values

    return im

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

def imshow(tensor):
    tensor=tensor/2.0+0.5
    tensor=tensor.numpy()
    plt.imshow(np.transpose(tensor,(1,2,0)))
    plt.show()

def eliminateNan(list_a):
    # list_a = [elem if not np.isnan(elem) else None for elem in list_a]
    print(np.isnan(list_a).any(axis=1))
    while None in list_a:
        list_a.remove(None)
    return list_a

if __name__ == "__main__":

    # pc_path='./data/outdoor02/'
    pc_path = '/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/pcds/'

    result_filepath='/home/fs/PycharmProjects/KYXZ2018-G1/data/Raw-001/bird-view'
    pc_path_out = join(result_filepath, 'imgs')
    os.makedirs(pc_path_out) if not exists(pc_path_out) else None

    seq_list = np.sort(os.listdir(pc_path))

    for i in range(len(seq_list)):
        points = read_pcd2(os.path.join(pc_path, seq_list[i]))
        # points = getBin(os.path.join(pc_path, seq_list[i]))
        points = points.astype(np.float32)
        points = eliminateNan(points)
        points = points[~np.isnan(points).any(axis=1)]
        print(points)
        print(points.shape)
        filename = seq_list[i].split('.')[0] + '.png'
        saveImgData = os.path.join(pc_path_out, filename)

        img=point_cloud_2_birdseye(points)
        print(img.shape)


        # imshow(img)
        im = Image.fromarray(img)
        # im.save(saveImgData)
        im.convert('L').save(saveImgData)

        # if i==10:
        #     break


