from argparse import ArgumentParser
from os import listdir, mkdir
from os.path import isfile, join, isdir
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np


def convert_files(img_dir, class_name, out_size, dirs):
    def find_first(a):
        for i in range(len(a)):
            if a[i] > 0:
                return i
        return None

    def find_last(a):
        for i in range(len(a) - 1, -1, -1):
            if a[i] > 0:
                return i
        return None

    def find_fl(a):
        return find_first(a), find_last(a)

    count = 0
    for f in listdir(img_dir):
        fn = join(img_dir, f)
        if f.endswith('.png') and isfile(fn):
            data = imread(fn)
            shape = data.shape
            data = 255 - data[:, :, 0] #shape[2] - 1]
            wnz = np.max(data, axis=0) > 0
            hnz = np.max(data, axis=1) > 0
            (min_x, max_x) = find_fl(wnz)
            (min_y, max_y) = find_fl(hnz)
            if min_x is None:
                continue
            w, h = max_x - min_x + 1, max_y - min_y + 1
            size = max(w, h)
            data = data[min_y:max_y + 1, min_x:max_x + 1]
            if w < size:
                dw = int((size - w) / 2)
                data = np.hstack((np.zeros((h, dw)), data, np.zeros((h, dw))))
            elif h < size:
                dh = int((size - h) / 2)
                data = np.vstack((np.zeros((dh, w)), data, np.zeros((dh, w))))
            data = resize(data / 255.0, (out_size, out_size))
            if count == 9:
                out_dir = dirs[2]
            elif count >= 7:
                out_dir = dirs[1]
            else:
                out_dir = dirs[0]
            out_file = join(out_dir, class_name, f)
            imsave(out_file, data)
            count += 1
            if count >= 10:
                count = 0


def create_dir(dir_type, class_name):
    if not isdir(dir_type):
        mkdir(dir_type)
    cd = join(dir_type, class_name)
    if not isdir(cd):
        mkdir(cd)


parser = ArgumentParser(description='Image loader')
parser.add_argument('class_name', type=str, help='Class name')
parser.add_argument('-s', type=int, default=128, help='Image size')
parser.add_argument('image_directory', type=str, help='Directory with input images')
args = parser.parse_args()
class_name = args.class_name
size = str(args.s)

dirs = ('train' + size, 'validate' + size, 'test' + size)

for d in dirs:
    create_dir(d, class_name)

convert_files(args.image_directory, args.class_name, args.s, dirs)
