import os
import random
import shutil

from copy import deepcopy


def rename_img(root):
    dataset_dir = os.path.join(root)
    class_name = ['1', '3', '4', '5', '6', '7',
                  '8', '9', '10', '11', '12', '13', '14',
                  '15', '16', '17']

    for c_name in class_name:
        c_dir = os.path.join(dataset_dir, c_name)
        pathdir = os.listdir(c_dir)
        list_len = len(pathdir)
        print(f"the number of val sample is {list_len}")
        for sample in pathdir:
            old_name = sample.split('.')[0]
            new_name = old_name + '_12pm'
            shutil.move(os.path.join(c_dir, old_name + '.jpg'), os.path.join(c_dir, new_name + '.jpg'))


def copy_file(src_root, des_root):
    dataset_dir = os.path.join(src_root)
    class_name = ['1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '10', '11', '12', '13', '14',
                  '15', '16', '17']

    for c_name in class_name:
        c_dir = os.path.join(dataset_dir, c_name)
        des_dir = os.path.join(des_root, c_name)
        pathdir = os.listdir(c_dir)
        list_len = len(pathdir)
        print(f"the number of val sample is {list_len}")
        for sample in pathdir:
            shutil.copy(os.path.join(c_dir, sample), os.path.join(des_dir, sample))


if __name__ == '__main__':
    copy_file('./datasets/cow_face/17/val', './datasets/cow_face/mix/val')
