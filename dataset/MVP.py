import torch
import numpy as np
import torch.utils.data as data
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import h5py
from .build import DATASETS

def rotate_point_cloud(complete,partial,part, angle_x=0, angle_y=0, angle_z=0, rx=False,ry = False,rz = False):
    rcomplete = np.copy(complete)
    rpartial = np.copy(partial)
    rpart = np.copy(part)

    
    if rx:
        # 随机旋转角度
        angle_x = (np.random.uniform() * 2 * np.pi)
        # angle_x = angle_x
    if ry:
        # 随机旋转角度
        angle_y = np.random.uniform() * 2 * np.pi
    if rz:
        # 随机旋转角度
        angle_z = np.random.uniform() * 2 * np.pi
    
    # 绕 x 轴旋转
    rotation_matrix_x = np.array([[1, 0, 0],
                                  [0, np.cos(angle_x), -np.sin(angle_x)],
                                  [0, np.sin(angle_x), np.cos(angle_x)]])
    rcomplete = np.dot(rcomplete, rotation_matrix_x.T)
    rpartial = np.dot(rpartial, rotation_matrix_x.T)
    rpartial = np.dot(rpartial, rotation_matrix_x.T)

    # 绕 y 轴旋转
    rotation_matrix_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                                  [0, 1, 0],
                                  [-np.sin(angle_y), 0, np.cos(angle_y)]])
    # rotated_point_cloud = np.dot(rotated_point_cloud, rotation_matrix_y.T)
    rcomplete = np.dot(rcomplete, rotation_matrix_y.T)
    rpartial = np.dot(rpartial, rotation_matrix_y.T)
    rpartial = np.dot(rpartial, rotation_matrix_y.T)

    # 绕 z 轴旋转
    rotation_matrix_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                                  [np.sin(angle_z), np.cos(angle_z), 0],
                                  [0, 0, 1]])
    rcomplete = np.dot(rcomplete, rotation_matrix_z.T).astype(np.float32)
    rpartial = np.dot(rpartial, rotation_matrix_z.T).astype(np.float32)
    rpartial = np.dot(rpartial, rotation_matrix_z.T).astype(np.float32)

    return rcomplete,rpartial,rpart



@DATASETS.register_module()
class MVP_CP(data.Dataset):
    def __init__(self, config):
        prefix="train"
        if prefix=="train":
            self.file_path = '/home/hhy/dataset/MVP/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = '/home/hhy/dataset/MVP/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = '/home/hhy/dataset/MVP/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = config.subset

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        # print(self.input_data.shape)

        if prefix is not "test":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            # print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        if self.prefix is not "test":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
        else:
            return partial
        

@DATASETS.register_module()
class MVP_CP_R(data.Dataset):
    def __init__(self, config):
        prefix="train"
        if prefix=="train":
            self.file_path = '/home/hhy/dataset/MVP/MVP_Train_CP.h5'
        elif prefix=="test":
            self.file_path = '/home/hhy/dataset/MVP/MVP_Test_CP.h5'
        # elif prefix=="test":
        #     self.file_path = '/home/hhy/dataset/MVP/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = config.subset

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        # print(self.input_data.shape)

        if prefix is not "val":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            # print(self.gt_data.shape, self.labels.shape)


        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        if self.prefix is not "val":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            complete_pc_r,partial_pc_r,_ = rotate_point_cloud(complete,partial,partial,ry=False,rx= True,rz=False)
            return label, label, (partial_pc_r, complete) # partial_pc_r B N 2048 , complete B N 2048
        else:
            return partial



