from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time
import os, cv2, glob, sys
import matplotlib.pyplot as plt
import saliency

def get_data_visual(filename, num_frames_per_clip, s_index=0):
    ret_arr = []
    filenames = ''
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        #print(filenames)
        if len(filenames)==0:
            print("Error, please check ...")

            return []
        if (len(filenames)-s_index) <= num_frames_per_clip:
            print("Not long enough, please check s_index ...")

            return []
        for i in range(num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i+s_index])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr

def get_data_vis(filename, num_frames_per_clip, s_index=0):
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename):
        filenames = sorted(filenames)
        num_frames = len(filenames)
        #print(filenames)
        if len(filenames)==0:
            print("Error, please check ...")
            return []

        if num_frames <= num_frames_per_clip:
            print("Not long enough, please check s_index ...")
            return []

        for i in range(num_frames_per_clip):
            image_name = str(filename) + '/' + str(filenames[i+s_index])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, num_frames


def get_data_vis_flow(filename, num_frames_per_clip, s_index=0):
    filename_x = filename +'/x/'
    filename_y = filename +'/y/'
    ret_arr = []
    for parent, dirnames, filenames in os.walk(filename_x):
        filenames = sorted(filenames)
        num_frames = len(filenames)
        #print(filenames)
        if len(filenames)==0:
            print("Error, please check ...")
            return []

        if num_frames <= num_frames_per_clip:
            print("Not long enough, please check s_index ...")
            return []

        for i in range(num_frames_per_clip):
            image_name_x = str(filename_x) + '/' + str(filenames[i+s_index])
            image_name_y = str(filename_y) + '/' + str(filenames[i+s_index])
            img = Image.open(image_name)
            img_data = np.array(img)
            ret_arr.append(img_data)
    return ret_arr, num_frames




def data_resize(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        height, width = tmp_data[j].shape[:2]
        if width > height:
            scale = float(crop_size) / float(height)
            img = cv2.resize(tmp_data[j], (int(width * scale + 1), crop_size))
        else:
            scale = float(crop_size) / float(width)
            img = cv2.resize(tmp_data[j], (int(high * scale + 1), crop_size))
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)

    return img_datas


def data_process(tmp_data, crop_size):
    img_datas = []
    for j in xrange(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale + 1), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale + 1)))).astype(np.float32)
        crop_x = int((img.shape[0] - crop_size) / 2)
        crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img_datas.append(img)

    return img_datas


def vis_ori(pathsave, video_tensor, num_frames):
    for i in range(num_frames):
        plt.imshow(video_tensor[i], cmap='jet')
        # plt.show()
        filesave = pathsave + "/orig_"+ str(i).zfill(4) +'.jpg'
        print(filesave)
        plt.savefig(filesave)

def vis_maps(pathsave, feature, num_frames):
    for i in range(num_frames):
        plt.imshow(feature[i], cmap='jet')
        # plt.show()
        filesave = pathsave + "/maps_"+ str(i).zfill(4) +'.jpg'
        # print(filesave)
        plt.savefig(filesave)

def vis_heat(pathsave, feature, num_frames):
    for i in range(num_frames):
        plt.imshow(feature[i], cmap='jet')
        # plt.show()
        filesave = pathsave + "/heat_"+ str(i).zfill(4) +'.jpg'
        # print(filesave)
        plt.savefig(filesave)

