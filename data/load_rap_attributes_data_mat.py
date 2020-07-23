# file: load_rap_attributes_data_mat.py
# brief: To use the mat data under python.
# author: CRIPAC
# version: 1.0
import sys
import numpy as np
from utils.file_operation import *

import random

def loadRAPAttr(data_filename):
    """
    load mat file under python.
    Note: It's not a general function, but based on the structure of 
    the variable in mat file.

    Input:
        data_filename - the mat file
    Return:
        Rap attributes data in a dictionary whose key is the same with
        that in mat file.
    """

    data = loadMat(data_filename)
    char_existed = data_filename.find('/')
    if char_existed == -1:
        root_key = data_filename.split('.')[0]
    else:
        filename = data_filename.split('/')[-1]
        root_key = filename.split('.')[0]

    # Train_Validation and Test sets
    tr_val_sets = data[root_key][0][0][0][0][0]
    test_set_items = data[root_key][0][0][1]
    test_set = np.asarray([ item[0][0] for item in test_set_items ])

    # Variables in Train_Validation
    tr_val_img_filenames_items = tr_val_sets[0]
    tr_val_img_filenames = np.asarray([ item[0][0] for item in \
        tr_val_img_filenames_items])
    attr_data = tr_val_sets[1]
    attr_names_cn_items = tr_val_sets[2]
    attr_names_cn = np.asarray([ item[0][0] for item in attr_names_cn_items ])
    attr_names_en_items = tr_val_sets[3]
    attr_names_en = np.asarray([ item[0][0] for item in attr_names_en_items ])
    partition = tr_val_sets[4]
    selected_attributes = tr_val_sets[5][0] - 1
    # Note that the minus one in selected_attributes as the index start with 0
    # in python but 1 in matlab!!

    # Partitions
    train_index = partition[0][0][0][0]
    val_index = partition[0][0][1][0]
    part = {'train_index': train_index, 'val_index': val_index}

    training_validation_sets = {'image_filenames': tr_val_img_filenames, \
        'attr_data': attr_data, 'attr_names_cn': attr_names_cn, \
        'attr_names_en': attr_names_en, 'partition': part, \
        'selected_attributes': selected_attributes}
    res = {'training_validation_sets': training_validation_sets, \
        'test_set': test_set}
    return res


def processRAPAttr(data):
    image_filenames = data['training_validation_sets']['image_filenames']
    attr_data = data['training_validation_sets']['attr_data']
    attr_names_cn = data['training_validation_sets']['attr_names_cn']
    attr_names_en = data['training_validation_sets']['attr_names_en']
    partition_train = data['training_validation_sets']['partition']['train_index']
    partition_val = data['training_validation_sets']['partition']['val_index']
    sel = data['training_validation_sets']['selected_attributes']
    test_set = data['test_set']

    attr_names_cn_selected = [attr_names_cn[i] for i in sel]
    attr_names_en_selected = [attr_names_en[i] for i in sel]
    attr_data_selected = np.array([attr_data[:, i] for i in sel]).T

    female = attr_data_selected[:, 0]
    male = 1 - female
    attr_data_extend = np.concatenate((male.reshape((-1, 1)), attr_data_selected), axis=1)
    attr_names_cn_extend = ['男性', '女性'] + attr_names_cn_selected[1:]
    attr_names_en_extend = ['Male', 'Female'] + attr_names_en_selected[1:]

    training_set = []
    validation_set = []
    for i in partition_train:
        image_filename = image_filenames[i - 1]
        attr = attr_data_extend[i - 1, :]
        training_set.append((image_filename, attr))
    for i in partition_val:
        image_filename = image_filenames[i - 1]
        attr = attr_data_extend[i - 1, :]
        validation_set.append((image_filename, attr))

    res = {'training_set': training_set, 'validation_set': validation_set, \
           'attr_names_cn': attr_names_cn_extend, 'attr_names_en': attr_names_en_extend, \
           'test_set': test_set}

    return res


def cleanRAPAttr(data):
    # wash data for gender=2
    training_set = data['training_set']
    validation_set = data['validation_set']
    training_set_clean = []
    validation_set_clean = []

    for info in training_set:
        if info[1][0] != 0 and info[1][0] != 1:
            continue
        training_set_clean.append(info)

    for info in validation_set:
        if info[1][0] != 0 and info[1][0] != 1:
            continue
        validation_set_clean.append(info)

    data['training_set'] = training_set_clean
    data['validation_set'] = validation_set_clean

    return data

if __name__=='__main__':

    #filename = \
    #    '/data1/da.li/projects/LSPR/data/Attributes/RAP_attributes_data.mat'
    filename = '/Users/tenghehan/Desktop/LSPRC/data/RAP_attributes_data.mat'
    data = loadRAPAttr(filename)

    data = processRAPAttr(data)

    data = cleanRAPAttr(data)

    # training_set = data['training_set']
    # validation_set = data['validation_set']
    # attr_names_cn = data['attr_names_cn']
    #
    # with open('training_attr.txt', 'w') as training_file:
    #     for info in training_set:
    #         print(info[0], file=training_file)
    #         for i in range(len(info[1])):
    #             if info[1][i] != 0:
    #                 print(attr_names_cn[i], ':', info[1][i], file=training_file)
    #
    # with open('validation_attr.txt', 'w') as val_file:
    #     for info in validation_set:
    #         print(info[0], file=val_file)
    #         for i in range(len(info[1])):
    #             if info[1][i] != 0:
    #                 print(attr_names_cn[i], ':', info[1][i], file=val_file)


