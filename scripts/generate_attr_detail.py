import sys
import numpy as np
from utils.file_operation import *
from data.load_rap_attributes_data_mat import *
import random

if __name__=='__main__':

    #filename = \
    #    '/data1/da.li/projects/LSPR/data/Attributes/RAP_attributes_data.mat'
    filename = '../data/RAP/RAP_attributes_data.mat'
    data = loadRAPAttr(filename)

    data = processRAPAttr(data)

    data = cleanRAPAttr(data)

    training_set = data['training_set']
    validation_set = data['validation_set']
    attr_names_cn = data['attr_names_cn']

    with open('training_attr.txt', 'w') as training_file:
        for info in training_set:
            print(info[0], file=training_file)
            for i in range(len(info[1])):
                if info[1][i] != 0:
                    print(attr_names_cn[i], ':', info[1][i], file=training_file)

    with open('validation_attr.txt', 'w') as val_file:
        for info in validation_set:
            print(info[0], file=val_file)
            for i in range(len(info[1])):
                if info[1][i] != 0:
                    print(attr_names_cn[i], ':', info[1][i], file=val_file)


