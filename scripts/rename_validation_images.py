from data.load_rap_attributes_data_mat import *
import os
import shutil


if __name__ == '__main__':

    annotation_data = loadRAPAttr('data/RAP/RAP_attributes_data.mat')
    annotation_data = processRAPAttr(annotation_data)
    annotation_data = cleanRAPAttr(annotation_data)

    validation_set = annotation_data['validation_set']

    for index, (image_filename, annotation) in enumerate(validation_set):
        origin_file = os.path.join('data/RAP/training_validation_images', image_filename)
        new_filename = str(index + 1) + '.png'
        new_file = os.path.join('data/renamed_validation', new_filename)
        shutil.copy(origin_file, new_file)


