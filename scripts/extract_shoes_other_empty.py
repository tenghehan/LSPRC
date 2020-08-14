from data.load_rap_attributes_data_mat import *
import os
import shutil


if __name__ == '__main__':

    annotation_data = loadRAPAttr('data/RAP/RAP_attributes_data.mat')
    annotation_data = processRAPAttr(annotation_data)
    annotation_data = cleanRAPAttr(annotation_data)

    training_set = annotation_data['training_set']

    for index, (image_filename, annotation) in enumerate(training_set):
        origin_file = os.path.join('data/RAP/training_validation_images', image_filename)

        if annotation[2] == 1:
            new_file = os.path.join('data/age16', image_filename)
            shutil.copy(origin_file, new_file)
        # if annotation[37] == 1:
        #     new_file = os.path.join('data/shoes_other', image_filename)
        #     shutil.copy(origin_file, new_file)
        # if np.sum(annotation[32:38]) < 0.5:
        #     new_file = os.path.join('data/shoes_empty', image_filename)
        #     shutil.copy(origin_file, new_file)


