from interfaces.isee_interface import ISEEVisAlgIntf
from interfaces.impl import AttrRecognitionAlg
import os
from PIL import Image


def init_model(alg: ISEEVisAlgIntf):
    params_dict = {
        'gpu_id': gpu_id,
        'model_path': model_path
    }
    r = alg.init(config_path, params_dict)
    assert r == 0


def inference(alg: ISEEVisAlgIntf):
    filenames = os.listdir(image_dir_path)
    image_list = []
    for filename in filenames:
        image = Image.open(os.path.join(image_dir_path, filename))
        image_list.append(image)

    r = alg.process(image_list)
    assert r == 0

    return filenames, alg.getResults()


def output(filenames, results):
    for i in range (len(filenames)):
        print(filenames[i], ':', end='')
        print(','.join(['%.6f' % a for a in results[i]]))


def demo(alg: ISEEVisAlgIntf):
    init_model(alg)
    filenames, results = inference(alg)
    output(filenames, results)


if __name__ == '__main__':
    image_dir_path = 'images'
    config_path = 'config.json'
    model_path = '../results/deepMAR/max.pth'

    gpu_id = [-1]

    impl = AttrRecognitionAlg()
    demo(impl)
