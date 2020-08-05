from typing import List

from interfaces.isee_interface import ISEEVisAlgIntf
import torch
from models.DeepMAR import DeepMAR_ResNet50
import os
import json
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms as T
from utils.tools import show_PILImage


class AttrRecognitionAlg(ISEEVisAlgIntf):

    def __init__(self):
        self.model = None
        self.device = None

        self.config = None

        self.results = None

    def init(self, config_file, params_dict=None):
        """
        Load model.
        :param config_file: the path of the configuration json file containing the necessary parameters.
        :param params_dict: the necessary parameters to initialize the project.
        {
            gpu_id: [-1], # the gpu id (a list of Integers), -1 means using CPU.
            model_path: ['/home/yourmodelpath', ..., ''], # a list of strings.
        }
        :return: error code: 0 for success; a negative number for the ERROR type.
        """
        if not os.path.exists(config_file):
            return -1
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        gpu_id = params_dict['gpu_id']
        model_path = params_dict['model_path']

        if len(gpu_id) < 1:
            return -4
        if gpu_id[0] == -1:
            self.device = torch.device('cpu')
        else:
            torch.cuda.set_device(gpu_id[0])
            self.device = torch.device(gpu_id[0])

        if not os.path.exists(model_path):
            return -1
        self.model = DeepMAR_ResNet50(self.config['attr_num'])
        self.model = self.model.to(self.device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        return 0

    def process(self, imgs_data: List[Image.Image], **kwargs):
        """
        Inference through loaded model.
        :param imgs_data: a list images to process
        :param kwargs: the necessary parameters to implement inference combining
                     the results of other tasks.
        :return: error code: 0 for success; a negative number for the ERROR type.
        """
        preds_probs = []

        with torch.no_grad():
            for image in tqdm(imgs_data):
                image = self.preprocess_image(image)
                image = image.to(self.device)
                batch = image.reshape([1, *image.shape])
                logit = self.model(batch)
                prob = torch.sigmoid(logit)
                preds_probs.append(prob.cpu().numpy())

        self.results = np.concatenate(preds_probs, axis=0)

        return 0

    def getResults(self):
        """
        Get the processed results.
        :return: The processing results. None without calling the function of process.
            type: Ndarray
            shape: (image_num x attr_num)
            value: float32 [0., 1.]
        """
        return self.results

    def release(self):
        pass

    def preprocess_image(self, image: Image.Image):
        """
        :param image: an image to preprocess
        :return: a tensor obtained after preprocessing
        """
        height = self.config['height']
        width = self.config['width']
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)