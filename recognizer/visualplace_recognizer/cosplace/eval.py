
import sys
import os
import numpy as np
import faiss
import tqdm
import torch
import logging
import multiprocessing
from datetime import datetime
from torch.utils.data import DataLoader
from PIL import Image
torch.backends.cudnn.benchmark= True  # Provides a speedup

import recognizer.visualplace_recognizer.cosplace.test as test
import configparser
import recognizer.visualplace_recognizer.cosplace.commons as commons
from recognizer.visualplace_recognizer.cosplace.model import network
from recognizer.visualplace_recognizer.cosplace.datasets.test_dataset import RawDataset

class CosPlace:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='cosplace'):
        self.config = self._parse_config()
        commons.make_deterministic(self.config['seed'])
        self.results = dict()
        self.model_name = model

        self.model = network.GeoLocalizationNet(self.config['backbone'], self.config['fc_output_dim'])

    def _get_db_descriptors(self):
        """
        TODO:
        :return:
        """

    def _parse_config(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(self.path, 'config/config.yml'))

        return config


    def inference_by_image(self, image):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image File {image} does not exist!")

        query_img = Image.open(image).convert("RGB")
        if '.npy' in self.config['dataset_path']:
            # Get Database by file
            # TODO: try catch로 예외 처리 나중에 넣기
            database_descriptors = np.load(self.config['dataset_path'])
        else:
            database_ds = RawDataset(self.config['dataset_path'])
            with torch.no_grad():

                # Get Database Descriptor by Directly
                logging.debug("Extracting database descriptors for evaluation/testing")
                database_dataloader = DataLoader(dataset=database_ds, num_workers=self.config['num_workers'],
                                                 batch_size=self.config['infer_batch_size'], pin_memory=(self.config['device'] == "cuda"))
                database_descriptors = np.empty((len(database_ds), self.config['fc_output_dim']), dtype="float32")
                for images, indices in tqdm(database_dataloader, ncols=100):
                    descriptors = self.model(images.to(self.config['device']))
                    descriptors = descriptors.cpu().numpy()
                    database_descriptors[indices.numpy(), :] = descriptors

                # Get Query Image Descriptor
                query_descriptor = self.model(query_img.to(self.config['device']))
                query_descriptor = query_descriptor.cpu().numpy()

        faiss_index = faiss.IndexFlatL2(self.config['fc_output_dim'])
        faiss_index.add(database_descriptors)
        _, prediction = faiss_index.search(query_descriptor, 1)

        pred_img_path = database_ds.get_datapaths()[prediction]

        if not os.path.exists(pred_img_path):
            raise FileNotFoundError(f"Database Image File {pred_img_path} does not exist!")

        pred_img = Image.open(pred_img_path).convert('RGB')

        return pred_img, image





if __name__ == '__main__':
    test = CosPlace()
    pred_img, result_img = CosPlace.inference_by_image('')





