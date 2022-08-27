
import sys
import os
import numpy as np
import faiss
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import logging
import yaml

import multiprocessing
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from PIL import Image
torch.backends.cudnn.benchmark= True  # Provides a speedup

import recognizer.visualplace_recognizer.cosplace.test as test
import recognizer.visualplace_recognizer.cosplace.commons as commons
from recognizer.visualplace_recognizer.cosplace.model import network
from recognizer.visualplace_recognizer.cosplace.datasets.test_dataset import RawDataset, QueryDataset

class Cosplace:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='cosplace'):
        self.config = self._parse_config()
        commons.make_deterministic(self.config['seed'])
        self.results = dict()
        self.model_name = model

        self.model = self._load_model()
        self.database_descriptors, self.database_ds = self._set_descriptor()

    def _load_model(self):
        #### Model
        model = network.GeoLocalizationNet(self.config['backbone'], self.config['fc_output_dim'])

        logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

        if self.config['resume_model'] != None:
            logging.info(f"Loading model from {self.config['resume_model']}")
            model_state_dict = torch.load(self.config['resume_model'])
            model.load_state_dict(model_state_dict)
        else:
            logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                         "Evaluation will be computed using randomly initialized weights.")

        model = model.to(self.config['device'])

        return model


    def _parse_config(self):
        with open(os.path.join(self.path, 'config/config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        return config

    def _get_query_img_tensor(self, query_img):
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transformed_image = base_transform(query_img)

        return transformed_image.view((1,) + tuple(transformed_image.shape))


    def inference_by_image(self, image):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image File {image} does not exist!")


        query_img = Image.open(image).convert("RGB")

        with torch.no_grad():
                # Get Database Descriptor by Directly
            logging.debug("Extracting database descriptors for evaluation/testing")


            # Get Query Image Descriptor
            query_ds = QueryDataset(image)
            with torch.no_grad():
                # Get Database Descriptor by Directly
                logging.debug("Extracting database descriptors for evaluation/testing")
                database_subset_ds = Subset(query_ds, list(range(query_ds.queries_num)))
                database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=self.config['num_workers'],
                                                 batch_size=self.config['infer_batch_size'],
                                                 pin_memory=(self.config['device'] == "cuda"))
                query_descriptor = np.empty((len(query_ds), self.config['fc_output_dim']), dtype="float32")
                for images, indices in tqdm(database_dataloader, ncols=100):
                    descriptors = self.model(images.to(self.config['device']))
                    descriptors = descriptors.cpu().numpy()
                    query_descriptor[indices.numpy(), :] = descriptors


        faiss_index = faiss.IndexFlatL2(self.config['fc_output_dim'])
        faiss_index.add(self.database_descriptors)
        _, prediction = faiss_index.search(query_descriptor, 1)

        print(prediction)
        pred_img_path = self.database_ds.get_datapaths()[prediction[0][0]]

        if not os.path.exists(pred_img_path):
            raise FileNotFoundError(f"Database Image File {pred_img_path} does not exist!")

        pred_img = Image.open(pred_img_path).convert('RGB')

        return pred_img, pred_img_path, query_img, image

    # def inference_by_image(self, image):
    #     if not os.path.exists(image):
    #         raise FileNotFoundError(f"Image File {image} does not exist!")
    #
    #     query_img = Image.open(image).convert("RGB")
    #
    #     self.database_descriptors, self.database_ds = self._set_descriptor(image)
    #
    #     queries_descriptors = self.database_descriptors[self.database_ds.database_num:]
    #     database_descriptors = self.database_descriptors[:self.database_ds.database_num]
    #
    #     # Use a kNN to find predictions
    #     faiss_index = faiss.IndexFlatL2(self.config['fc_output_dim'])
    #     faiss_index.add(database_descriptors)
    #     del database_descriptors, self.database_descriptors
    #
    #     _, predictions = faiss_index.search(queries_descriptors, 1)
    #
    #     print(predictions)
    #     pred_img_path = self.database_ds.get_datapaths()[predictions[0][0]]
    #
    #
    #
    #     if not os.path.exists(pred_img_path):
    #         raise FileNotFoundError(f"Database Image File {pred_img_path} does not exist!")
    #
    #     pred_img = Image.open(pred_img_path).convert('RGB')
    #
    #     return pred_img, pred_img_path, query_img, image

    def _set_descriptor(self):
        eval_ds = RawDataset(self.config['dataset_path'])
        with torch.no_grad():
            # Get Database Descriptor by Directly
            logging.debug("Extracting database descriptors for evaluation/testing")
            database_subset_ds = Subset(eval_ds, list(range(eval_ds.dataset_num)))
            database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=self.config['num_workers'],
                                             batch_size=self.config['infer_batch_size'],
                                             pin_memory=(self.config['device'] == "cuda"))
            all_descriptors = np.empty((len(eval_ds), self.config['fc_output_dim']), dtype="float32")
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = self.model(images.to(self.config['device']))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors

        return all_descriptors, eval_ds


if __name__ == '__main__':
    test = Cosplace()
    pred_img, __, query_img, _ = test.inference_by_image('/nfs_shared/STR_Data/RoadView/data/queries_no_panorama/@1@1@21401300995_E_B.png')
    pred_img.save('pred_img.jpg')
    query_img.save('query_img.jpg')





