
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
from PIL import Image
torch.backends.cudnn.benchmark= True  # Provides a speedup

import recognizer.visualplace_recognizer.cosplace.test as test
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

        self.model = self._load_model()

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
        if '.npy' in self.config['dataset_path']:
            # Get Database by file
            # TODO: try catch로 예외 처리 나중에 넣기
            database_descriptors = np.load(self.config['dataset_path'])
        else:
            database_ds = RawDataset(self.config['dataset_path'])
            self.model.eval()
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
                query_img_tensor = self._get_query_img_tensor(query_img)
                query_descriptor = self.model(query_img_tensor.to(self.config['device']))
                query_descriptor = query_descriptor.cpu().numpy()

        faiss_index = faiss.IndexFlatL2(self.config['fc_output_dim'])
        faiss_index.add(database_descriptors)
        _, prediction = faiss_index.search(query_descriptor, 1)

        print(prediction)
        pred_img_path = database_ds.get_datapaths()[prediction[0][0]]

        if not os.path.exists(pred_img_path):
            raise FileNotFoundError(f"Database Image File {pred_img_path} does not exist!")

        pred_img = Image.open(pred_img_path).convert('RGB')

        return pred_img, query_img





if __name__ == '__main__':
    test = CosPlace()
    pred_img, result_img = test.inference_by_image('/nfs_shared/STR_Data/RoadView/sample3/파노라마_2/2019/19631104043_E_B.png')
    pred_img.save('pred_img.jpg')
    result_img.save('result_img.jpg')





