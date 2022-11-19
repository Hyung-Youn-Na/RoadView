import os
import numpy
import torch
import yaml
import cv2
import numpy as np

import sys

sys.path.insert(0, '/workspace/detector/object_detection/yolov7')

from detector.object_detection.yolov7.models.experimental import attempt_load
from detector.object_detection.yolov7.yolo_utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detector.object_detection.yolov7.yolo_utils.torch_utils import select_device
from detector.object_detection.yolov7.yolo_utils.datasets import letterbox

class TorchYOLOv7:
    model = None
    result = None
    conf = 0.0
    nms_thres = 0.0
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='yolov7'):
        """
        :param params: model parameters
        :param logger: model logger
        """
        params = self._parse_config()
        weights = params["model_path"]
        self.image_size = 640
        self.conf = params["conf"]
        self.nms_thres = params["nms"]
        # self.gpu = str(params["gpu"])
        self.is_batch = params["is_batch"]

        # batch option
        if self.is_batch:
            self.batch_size =  params["batch_size"]
        else:
            self.batch_size = 1

        # model Dataset
        self.class_names = ['signboard']

        # self.device = select_device(self.gpu, batch_size=self.batch_size)
        self.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.model.eval()
        self.model(torch.zeros(1, 3, self.image_size, self.image_size).to(self.device).type_as(next(self.model.parameters())))  # run once


    def inference_image(self, image):
        """
        :param image: input image(pil image)
        :return: dict format bounding box(x1, y1, x2, y2), scor, class, class index
            - format:
                [{"label": {"class": cls, "score": score, "class_idx": cls_idx},
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...]
        """

        numpy_image = np.array(image)

        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        origin_image_shape = opencv_image.copy().shape
        augment = False
        stride = int(self.model.stride.max())
        opencv_image = letterbox(opencv_image, self.image_size, stride=stride)[0]
        opencv_image = opencv_image[:, :, ::-1].transpose(2, 0, 1)
        opencv_image = numpy.ascontiguousarray(opencv_image)

        opencv_image = torch.from_numpy(opencv_image).to(self.device)
        opencv_image = opencv_image.float()  # uint8 to fp16/32
        opencv_image /= 255.0  # 0 - 255 to 0.0 - 1.0
        if opencv_image.ndimension() == 3:
            opencv_image = opencv_image.unsqueeze(0)
        pred = self.model(opencv_image, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.conf, iou_thres=self.nms_thres, labels=None,
                                   multi_label=True)
        results = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(opencv_image.shape[2:], det[:, :4], origin_image_shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if conf > self.conf:
                        score = float(conf)
                        x1 = float(xyxy[0])
                        y1 = float(xyxy[1])
                        x2 = float(xyxy[2])
                        y2 = float(xyxy[3])
                        str_class = self.class_names[int(cls)]
                        results.append({
                            'label': {
                                'class': str_class,
                                'score': score,
                                'class_idx': self.class_names.index(str_class)
                            },
                            'position': {
                                'xmin': x1,
                                'ymin': y1,
                                'xmax': x2,
                                'ymax': y2
                            }
                        })
        self.results = results

        return results


    def inference_image_batch(self, images):
        """
        :param image: input images(list in dict: [np array])
        :return: detection results(bounding box(x1, y1, x2, y2), score, class, class index) of each images
            - format:
                [[{"label": {"class": cls, "score": score, "class_idx": cls_idx},
                 "position": {"x": x, "y": y, "w": w, "h": h}}, ...], ...]
        """
        results = []
        tensor_images = []
        shapes = []
        stride = int(self.model.stride.max())
        for image in images:
            shapes.append([[image.shape[0], image.shape[1]], [[0.3333333333333333, 0.3333333333333333], [16.0, 12.0]]])
            image = letterbox(image, self.image_size, stride=stride)[0]
            image = image[:, :, ::-1].transpose(2, 0, 1)
            image = numpy.ascontiguousarray(image)
            tensor_images.append(torch.from_numpy(image))

        targets =  torch.zeros((0, 6))
        image = torch.stack(tensor_images, 0)
        image = image.to(self.device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(self.device)
        nb, _, height, width = image.shape  # batch size, channels, height, width

        with torch.no_grad():
            out, __ = self.model(image, augment=False)
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(self.device)  # to pixels
            labels = [targets[targets[:, 0] == i, 1:] for i in range(nb)]
            out = non_max_suppression(out, conf_thres=self.conf, iou_thres=self.nms_thres, labels=labels, multi_label=True)

        for si, det in enumerate(out):
            result = []
            if len(det):
                detn = det.clone()
                detn[:, :4] = scale_coords(image[si].shape[1:], detn[:, :4], tuple(shapes[si][0])).round()
                for *xyxy, conf, cls in reversed(detn.tolist()):
                    if conf > self.conf:
                        score = float(conf)
                        x1 = int(xyxy[0])
                        y1 = int(xyxy[1])
                        x2 = int(xyxy[2])
                        y2 = int(xyxy[3])
                        str_class = self.class_names[int(cls)]
                        result.append({
                            'label': {
                                    'class': str_class,
                                    'score': score,
                                    'class_idx': self.class_names.index(str_class)
                                },
                            'position': {
                                'xmin': x1,
                                'ymin': y1,
                                'xmax': x2,
                                'ymax': y2
                            }
                        })
            results.append(result)
        self.results = results

        return results

    def _parse_config(self):
        with open(os.path.join(self.path, 'config/config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        return config

