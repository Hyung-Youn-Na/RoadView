import argparse
import os
import random
import torch
import yaml

from PIL import ImageDraw,Image
from torchvision.transforms import transforms
from detector.object_detection.Faster_RCNN.dataset.base import Base as DatasetBase
from detector.object_detection.Faster_RCNN.backbone.base import Base as BackboneBase
from detector.object_detection.Faster_RCNN.bbox import BBox
from detector.object_detection.Faster_RCNN.model import Model
from detector.object_detection.Faster_RCNN.roi.pooler import Pooler
from detector.object_detection.Faster_RCNN.config.eval_config import EvalConfig as Config

class FasterRCNN:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model='Faster-RCNN'):
        self.config = self._parse_config()
        self.results = dict()
        self.model_name = model
        self.dataset = DatasetBase.from_name('voc2007-ganpan')
        self.model = self._load_model()

    def _load_model(self):

        backbone = BackboneBase.from_name(self.config['backbone'])(pretrained=False)
        model = Model(backbone, self.dataset.num_classes(), pooler_mode=self.config['pooler_mode'],
                      anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                      rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()

        model.load(self.config['checkpoint'])



        return model


    def inference_by_image(self, pil_image):

        with torch.no_grad():
            # image = transforms.Image.open(path_to_input_image)
            image_tensor, scale = self.dataset.preprocess(pil_image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)

            print(image_tensor.shape)

            detection_bboxes, detection_classes, detection_probs, _ = \
                self.model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
            detection_bboxes /= scale

            kept_indices = detection_probs > self.config['probability_threshold']
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]

            draw = ImageDraw.Draw(pil_image)

            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                category = self.dataset.LABEL_TO_CATEGORY_DICT[cls]

                draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

            pil_image.save('sample.png')
            print(f'Output image is saved to sample.png')
        """
        TODO
        :param image:
        :return:
        """

    def _parse_config(self):
        with open(os.path.join(self.path, 'infer_config/config.yml'), 'r') as f:
            config = yaml.safe_load(f)

        return config

#
# def _infer(path_to_input_image: str, path_to_output_image: str, path_to_checkpoint: str, dataset_name: str, backbone_name: str, prob_thresh: float):
#     dataset_class = DatasetBase.from_name(dataset_name)
#     backbone = BackboneBase.from_name(backbone_name)(pretrained=False)
#     model = Model(backbone, dataset_class.num_classes(), pooler_mode=Config.POOLER_MODE,
#                   anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
#                   rpn_pre_nms_top_n=Config.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=Config.RPN_POST_NMS_TOP_N).cuda()
#     model.load(path_to_checkpoint)
#
#     with torch.no_grad():
#         image = transforms.Image.open(path_to_input_image)
#         image_tensor, scale = dataset_class.preprocess(image, Config.IMAGE_MIN_SIDE, Config.IMAGE_MAX_SIDE)
#
#         detection_bboxes, detection_classes, detection_probs, _ = \
#             model.eval().forward(image_tensor.unsqueeze(dim=0).cuda())
#         detection_bboxes /= scale
#
#         kept_indices = detection_probs > prob_thresh
#         detection_bboxes = detection_bboxes[kept_indices]
#         detection_classes = detection_classes[kept_indices]
#         detection_probs = detection_probs[kept_indices]
#
#         draw = ImageDraw.Draw(image)
#
#         for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
#             color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
#             bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
#             category = dataset_class.LABEL_TO_CATEGORY_DICT[cls]
#
#             draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
#             draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)
#
#         image.save(path_to_output_image)
#         print(f'Output image is saved to {path_to_output_image}')
#
#
# if __name__ == '__main__':
#     def main():
#         parser = argparse.ArgumentParser()
#         parser.add_argument('-s', '--dataset', type=str, choices=DatasetBase.OPTIONS, required=True, help='name of dataset')
#         parser.add_argument('-b', '--backbone', type=str, choices=BackboneBase.OPTIONS, required=True, help='name of backbone model')
#         parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint')
#         parser.add_argument('-p', '--probability_threshold', type=float, default=0.6, help='threshold of detection probability')
#         parser.add_argument('--image_min_side', type=float, help='default: {:g}'.format(Config.IMAGE_MIN_SIDE))
#         parser.add_argument('--image_max_side', type=float, help='default: {:g}'.format(Config.IMAGE_MAX_SIDE))
#         parser.add_argument('--anchor_ratios', type=str, help='default: "{!s}"'.format(Config.ANCHOR_RATIOS))
#         parser.add_argument('--anchor_sizes', type=str, help='default: "{!s}"'.format(Config.ANCHOR_SIZES))
#         parser.add_argument('--pooler_mode', type=str, choices=Pooler.OPTIONS, help='default: {.value:s}'.format(Config.POOLER_MODE))
#         parser.add_argument('--rpn_pre_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_PRE_NMS_TOP_N))
#         parser.add_argument('--rpn_post_nms_top_n', type=int, help='default: {:d}'.format(Config.RPN_POST_NMS_TOP_N))
#         parser.add_argument('input', type=str, help='path to input image')
#         parser.add_argument('output', type=str, help='path to output result image')
#         args = parser.parse_args()
#
#         path_to_input_image = args.input
#         path_to_output_image = args.output
#         dataset_name = args.dataset
#         backbone_name = args.backbone
#         path_to_checkpoint = args.checkpoint
#         prob_thresh = args.probability_threshold
#
#         os.makedirs(os.path.join(os.path.curdir, os.path.dirname(path_to_output_image)), exist_ok=True)
#
#         Config.setup(image_min_side=args.image_min_side, image_max_side=args.image_max_side,
#                      anchor_ratios=args.anchor_ratios, anchor_sizes=args.anchor_sizes, pooler_mode=args.pooler_mode,
#                      rpn_pre_nms_top_n=args.rpn_pre_nms_top_n, rpn_post_nms_top_n=args.rpn_post_nms_top_n)
#
#         print('Arguments:')
#         for k, v in vars(args).items():
#             print(f'\t{k} = {v}')
#         print(Config.describe())
#
#         _infer(path_to_input_image, path_to_output_image, path_to_checkpoint, dataset_name, backbone_name, prob_thresh)
#
#     main()

if __name__ == '__main__':
    test = FasterRCNN()
    # img = Image.open('/nfs_shared/STR_Data/RoadView/sample3/파노라마_2/2019/19631104043_E_B.png').convert('RGB')
    img = transforms.Image.open('/nfs_shared/STR_Data/RoadView/sample3/파노라마_2/2019/19631104043_E_B.png')
    test.inference_by_image(img)