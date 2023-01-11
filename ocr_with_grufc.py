import os
import random
import datetime
from collections import OrderedDict

import pprint
import numpy

from PIL import Image, ImageDraw, ImageFont
from detector.object_detection.yolov7.yolov7 import TorchYOLOv7
from detector.scenetext_detection.craft.CRAFT_Infer import CRAFT_Infer
from recognizer.scenetext_recognizer.deeptext.DeepText import DeepText

from language.custom_lm import Custom_lm

from utils import Logging, merge_filter, save_result


class ShopSignModule:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.area_threshold = 0.5
        self.result = dict()
        self.od_model, self.td_model, self.tr_model, self.lg_model = self.load_models()





    def _parse_settings(self, dict_settings):
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return td_option, tr_option

    def load_models(self):
        od_model = TorchYOLOv7()
        td_model = CRAFT_Infer()
        tr_model = DeepText()
        lg_model = Custom_lm().load_state_dict('./')



        return od_model, td_model, tr_model, lg_model

    def inference_by_image_recognition_before(self, image):
        result = {
            'image_path': image
        }

        img = Image.open(image)

        print('TD model inference...')
        img_text_bboxes, img_td_confidences = self.td_model.inference_by_image(img)
        result['img_text_bboxes'] = img_text_bboxes

        print("OD model inference...")
        img_od_results = self.od_model.inference_image(img)

        print(img_od_results)

        img_group_texts = []
        img_group_text_confidences = []
        img_group_text_category = []
        print('TR model inference...')
        for text_bbox in img_text_bboxes:
            crop_img = img.crop(tuple(text_bbox))
            texts, tr_confidences = self.tr_model.inference_by_image([crop_img])
            img_group_texts.extend(texts)
            img_group_text_confidences.extend(tr_confidences)
            # lg_dic = self.lg_model(texts[0])
            # img_group_text_category.append(lg_dic[0]['entity'])
        result['img_group_texts'] = img_group_texts
        result['img_group_text_confidences'] = img_group_text_confidences
        # result['img_group_text_categories'] = img_group_text_category
        result['img_object_results'] = img_od_results

        self.result = result


    def plot_result(self, final_result_path):
        image = Image.open(self.result['image_path']).convert('RGB')

        image_basename = os.path.basename(self.result['image_path']).split('.')[0]

        image_size = image.size
        new_size = (image_size[0], 800 + image_size[1])
        image_border = Image.new("RGB", new_size)

        image_draw = ImageDraw.Draw(image)

        fontpath = "/nfs_shared/STR_Data/graduate_project/utils/Gulim.ttf"
        font = ImageFont.truetype(fontpath, 50)
        font_small = ImageFont.truetype(fontpath, 20)

        text = []

        for text_bbox, text, text_category in \
                zip(self.result['img_text_bboxes'], self.result['img_group_texts'], self.result['img_group_text_categories']):
            if 'GANPAN' in text_category:
                color = 'blue'
            elif 'NOISE' in text_category:
                color = 'red'
            elif 'TEL' in text_category:
                color = 'green'
            image_draw.rectangle(((text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])), outline=color, width=3)
            text_position = (text_bbox[0], max(0, text_bbox[1] - 20))
            text_left, text_top, text_right, text_bottom = image_draw.textbbox(text_position, text, font=font_small)
            image_draw.rectangle((text_left - 5, text_top - 5, text_right + 5, text_bottom + 5), fill=color)
            image_draw.text(text_position, text, font=font_small, fill="white")

        for od_result in self.result['img_object_results']:
            od_position = od_result['position']
            x1 = od_position['xmin']
            y1 = od_position['ymin']
            x2 = od_position['xmax']
            y2 = od_position['ymax']
            image_draw.rectangle(((x1,y1), (x2,y2)), outline='orange', width=3)

        image.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))

    def parse_result(self):
        new_result = {
            'image_path': self.result['image_path'],
            'roi': []
        }
        for od_result in self.result['img_object_results']:
            roi_dic = {}
            text_group = []
            od_position = od_result['position']
            x1 = od_position['xmin']
            y1 = od_position['ymin']
            x2 = od_position['xmax']
            y2 = od_position['ymax']
            od_position = [x1, y1, x2, y2]
            roi_dic['points'] = od_position
            for text_bbox, text in zip(self.result['img_text_bboxes'], self.result['img_group_texts']):
                text_dic = {}
                text_region_ratio = merge_filter.get_inner_text_region_ratio(od_position, text_bbox)
                if text_region_ratio > 0.75:
                    text_dic['points'] = text_bbox
                    text_dic['texts'] = text
                    text_group.append(text_dic)
            roi_dic['words']= text_group

            new_result['roi'].append(roi_dic)

        pprint.pprint(new_result)






if __name__ == '__main__':
    # result_path = '/nfs_shared/STR_Data/graduate_project/results_detector/'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result_path = './'

    td_result_path = '/nfs_shared/STR_Data/graduate_project/SceneTextDetection/result/'
    od_result_path = '/nfs_shared/STR_Data/graduate_project/ObjectDetection/result/'
    nowDate = datetime.datetime.now()
    nowDate_str = nowDate.strftime("%Y-%m-%d-%H-%M-%S")
    nowDate_result_path = os.path.join(result_path, nowDate_str)

    if not os.path.exists(nowDate_result_path):
        os.mkdir(nowDate_result_path)

    td_result_path = os.path.join(td_result_path, nowDate_str)

    if not os.path.exists(td_result_path):
        os.mkdir(td_result_path)

    od_result_path = os.path.join(od_result_path, nowDate_str)

    if not os.path.exists(od_result_path):
        os.mkdir(od_result_path)

    main = ShopSignModule()
    q = '/nfs_shared/STR_Data/RoadView/img/'
    q_paths = os.listdir(q)
    for path in q_paths[:1]:
        print(path)
        if '.png' not in path and '.jpg' not in path:
            continue
        main.inference_by_image_recognition_before(os.path.join(q, path))
        main.plot_result(nowDate_result_path)
        main.parse_result()
        # pprint.pprint(main.result)
