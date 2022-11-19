import os
import random
import datetime
from collections import OrderedDict

import pprint
import numpy

from PIL import Image, ImageDraw, ImageFont
from detector.object_detection.Faster_RCNN.FasterRCNN import FasterRCNN
from detector.scenetext_detection.craft.CRAFT_Infer import CRAFT_Infer
from recognizer.scenetext_recognizer.deeptext.DeepText import DeepText

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from utils import Logging, merge_filter, save_result


class EdgeModule:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.area_threshold = 0.5
        self.result = dict()
        self.td_model, self.tr_model, self.lg_model = self.load_models()





    def _parse_settings(self, dict_settings):
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return td_option, tr_option

    def load_models(self):
        td_model = CRAFT_Infer()
        tr_model = DeepText()

        model = 'HyungYoun/autotrain-xlm-roberta-base-shopsign-refined-finetune-2112568275'

        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token='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')
        model = AutoModelForTokenClassification.from_pretrained(model, use_auth_token='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')

        lg_model = pipeline("ner", model=model, tokenizer=tokenizer, device_map="auto")



        return td_model, tr_model, lg_model

    def inference_by_image_recognition_before(self, image):
        result = {
            'image_path': image
        }

        img = Image.open(image)

        print('TD model inference...')
        img_text_bboxes, img_td_confidences = self.td_model.inference_by_image(img)

        result['img_text_bboxes'] = img_text_bboxes

        img_group_texts = []
        img_group_text_confidences = []
        img_group_text_category = []
        print('TR model inference...')
        for text_bbox in img_text_bboxes:
            crop_img = img.crop(tuple(text_bbox))
            texts, tr_confidences = self.tr_model.inference_by_image([crop_img])
            img_group_texts.extend(texts)
            img_group_text_confidences.extend(tr_confidences)
            lg_dic = self.lg_model(texts[0])
            img_group_text_category.append(lg_dic[0]['entity'])
        result['img_group_texts'] = img_group_texts
        result['img_group_text_confidences'] = img_group_text_confidences
        result['img_group_text_categories'] = img_group_text_category

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
            image_draw.text(text_position, text, font=font_small, fill="black")

        image.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))



if __name__ == '__main__':
    # result_path = '/nfs_shared/STR_Data/graduate_project/results_detector/'
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

    main = EdgeModule()
    q = '/nfs_shared/STR_Data/RoadView/img/'
    q_paths = os.listdir(q)
    for path in q_paths:
        print(path)
        if '.png' not in path and '.jpg' not in path:
            continue
        main.inference_by_image_recognition_before(os.path.join(q, path))
        main.plot_result(nowDate_result_path)
        # pprint.pprint(main.result)
