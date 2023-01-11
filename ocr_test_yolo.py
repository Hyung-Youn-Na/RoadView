import os
import random
import datetime
from functools import cmp_to_key
from collections import OrderedDict

import pprint
import numpy

from PIL import Image, ImageDraw, ImageFont
from detector.object_detection.yolov7.yolov7 import TorchYOLOv7
from detector.scenetext_detection.craft.CRAFT_Infer import CRAFT_Infer
from recognizer.scenetext_recognizer.deeptext.DeepText import DeepText

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from utils import Logging, merge_filter, save_result


class ShopSignModule:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.area_threshold = 0.5
        self.result = dict()
        self.od_model, self.td_model, self.tr_model, self.lg_model = self.load_models()
        self.new_result = dict()




    def _parse_settings(self, dict_settings):
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return td_option, tr_option

    def load_models(self):
        od_model = TorchYOLOv7()
        td_model = CRAFT_Infer()
        tr_model = DeepText()

        # model = 'HyungYoun/autotrain-xlm-roberta-base-shopsign-refined-finetune-2112568275'
        model = 'HyungYoun/autotrain-xlm-no-i-finetune-2267372054'
        tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token='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')
        model = AutoModelForTokenClassification.from_pretrained(model, use_auth_token='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')

        lg_model = pipeline("ner", model=model, tokenizer=tokenizer)



        return od_model, td_model, tr_model, lg_model


    def inference_by_image_recognition_before(self, image):
        result = {
            'image_path': image
        }

        img = Image.open(image)

        print('TD model inference...')
        img_text_bboxes, img_td_confidences, img_char_bboxes, img_char_confidences = self.td_model.inference_by_image(img)

        img_text_bboxes = sorted(img_text_bboxes, key=cmp_to_key(merge_filter.cmp_text_bbox))

        result['img_text_bboxes'] = img_text_bboxes
        result['img_text_bboxes_confidences'] = img_td_confidences
        result['img_char_bboxes'] = img_char_bboxes
        result['img_char_bboxes_confidences'] = img_char_confidences
        print("OD model inference...")
        img_od_results = self.od_model.inference_image(img)

        img_group_texts = []
        img_group_text_confidences = []
        img_group_text_category = []
        img_text_top5 = []
        print('TR model inference...')
        img_basename = os.path.basename(image).split('.')[0] + '.txt'
        for text_bbox in img_text_bboxes:
            crop_img = img.crop(tuple(text_bbox))
            texts, tr_confidences, text_top5_list = self.tr_model.inference_by_image([crop_img])
            img_group_texts.extend(texts)
            img_group_text_confidences.extend(tr_confidences)
            lg_dic = self.lg_model(texts[0])
            if len(lg_dic) > 0:
                img_group_text_category.append(lg_dic[0]['entity'])
        result['img_group_texts'] = img_group_texts
        result['img_group_text_confidences'] = img_group_text_confidences
        result['img_group_text_categories'] = img_group_text_category
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

        # image.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))

    def parse_result(self):
        new_result = {
            'image_path': self.result['image_path'],
            'roi': []
        }
        image_basename = os.path.basename(self.result['image_path']).split('.')[0]
        # pure_ocr = open(f'/hdd/RoadView/pure_ocr/{image_basename}.txt', 'a', encoding='utf-8')
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
            roi_dic['score'] = od_result['label']['score']
            all_str = ''
            for text_bbox, text, text_confidence in zip(self.result['img_text_bboxes'], self.result['img_group_texts'], self.result['img_group_text_confidences']):
                text_dic = {}
                text_region_ratio = merge_filter.get_inner_text_region_ratio(od_position, text_bbox)
                # if text_confidence < 0.4:
                #     continue
                # pure_ocr.write(f'text\t0.9\t{text_bbox[0]}\t{text_bbox[1]}\t{text_bbox[2]}\t{text_bbox[3]}\t{text}\n')
                if text_region_ratio > 0.75:
                    text_dic['points'] = text_bbox
                    text_dic['texts'] = text
                    text_dic['confidence'] = text_confidence
                    text_group.append(text_dic)
                    all_str = all_str + ' ' + text
            roi_dic['words']= text_group
            lg_dic = self.lg_model(all_str)
            print(all_str)
            roi_dic['ganpan_dic'] = lg_dic
            entity_word_str = ''
            label_token_list = []
            for entity in lg_dic:
                entity_word_str += entity['word']
                label = entity['entity']
                word = entity['word'].replace('▁', '')
                score = entity['score']
                if label == 'B-GANPAN' and score < 0.75:
                    label = 'B-NOISE'
                label_list = [label] * len(word)
                label_token_list.extend(label_list)
            roi_dic['char_category_list'] = label_token_list
            new_result['roi'].append(roi_dic)
        # pure_ocr.close()
        self.new_result = new_result
        # pprint.pprint(new_result)



    def plot_new_result(self, final_result_path):
        image = Image.open(self.result['image_path']).convert('RGB')

        image_basename = os.path.basename(self.result['image_path']).split('.')[0]

        image_size = image.size

        image_draw = ImageDraw.Draw(image)

        fontpath = "/nfs_shared/STR_Data/graduate_project/utils/Gulim.ttf"
        font = ImageFont.truetype(fontpath, 50)
        font_small = ImageFont.truetype(fontpath, 30)

        roi_list = self.new_result['roi']
        for roi in roi_list:
            roi_points = roi['points']
            x1 = roi_points[0]
            y1 = roi_points[1]
            x2 = roi_points[2]
            y2 = roi_points[3]
            image_draw.rectangle(((x1, y1), (x2, y2)), outline='purple', width=5)

            words = roi['words']
            char_category_list = roi['char_category_list']
            char_start_idx = 0
            for word in words:
                text = word['texts']
                text_conf = word['confidence']
                char_category_slice = char_category_list[char_start_idx:char_start_idx + len(text)]
                ganpan_num = char_category_slice.count('B-GANPAN')
                tel_num = char_category_slice.count('B-TEL')
                noise_num = char_category_slice.count('B-NOISE')
                num_list = [ganpan_num, tel_num, noise_num]
                max_idx = num_list.index(max(num_list))
                color = 'blue'
                if max_idx == 0:
                    color = 'blue'
                elif max_idx == 1:
                    color = 'green'
                elif max_idx == 2:
                    color = 'red'
                text_bbox = word['points']

                # if text_conf < 0.4:
                #     continue

                image_draw.rectangle(((text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[3])), outline=color,
                                     width=3)
                text_position = (text_bbox[0], max(0, text_bbox[1] - 30))
                text_left, text_top, text_right, text_bottom = image_draw.textbbox(text_position, text, font=font_small)
                image_draw.rectangle((text_left - 5, text_top - 5, text_right + 5, text_bottom + 5), fill=color)
                image_draw.text(text_position, text, font=font_small, fill="black")
                char_start_idx += len(text)
        # image.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))

    def dump_object_result(self):

        image_basename = os.path.basename(self.result['image_path']).split('.')[0]

        f = 'q'


        roi_list = self.new_result['roi']
        object_telephone_pred = open(f'/hdd/RoadView/{f}/1/{image_basename}.txt', 'a', encoding='utf-8')
        telephone_text_pred = open(f'/hdd/RoadView/{f}/2/{image_basename}.txt', 'a', encoding='utf-8')
        ganpan_text_pred = open(f'/hdd/RoadView/{f}/3/{image_basename}.txt', 'a', encoding='utf-8')
        object_pred = open(f'/hdd/RoadView/{f}/4/{image_basename}.txt', 'a', encoding='utf-8')
        for roi in roi_list:
            roi_points = roi['points']
            x1 = roi_points[0]
            y1 = roi_points[1]
            x2 = roi_points[2]
            y2 = roi_points[3]

            words = roi['words']
            score = roi['score']
            char_category_list = roi['char_category_list']
            char_start_idx = 0
            ganpan_object_str = ''
            tel_object_str = ''
            for word in words:
                text = word['texts']
                text_points = word['points']
                text_x1, text_y1, text_x2, text_y2 = text_points
                text_conf = word['confidence']

                # if text_conf < 0.3:
                #     continue

                char_category_slice = char_category_list[char_start_idx:char_start_idx + len(text)]
                ganpan_num = char_category_slice.count('B-GANPAN')
                tel_num = char_category_slice.count('B-TEL')
                noise_num = char_category_slice.count('B-NOISE')
                num_list = [ganpan_num, tel_num, noise_num]
                max_idx = num_list.index(max(num_list))
                category = ''
                if max_idx == 0:
                    category = 'GANPAN'
                    ganpan_object_str = ganpan_object_str + text
                    ganpan_text_pred.write(f'text\t0.9\t{text_x1}\t{text_y1}\t{text_x2}\t{text_y2}\t{text}\n')
                elif max_idx == 1:
                    category = 'TEL'
                    tel_object_str = tel_object_str + text
                    telephone_text_pred.write(f'text\t0.9\t{text_x1}\t{text_y1}\t{text_x2}\t{text_y2}\t{text}\n')
                elif max_idx == 2:
                    category = 'NOISE'
                char_start_idx += len(text)
            if len(ganpan_object_str) == 0:
                ganpan_object_str = '[EMPTY]'
            if len(tel_object_str) != 0:
                object_telephone_pred.write(f'text\t{score}\t{x1}\t{y1}\t{x2}\t{y2}\t{tel_object_str}\n')
            object_pred.write(f'text\t{score}\t{x1}\t{y1}\t{x2}\t{y2}\t{ganpan_object_str}\n')

        ganpan_text_pred.close()
        telephone_text_pred.close()
        object_telephone_pred.close()
        object_pred.close()





if __name__ == '__main__':
    # result_path = '/nfs_shared/STR_Data/graduate_project/results_detector/'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    result_path = '/nfs_shared/STR_Data/graduate_project/results/'

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
    q = '/nfs_shared/STR_Data/graduate_project/data/test_img_yolo/'
    q_paths = os.listdir(q)
    # 2252
    for path in q_paths:
        print(path)
        main.inference_by_image_recognition_before(os.path.join(q, path))
        # main.plot_result(nowDate_result_path)
        main.parse_result()
        main.dump_object_result()
        # main.plot_new_result(od_result_path)
        # pprint.pprint(main.result)
