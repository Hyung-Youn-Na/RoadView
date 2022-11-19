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
        self.od_model, self.td_model, self.tr_model, self.lg_model = self.load_models()





    def _parse_settings(self, dict_settings):
        od_option = dict_settings['model']['object_detection']
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return od_option, td_option, tr_option

    def load_models(self):
        td_model = CRAFT_Infer()
        tr_model = DeepText()

        model = 'HyungYoun/xlm-roberta-base-char-shopsign'

        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForTokenClassification.from_pretrained(model)

        lg_model = pipeline("ner", model=model, tokenizer=tokenizer)



        return td_model, tr_model, lg_model

    def inference_by_image_recognition_before(self, image):
        result = {
            'image_path': image
        }

        img = Image.open(image)

        print('OD model inference...')
        img_object_bboxes = self.od_model.inference_by_image(img)

        print('TD model inference...')
        img_text_bboxes, img_td_confidences = self.td_model.inference_by_image(img)

        img_text_groups = merge_filter.get_text_groups(img_object_bboxes, img_text_bboxes)

        result['img_text_bboxes'] = img_text_bboxes
        result['img_object_bboxes'] = img_object_bboxes
        result['img_text_groups'] = img_text_groups

        img_group_texts = []

        img_group_text_confidences = []
        print('TR model inference...')
        for text_group in img_text_groups:
            pil_corp_imgs = []
            for text_bbox in text_group:
                crop_img = img.crop(tuple(text_bbox))
                pil_corp_imgs.append(crop_img)
            texts, tr_confidences = self.tr_model.inference_by_image(pil_corp_imgs)

            img_group_texts.append(texts)
            img_group_text_confidences.append(tr_confidences)


        result['img_group_texts'] = img_group_texts
        result['img_group_text_confidences'] = img_group_text_confidences

        for group_text in img_group_texts:
            text_str = ' '.join(group_text)
            print(text_str)

            ner_results = self.lg_model(text_str)
            for data in ner_results:
                entity = data['entity']
                word = data['word']
                print(entity, word)


        self.result = result

        self.result = self._parse_result()



    def plot_result(self, final_result_path):
        image = Image.open(self.result['image_path']).convert('RGB')

        image_basename = os.path.basename(self.result['image_path']).split('.')[0]

        image_size = image.size
        new_size = (image_size[0], 800 + image_size[1])
        image_border = Image.new("RGB", new_size)


        image_draw = ImageDraw.Draw(image)
        image_border_draw = ImageDraw.Draw(image_border)

        fontpath = "/nfs_shared/STR_Data/graduate_project/utils/Gulim.ttf"
        font = ImageFont.truetype(fontpath, 50)
        font_small = ImageFont.truetype(fontpath, 20)

        text = []

        for roi in self.result['roi']:
            od_bbox = roi['points']
            image_draw.rectangle(((od_bbox[0], od_bbox[1]), (od_bbox[2], od_bbox[3])), outline='red', width=3)

            for word in roi['words']:
                img_text_bbox = word['points']

                if word['text_conf'] >= 0.0:
                    text.append(word['text'])
                    image_draw.rectangle(((img_text_bbox[0], img_text_bbox[1]),
                                          (img_text_bbox[2], img_text_bbox[3])), outline='blue', width=3)

        pred_str = ', '.join(text) + '\n'

        image_border_draw.text((0, image_size[1]), text=pred_str, fill='white',font=font)

        image_border.paste(image, (0,0))
        image_border.save(os.path.join(final_result_path, f'{image_basename}_result.jpg'))

        # save_result.save_text_detection_result(self.result, td_result_path)
        #
        # save_result.save_object_detection_result(self.result, od_result_path)


        # pred_border.save('pred_image.jpg')
        # query_border.save('query_image.jpg')

    def _parse_result(self):
        new_result = OrderedDict()

        new_result['image_path'] = self.result['image_path']

        roi_list = []
        for i, object_bbox in enumerate(self.result['img_object_bboxes']):
            roi = OrderedDict()
            roi['points'] = list(object_bbox)

            word_list = []

            text_bbox_list = self.result['img_text_groups'][i]
            text_list = self.result['img_group_texts'][i]
            text_conf_list = self.result['img_group_text_confidences'][i]

            for j, (text_bbox, text, text_conf) in enumerate(zip(text_bbox_list, text_list, text_conf_list)):
                word = OrderedDict()
                word['points'] = text_bbox
                word['text'] = text
                word['text_conf'] = text_conf
                word_list.append(word)
            roi['words'] = word_list
            roi_list.append(roi)
        new_result['roi'] = roi_list
        pprint.pprint(new_result)
        return new_result



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
    for path in q_paths[:1]:
        print(path)
        if '.png' not in path and '.jpg' not in path:
            continue
        main.inference_by_image_recognition_before(os.path.join(q, path))
        main.plot_result(nowDate_result_path)
        # pprint.pprint(main.result)
