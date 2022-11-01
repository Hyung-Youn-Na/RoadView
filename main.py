import os
import random
import datetime

from PIL import Image, ImageDraw, ImageFont
from detector.object_detection.Faster_RCNN.FasterRCNN import FasterRCNN
from detector.scenetext_detection.craft.CRAFT_Infer import CRAFT_Infer
from recognizer.visualplace_recognizer.cosplace.Cosplace import Cosplace
from recognizer.scenetext_recognizer.deeptext.DeepText import DeepText
from utils import Logging, merge_filter, save_result

class EdgeModule:
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        self.area_threshold = 0.5
        self.result = dict()
        self.vr_model, self.od_model, self.td_model, self.tr_model = self.load_models()





    def __parse_settings(self, dict_settings):
        vp_option = dict_settings['model']['visual_place_recognition']
        od_option = dict_settings['model']['object_detection']
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        return vp_option, od_option, td_option, tr_option

    def load_models(self):
        vr_model = Cosplace()
        od_model = FasterRCNN()
        td_model = CRAFT_Infer()
        tr_model = DeepText()


        return vr_model, od_model, td_model, tr_model


    def inference_by_image(self, image):
        pred_img, pred_img_path, query_img, query_img_path = self.vr_model.inference_by_image(image)

        result = {
            'query_img_path': query_img_path,
            'pred_img_path': pred_img_path
        }

        query_img_object_bboxes = self.od_model.inference_by_image(query_img)
        pred_img_object_bboxes = self.od_model.inference_by_image(pred_img)

        result['query_img_object_bboxes'] = query_img_object_bboxes
        result['pred_img_object_bboxes'] = pred_img_object_bboxes


        query_img_text_bboxes, query_img_td_confidences = self.td_model.inference_by_image(query_img)
        pred_img_text_bboxes, pred_img_td_confidences = self.td_model.inference_by_image(pred_img)

        result['query_img_text_bboxes'] = query_img_text_bboxes
        result['pred_img_text_bboxes'] = pred_img_text_bboxes


        query_img_bboxes = merge_filter.merge_and_filter_bboxes(query_img_object_bboxes, query_img_text_bboxes,
                                                        query_img_td_confidences)
        pred_img_bboxes = merge_filter.merge_and_filter_bboxes(pred_img_object_bboxes, pred_img_text_bboxes,
                                                        pred_img_td_confidences)


        result['query_img_bboxes'] = query_img_bboxes
        result['pred_img_bboxes'] = pred_img_bboxes

        query_pil_corp_imgs = []
        pred_pil_crop_imgs = []

        for query_img_bbox, pred_img_bbox in zip(query_img_bboxes, pred_img_bboxes):
            query_crop = query_img.crop(tuple(query_img_bbox))
            pred_crop = pred_img.crop(tuple(pred_img_bbox))

            query_pil_corp_imgs.append(query_crop)
            pred_pil_crop_imgs.append(pred_crop)
        query_texts, query_tr_confidences = self.tr_model.inference_by_image(query_pil_corp_imgs)
        pred_texts, pred_tr_confidences = self.tr_model.inference_by_image(pred_pil_crop_imgs)

        result['query_texts'] = query_texts
        result['pred_texts'] = pred_texts
        result['query_tr_confidences'] = query_tr_confidences
        result['pred_tr_confidences'] = pred_tr_confidences
        self.result = result

    def inference_by_image_recognition_before(self, image):
        print("VR model inference...")
        pred_img, pred_img_path, query_img, query_img_path = self.vr_model.inference_by_image(image)

        result = {
            'query_img_path': query_img_path,
            'pred_img_path': pred_img_path
        }

        print("OD model inference...")
        query_img_object_bboxes = self.od_model.inference_by_image(query_img)
        pred_img_object_bboxes = self.od_model.inference_by_image(pred_img)

        print("TD model inference...")
        query_img_text_bboxes, query_img_td_confidences = self.td_model.inference_by_image(query_img)
        pred_img_text_bboxes, pred_img_td_confidences = self.td_model.inference_by_image(pred_img)

        query_img_text_groups = merge_filter.get_text_groups(query_img_object_bboxes,
                                                                                 query_img_text_bboxes)
        pred_img_text_groups = merge_filter.get_text_groups(pred_img_object_bboxes,
                                                      pred_img_text_bboxes)


        result['query_img_text_groups'] = query_img_text_groups
        result['pred_img_text_groups'] = pred_img_text_groups

        query_group_texts = []
        pred_group_texts = []

        query_group_text_confidences = []
        pred_group_text_confidences = []

        print("TR model inference...")
        for query_text_group, pred_text_group in zip(query_img_text_groups, pred_img_text_groups):
            query_pil_corp_imgs = []
            pred_pil_crop_imgs = []
            for query_text_bbox, pred_text_bbox in zip(query_text_group, pred_text_group):
                query_crop = query_img.crop(tuple(query_text_bbox))
                pred_crop = pred_img.crop(tuple(pred_text_bbox))
                query_pil_corp_imgs.append(query_crop)
                pred_pil_crop_imgs.append(pred_crop)
            print(query_text_group, pred_text_group)
            query_texts, query_tr_confidences = self.tr_model.inference_by_image(query_pil_corp_imgs)
            pred_texts, pred_tr_confidences = self.tr_model.inference_by_image(pred_pil_crop_imgs)

            query_group_texts.append(query_texts)
            query_group_text_confidences.append(query_tr_confidences)

            pred_group_texts.append(pred_texts)
            pred_group_text_confidences.append(pred_tr_confidences)

        result['query_group_texts'] = query_group_texts
        result['pred_group_texts'] = pred_group_texts
        result['query_group_text_confidences'] = query_group_text_confidences
        result['pred_group_text_confidences'] = pred_group_text_confidences




        self.result = result




    def plot_result(self, final_result_path, td_result_path, od_result_path):
        pred_image = Image.open(self.result['pred_img_path']).convert('RGB')
        query_image = Image.open(self.result['query_img_path']).convert('RGB')

        pred_image_basename = os.path.basename(self.result['pred_img_path']).split('.')[0]
        query_image_basename = os.path.basename(self.result['query_img_path']).split('.')[0]

        old_size = pred_image.size
        new_size = (10+ 2 * old_size[0], 800 + old_size[1])
        pred_border = Image.new("RGB", new_size)
        new_size = (old_size[0], 800 + old_size[1])
        query_border = Image.new("RGB", new_size)

        pred_draw = ImageDraw.Draw(pred_image)
        query_draw = ImageDraw.Draw(query_image)

        pred_border_draw = ImageDraw.Draw(pred_border)
        query_border_draw = ImageDraw.Draw(query_border)

        fontpath = "/nfs_shared/STR_Data/graduate_project/utils/Gulim.ttf"
        font = ImageFont.truetype(fontpath, 50)
        font_small = ImageFont.truetype(fontpath, 20)

        pred_str = []
        query_str = []

        for query_img_bbox, pred_img_bbox, query_text, pred_text, query_text_conf, pred_text_conf\
                in zip(self.result['query_img_bboxes'],self.result['pred_img_bboxes'],
                       self.result['query_texts'], self.result['pred_texts'],
                       self.result['query_tr_confidences'], self.result['pred_tr_confidences']):


            color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])

            if pred_text_conf >= 0.0:
                pred_str.append(pred_text)
                pred_draw.rectangle(((pred_img_bbox[0], pred_img_bbox[1]),
                                     (pred_img_bbox[2], pred_img_bbox[3])), outline='red', width=3)
                pred_draw.text((pred_img_bbox[0], pred_img_bbox[1]), text=pred_text, fill='yellow',
                               font=font_small)

            if query_text_conf >= 0.0:
                query_str.append(query_text)
                query_draw.rectangle(((query_img_bbox[0], query_img_bbox[1]),
                                    (query_img_bbox[2], query_img_bbox[3])), outline='red', width=3)
                query_draw.text((query_img_bbox[0], query_img_bbox[1]), text=query_text, fill='yellow',
                                font=font_small)

        pred_str = ', '.join(pred_str) + '\n연도: 2019'
        query_str = ', '.join(query_str) + '\n연도: 2021'

        pred_border_draw.text((0, old_size[1]), text=pred_str, fill='white',font=font)
        query_border_draw.text((0, old_size[1]), text=query_str, fill='white',font=font)

        pred_border.paste(pred_image, (0,0))
        query_border.paste(query_image,(0,0))

        pred_border.paste(query_border,(10 + old_size[0], 0))

        pred_border.save(os.path.join(final_result_path, f'{query_image_basename}_{pred_image_basename}.jpg'))

        save_result.save_text_detection_result(self.result, td_result_path)

        save_result.save_object_detection_result(self.result, od_result_path)


        # pred_border.save('pred_image.jpg')
        # query_border.save('query_image.jpg')


if __name__ == '__main__':
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

    main = EdgeModule()
    q = '/nfs_shared/STR_Data/RoadView/data/queries_no_panorama/'
    q_paths = os.listdir(q)
    for path in q_paths:
        print(path)
        if '.png' not in path and '.jpg' not in path:
            continue
        main.inference_by_image_recognition_before(os.path.join(q, path))
        # main.plot_result(nowDate_result_path, td_result_path=td_result_path, od_result_path=od_result_path)
        print(main.result)