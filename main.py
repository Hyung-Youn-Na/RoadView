import datetime
import json
import os
import threading
import time

from detector.object_detection.yolov4.yolov4 import YOLOv4
from utils import Logging
from config import DEBUG

class EdgeModule:
    path = os.path.dirname(os.path.abspath(__file__))
    # event_model_class = {
    #     "assault": AssaultEvent,
    #     "falldown": FalldownEvent,
    #     "kidnapping": KidnappingEvent,
    #     "tailing": TailingEvent,
    #     "wanderer": WandererEvent,
    #     "obstacle": ObstacleEvent
    # }
    #TODO: 실험할 Object Detection들 목록 만들어서 넣기
    vr_models = {
        'cosplace': "import ~~"
    }
    od_models = {

    }
    td_models = {

    }
    tr_models = {

    }
    def __init__(self):
        ret = self.load_settings()
        ret = self.load_models()




    def load_settings(self):
        try:
            setting_path = os.path.join(self.path, "settings.json")
            with open(setting_path, 'r') as setting_file:
                dict_settings = json.load(setting_file)
                setting_file.close()
            streaming_url, streaming_type, fps, communication_info, od_option, event_option = self.__parse_settings(dict_settings)


            self.fps = fps
            self.communication_info = communication_info
            self.od_option = od_option
            self.event_option = event_option
            self.load_decoder()
            self.decoder_thread = None
            return True
        except :
            print(Logging.e("Cannot load setting.json"))
            exit(0)

    def __parse_settings(self, dict_settings):
        vp_option = dict_settings['model']['visual_place_recognition']
        od_option = dict_settings['model']['object_detection']
        td_option = dict_settings['model']['scenetext_detection']
        tr_option = dict_settings['model']['scenetext_recognition']

        #TODO: Config의 각 모델들 옵션 추가하고 옵션마다 Loggging 추가

        print(Logging.i("Settings INFO"))
        print(Logging.s("Object detection model INFO: "))
        print(Logging.s("\tModel name\t\t: {}".format(od_option['model_name'])))
        print(Logging.s("\tScore Threshold\t\t: {}".format(od_option['score_thresh'])))
        print(Logging.s("\tNMS Threshold\t\t: {}".format(od_option['nms_thresh'])))

        return vp_option, od_option, td_option, tr_option

    def load_models(self):
        od_model_name = self.od_option["model_name"]
        score_thresh = self.od_option["score_thresh"]
        nms_thresh = self.od_option["nms_thresh"]
        try :
            if od_model_name == "yolov4-416":
                od_model = YOLOv4(model=od_model_name, score_threshold=score_thresh, nms_threshold=nms_thresh)
            else:
                od_model = YOLOv4(model=od_model_name, score_threshold=score_thresh, nms_threshold=nms_thresh)
            self.od_model = od_model
            print(Logging.i("{} model is loaded".format(od_model_name)))
        except:
            print(Logging.e("Cannot load object detection model({})".format(od_model_name)))
            exit(0)

        vr_models =

        # TODO: 아래의 event model 비슷하게 각 모델들 실험 (모델 목록에서 이름으로 선택하는 식으로)
        # event_models = []
        # event_model_names = self.event_option
        # for event_model_name in event_model_names:
        #     try:
        #         event_model = self.event_model_class[event_model_name](debug=True)
        #         event_models.append(event_model)
        #         print(Logging.i("{} model is loaded".format(event_model.model_name)))
        #     except:
        #         print(Logging.e("Cannot load event detection model({})".format(event_model_name)))
        #         exit(0)
        #
        # self.event_models = event_models

        return True


    def run_detection(self):
        process_framecount = 0
        process_time = 0

        frame_number = 1
        self.decoder_thread = threading.Thread(target=self.run_decoder,)
        self.decoder_thread.start()

        """
        TODO: 아래 코드는 Frame를 디코딩하는 즉시 바로 검출, 이걸 이미지 폴더를 받아서 디텍션 하는걸로 바꿔야함
        """

        while True:
            if len(self.frame_buffer) > 0 :
                frame = self.frame_buffer.pop(0)
                start_time = time.time()

                frame_info = {"frame": frame, "frame_number": frame_number}
                results = self.od_model.inference_by_image(frame)

                dict_result = dict()
                dict_result["frame_number"] = frame_number
                dict_result["results"] = []
                dict_result["results"].append({"detection_result": results})

                for event_model in self.event_models:
                    event_model.inference(frame_info, dict_result)
                    event_model.merge_sequence(frame_info, 0)

                for event_model in self.event_models:
                    now = datetime.datetime.now()
                    if event_model.new_seq_flag == True:
                        event_model.new_seq_flag = False
                        self.communicator.send_event(event_model.model_name, now, "start", None)
                        print(Logging.i("Send start time of {} event sequence({})".format(event_model.model_name, now)))
                    if len(event_model.frameseq) > 0:
                        sequence = event_model.frameseq.pop()
                        message = sequence
                        message["duration"] = (sequence["end_frame"] - sequence["start_frame"])/self.fps
                        self.communicator.send_event(event_model.model_name, now, "end", message)
                        print(Logging.i("Send start time of {} event sequence({})".format(event_model.model_name, now)))

                frame_number += 1

                end_time = time.time()
                process_time += (end_time - start_time)
                process_framecount += 1




    def __del__(self):
        now = datetime.datetime.now()
        self.communicator.send_event(None, now, "disconnect", None)

if __name__ == '__main__':
    main = EdgeModule()
    main.run_detection()