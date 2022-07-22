import datetime
import time

import pycuda.autoinit
import argparse
import json
import os
import cv2
import csv

from decoder.CvDecoder import CvDecoder
from detector.event.assault.main import AssaultEvent
from detector.event.falldown.main import FalldownEvent
from detector.event.kidnapping.main import KidnappingEvent
from detector.event.obstacle.main import ObstacleEvent
from detector.event.tailing.main import TailingEvent
from detector.event.wanderer.main import WandererEvent
from detector.object_detection.yolov4.yolov4 import YOLOv4
from detector.tracker.byte_tracker.BYTETracker import BYTETracker
from detector.tracker.sort.Sort import Sort

from utils.time_util import convert_framenumber2timestamp
from utils import Logging
from utils.yolo_classes import get_cls_dict
from utils.Visualize import BBoxVisualization


def split_model_names(model_names):
    return model_names.split(",")


def load_video(video_path, extract_fps):
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(capture.get(cv2.CAP_PROP_FPS)))

    print(Logging.i("Extract information"))
    print(Logging.s("video path: {}".format(video_path)))
    print(Logging.s("video fps: {}".format(fps)))
    print(Logging.s("video frame_count: {}".format(frame_count)))
    print(Logging.s("extract fps: {}".format(extract_fps)))
    print(Logging.s("extract frame number: {}".format(int(frame_count/(fps/extract_fps)))))

    return capture, frame_count, fps


def load_label(json_path, frame_count):
    cam_id_list = []
    if json_path is None:
        return cam_id_list
    if os.path.exists(json_path):
        with open(json_path) as json_file:
            metadata = json.load(json_file)
            events = metadata["event"]
            cam_id_list = ["none" for i in range(frame_count)]
            for i, event in enumerate(events):
                start_frame = int(event["start_frame"])
                end_frame = int(event["end_frame"])
                cam_id = event["cam_id"]
                for e in range(start_frame - 1, end_frame):
                    cam_id_list[e] = cam_id
            json_file.close()
        return cam_id_list
    else :
        return cam_id_list


def load_models(od_model_name="yolov4-416", tracker_params=None, score_threshold=0.5, nms_threshold=0.3, event_model_names="all"):
    if od_model_name == "yolov4-416":
        od_model = YOLOv4(model=od_model_name, score_threshold=score_threshold, nms_threshold=nms_threshold)
    else :
        od_model = YOLOv4(model=od_model_name, score_threshold=score_threshold, nms_threshold=nms_threshold)

    print(Logging.i("YOLO detector information"))
    print(Logging.s("model name: {}".format(od_model_name)))
    print(Logging.s("score threshold: {}".format(score_threshold)))
    print(Logging.s("nms threshold: {}".format(nms_threshold)))

    trackers = []
    for tracker_param in tracker_params:
        tracker = tracker_param["tracker_class"](tracker_param)
        trackers.append(tracker)

    event_detectors = []
    print(Logging.i("{} model is loaded".format("Object detection(yolov4)")))

    splited_model_names = split_model_names(event_model_names)

    if event_model_names == "all" or "assault" in splited_model_names:
        assault_detection_model = AssaultEvent()
        event_detectors.append(assault_detection_model)
        print(Logging.i("{} model is loaded".format(assault_detection_model.model_name)))

    if event_model_names == "all" or "falldown" in splited_model_names:
        falldown_detection_model = FalldownEvent()
        event_detectors.append(falldown_detection_model)
        print(Logging.i("{} model is loaded".format(falldown_detection_model.model_name)))

    if event_model_names == "all" or "obstacle" in splited_model_names:
        obstacle_detection_model = ObstacleEvent()
        event_detectors.append(obstacle_detection_model)
        print(Logging.i("{} model is loaded".format(obstacle_detection_model.model_name)))

    if event_model_names == "all" or "kidnapping" in splited_model_names:
        kidnapping_detection_model = KidnappingEvent()
        event_detectors.append(kidnapping_detection_model)
        print(Logging.i("{} model is loaded".format(kidnapping_detection_model.model_name)))

    if event_model_names == "all" or "tailing" in splited_model_names:
        tailing_detection_model = TailingEvent()
        event_detectors.append(tailing_detection_model)
        print(Logging.i("{} model is loaded".format(tailing_detection_model.model_name)))

    if event_model_names == "all" or "wanderer" in splited_model_names:
        wanderer_detection_model = WandererEvent()
        event_detectors.append(wanderer_detection_model)
        print(Logging.i("{} model is loaded".format(wanderer_detection_model.model_name)))

    return od_model, trackers, event_detectors


def make_result_dir(result_dir, video_name, save_frame_result):
    frame_dir = os.path.join(result_dir, video_name.split(".mp4")[0], "frame")
    fram_bbox_dir = os.path.join(result_dir, video_name.split(".mp4")[0], "fram_bbox")
    json_dir = os.path.join(result_dir, video_name.split(".mp4")[0], "json")
    event_dir = os.path.join(result_dir)

    if save_frame_result:
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if not os.path.exists(fram_bbox_dir):
            os.makedirs(fram_bbox_dir)
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)

    return frame_dir, fram_bbox_dir, json_dir, event_dir


def define_object_result(frame_dir, frame_name, video_path, frame_number, extract_fps, fps):
    dict_result = dict()
    dict_result["image_path"] = os.path.join(frame_dir, frame_name)
    dict_result["cam_address"] = video_path
    dict_result["module"] = od_model_name
    dict_result["frame_number"] = int(frame_number / extract_fps * fps)
    dict_result["timestamp"] = str(convert_framenumber2timestamp(frame_number / extract_fps * fps, fps))
    dict_result["results"] = []
    return dict_result


def define_event_result(video_path, frame_number, extract_fps, fps):
    event_result = dict()
    event_result["cam_address"] = video_path
    event_result["frame_number"] = int(frame_number / extract_fps * fps)
    event_result["timestamp"] = str(convert_framenumber2timestamp(frame_number / extract_fps * fps, fps))
    event_result["event_result"] = dict()
    return event_result


def load_viderwriter():
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(bbox_video_path, fourcc, 20, (640, 360))
    return video_writer


def extract_event_results(event_model_names, event_dir, video_name, event_detectors, event_results):
    event_csv_file_path = os.path.join(event_dir, "..", video_name.split(".mp4")[0] + ".csv")
    with open(event_csv_file_path, "w") as event_file:
        csv_writer = csv.writer(event_file)
        if type(event_model_names) == list:
            event_names = event_model_names
        elif event_model_names == "all":
            event_names = ["assault", "falldown", "obstacle", "kidnapping", "tailing", "wanderer"]
        else :
            event_names = [event_model_names]
        name = ["", ""]
        for i, event_detector in enumerate(event_detectors):
            name.append(event_detector.model_name)

        csv_writer.writerow(name)

        for event_result in event_results:
            row = ["{:>10}".format(event_result["frame_number"]), str(convert_framenumber2timestamp(event_result["frame_number"], 30))]
            for i, event_name in enumerate(event_names):
                if event_result["event_result"][event_name]:
                    row.append(i+1)
                else:
                    row.append("")
            csv_writer.writerow(row)
    print(Logging.i("Event result file is successfully extracted.(path: {})".format(event_csv_file_path)))


def extract_sequence_results(event_dir, video_name, sequence_results):
    # 빈클래스 추가 필요
    sequence_json_file_path = os.path.join(event_dir, video_name.replace(".mp4", ".json"))
    with open(sequence_json_file_path, "w") as sequence_file:
        json.dump(sequence_results, sequence_file, indent='\t')

    print(Logging.i("Sequence result file is successfully extracted.(path: {})".format(sequence_json_file_path)))
    return sequence_json_file_path


def draw_event(bbox_video_path, sequence_json_path, extract_fps, events):
    print(Logging.i("Generate result(video and json)"))
    date = datetime.datetime.now().strftime("%Y%m%d")[2:]

    origin_video_path = bbox_video_path
    event_video_path = bbox_video_path.replace("_bbox.avi", "") + "_edge_" + str(extract_fps) + "fps_" + date + ".mp4"
    json_file_path = bbox_video_path.replace("_bbox.avi", "") + "_edge_" + str(extract_fps) + "fps_" + date + ".json"
    resolution = (640, 360)

    cls_dict = get_cls_dict(15)
    bbox_visualization = BBoxVisualization(cls_dict)

    with open(sequence_json_path, 'r') as json_file:
        result = json.load(json_file)

    results = []

    video_capture = cv2.VideoCapture(origin_video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    video_writer = cv2.VideoWriter(event_video_path, fourcc, 20, resolution)

    for i in range(frame_count + 1):
        results.append({'assault': False, 'falldown': False, 'kidnapping': False, 'tailing': False, 'wanderer': False})

    for event in events:
        sequences = result[event]
        for sequence in sequences:
            start_frame = int(sequence['start_frame'] / 30 * 20)
            end_frame = int(sequence['end_frame'] / 30 * 20)
            for i in range(start_frame, end_frame + 1):
                results[i][event] = True

    frame_number = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        else:
            frame_event = bbox_visualization.put_text(frame, results[frame_number], events)
            video_writer.write(frame_event)
            print(Logging.ir("frame number: {:>6}/{} - {}".format(frame_number, frame_count, results[frame_number])), end="")
            frame_number += 1
    print()

    video_capture.release()
    video_writer.release()
    os.rename(sequence_json_path, json_file_path)
    print(Logging.i("Sequence result video is successfully generated(path: {})".format(event_video_path)))
    print(Logging.i("Sequence json file is successfully generated(path: {})".format(json_file_path)))


def generate_filter_numbers(extract_fps, fps):
    filter_frame_numbers = []
    if extract_fps is not 0 and fps is not 0:
        tmp_nums = []
        for i in range(1, fps + 1):
            filter_num = int(i * (extract_fps / fps))
            if filter_num != 0 and filter_num not in tmp_nums:
                tmp_nums.append(filter_num)
                filter_frame_numbers.append(i)
        return filter_frame_numbers
    else:
        return filter_frame_numbers


def run_detection(video_info, cam_ids, od_model, trackers, score_threshold, event_detectors, frame_dir, json_dir, bbox_video_path, save_frame_result, process_time):
    print(Logging.i("Processing..."))
    # Parameters
    fps = video_info["fps"]
    frame_count = video_info["frame_count"]
    extract_fps = video_info["extract_fps"]
    cam_id = video_info["cam_id"]
    filter_frame_numbers = generate_filter_numbers(extract_fps, fps)
    event_results = []
    cls_dict = get_cls_dict(15)
    bbox_visualization = BBoxVisualization(cls_dict)
    frame_number = 0
    filter_frame_number = 0

    # Load decoder
    decoder = CvDecoder(video_info["video_path"])
    decoder.load()
    # load video writer
    video_writer = load_viderwriter()

    sequence_result = dict()
    end_flag = 0
    total_object_process_time = 0
    max_object_process_time = 0
    min_object_process_time = 10
    total_tracker_process_times = [float(0) for tracker in trackers if tracker is not None]
    max_tracker_process_times = [float(0) for tracker in trackers if tracker is not None]
    min_tracker_process_times = [float(10) for tracker in trackers if tracker is not None]
    total_event_process_times = [float(0) for event_detector in event_detectors if event_detector is not None]
    max_event_process_times = [float(0) for event_detector in event_detectors if event_detector is not None]
    min_event_process_times = [float(10) for event_detector in event_detectors if event_detector is not None]

    total_process_start_time = time.time()
    while True:
        ret, frame = decoder.read()

        if ret == False:
            end_flag =1
            for event_detector in event_detectors:
                sequence_result[event_detector.model_name] = event_detector.merge_sequence(frame_info, end_flag)
            break

        frame_number += 1
        filter_frame_number += 1

        if filter_frame_number in filter_frame_numbers and frame is not None:
            if filter_frame_number == 30:
                filter_frame_number = 0
            frame_name = "{0:06d}.jpg".format(frame_number)
            if len(cam_ids) == 0:
                cam_id = "none"
            else:
                cam_id = cam_ids[frame_number-1]
            frame_info = {"frame": frame, "frame_number": frame_number, "cam_id": cam_id}
            start_time = time.time()
            od_result = od_model.inference_by_image(frame)
            end_time = time.time()
            if process_time:
                total_object_process_time += (end_time - start_time)
                if max_object_process_time < (end_time - start_time):
                    max_object_process_time = (end_time - start_time)
                if min_object_process_time > (end_time - start_time):
                    min_object_process_time = (end_time - start_time)

            # Object Detection
            object_result = define_object_result(frame_dir, frame_name, video_path, frame_number, extract_fps, fps)
            object_result["results"].append({"detection_result": od_result})
            tracking_results = dict()

            # Tracking(Byte Tracker, Sort)
            for i, tracker in enumerate(trackers):
                start_time = time.time()
                tracking_result = tracker.update(object_result.copy())
                tracking_results[tracker.tracker_name] = tracking_result
                end_time = time.time()
                if process_time:
                    total_tracker_process_times[i] += (end_time - start_time)
                    if max_tracker_process_times[i] < (end_time - start_time):
                        max_tracker_process_times[i] = (end_time - start_time)
                    if min_tracker_process_times[i] > (end_time - start_time):
                        min_tracker_process_times[i] = (end_time - start_time)

            # Draw object detection results
            frame_bbox = bbox_visualization.draw_bboxes(frame, od_result)
            video_writer.write(frame_bbox)

            # Event Detection
            event_process_times = []
            event_result = define_event_result(video_path, frame_number, extract_fps, fps)
            for i, event_detector in enumerate(event_detectors):
                if event_detector.tracker_name == "byte_tracker":
                    tracking_result = tracking_results["byte_tracker"]
                elif event_detector.tracker_name == "sort":
                    tracking_result = tracking_results["sort"]
                else:
                    tracking_result = None

                start_time = time.time()
                event_result["event_result"][event_detector.model_name] = event_detector.inference(frame_info, object_result, tracking_result, score_threshold[event_detector.model_name])
                sequence_result[event_detector.model_name] = event_detector.merge_sequence(frame_info, end_flag)
                end_time = time.time()

                if process_time:
                    event_process_times.append(end_time - start_time)
                    total_event_process_times[i] += (end_time - start_time)
                    if max_event_process_times[i] < (end_time - start_time):
                        max_event_process_times[i] = (end_time - start_time)
                    if min_event_process_times[i] > (end_time - start_time):
                        min_event_process_times[i] = (end_time - start_time)
            event_results.append(event_result)

            # Save frame results
            if save_frame_result:
                json_result_file = open(os.path.join(json_dir, frame_name.split(".jpg")[0] + ".json"), "w")
                json.dump(object_result, json_result_file, indent=4)
                json_result_file.close()

            # Print log
            print(Logging.ir("frame number: {:>6}/{}\t/ timestamp: {:>6}"
                .format(
                    frame_number,
                    frame_count,
                    str(convert_framenumber2timestamp(frame_number, fps)))
                ), end='')
            if process_time:
                print("\t/ process time: ", end="")
                for i, event_detector in enumerate(event_detectors):
                    print(" {0}: {1:03f}\t/".format(event_detector.model_name, event_process_times[i]), end="")
                print()
    total_process_end_time = time.time()

    video_writer.release()
    if process_time:
        print(Logging.i("Processing Time:"))
        print(Logging.s("\tObject Detection(yolov4-416) - average: {0:03f}\t/ max: {1:03f}\t / min: {2:03f}".format(
            total_object_process_time/frame_count,
            max_object_process_time,
            min_object_process_time
        )))
        print(Logging.s("\tTracker"))
        for i, tracker in enumerate(trackers):
            print(Logging.s("\t\t{0} - average: {1:03f}\t/ max: {2:03f}\t / min: {3:03f}".format(
                tracker.tracker_name,
                total_tracker_process_times[i]/frame_count,
                max_tracker_process_times[i],
                min_tracker_process_times[i]
            )))
        print(Logging.s("\tEvent Detection"))
        for i, event_detector in enumerate(event_detectors):
            print(Logging.s("\t\t{0} - average: {1:03f}\t/ max: {2:03f}\t / min: {3:03f}".format(
                event_detector.model_name,
                total_event_process_times[i]/frame_count,
                max_event_process_times[i],
                min_event_process_times[i]
            )))
        print(Logging.s("\tTotal Process Time: {0:03f}".format(total_process_end_time - total_process_start_time)))

    print(Logging.inl("Extraction is successfully completed(frame_count: {})".format(frame_number)))
    if os.path.exists(bbox_video_path) :
        print(Logging.i("BBox video is successfully generated(path: {})".format(bbox_video_path)))
    else :
        print(Logging.i("BBox video is failed to generated."))
    return event_results, sequence_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--video_path", type=str, default="videos/001_360.mp4", help="Video path")
    parser.add_argument("--fps", type=int, default=20, help="FPS of extraction frame ")
    parser.add_argument("--od_model_name", type=str, default="yolov4-416", help="Object detection model name")
    parser.add_argument("--score_threshold", type=float, default=0.1, help="Object detection nms threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.5, help="Object detection nms threshold")
    parser.add_argument("--byte_tracker_params", type=str, default="0.1,0.5,30,0.8,10,20",
                        help="Byte tracker parameters(score_threshold,track_threshold,tracker_buffer,match_threshold,min_box_area,frame_rate)")
    parser.add_argument("--sort_params", type=str, default="0.5,2,20", help="Sort tracker parameters(score_threshold,max_age,min_hits)")
    parser.add_argument("--event_model", type=str, default="all", help="Event model names")
    parser.add_argument("--event_model_score_threshold", type=str, default="0,0.5,0.5,0,0", help="Event model score threshold(assault,falldown,kidnapping,tailing,wanderer)")
    parser.add_argument("--result_dir", type=str, default="./result", help="Directory path of results and frame")
    parser.add_argument("--save_frame_result", action='store_true', help="Is save frame result")
    parser.add_argument("--process_time", action='store_true', help="print processing time event models")

    option = parser.parse_known_args()[0]

    video_path = option.video_path
    json_path = str(video_path).replace("_360p", "").replace(".mp4", ".json")
    if ".mp4" in video_path and os.path.exists(json_path):
        json_path = str(video_path).replace("_360p", "").replace(".mp4", ".json")
    else :
        json_path = None
    video_name = video_path.split("/")[-1]
    extract_fps = option.fps
    score_threshold = option.score_threshold
    nms_threshold = option.nms_threshold
    od_model_name = option.od_model_name
    byte_tracker_params = str(option.byte_tracker_params).split(",")
    sort_params = str(option.sort_params).split(",")
    event_model_names = option.event_model
    event_model_score_threshold = str(option.event_model_score_threshold).split(",")
    result_dir = option.result_dir
    save_frame_result = option.save_frame_result
    process_time = option.process_time
    event_model_tracker = {
        "assault": "byte_tracker",
        "falldown": "none",
        "kidnapping": "none",
        "tailing": "byte_tracker",
        "wanderer": "sort",
    }
    event_model_score_thresholds = {
        "assault": float(event_model_score_threshold[0]),
        "falldown": float(event_model_score_threshold[1]),
        "kidnapping": float(event_model_score_threshold[2]),
        "tailing": float(event_model_score_threshold[3]),
        "wanderer": float(event_model_score_threshold[4]),
    }

    bbox_video_path = os.path.join(result_dir, video_name.split(".mp4")[0] + "_bbox.avi")
    print(Logging.i("Argument Info:"))
    print(Logging.s("input video path: {}".format(video_path)))
    print(Logging.s("input json path: {}".format(json_path)))
    print(Logging.s("extract fps: {}".format(extract_fps)))
    print(Logging.s("result directory path: {}".format(result_dir)))
    print(Logging.s("bbox video path: {}".format(bbox_video_path)))
    print(Logging.s("Object Detection model params"))
    print(Logging.s("\tmodel name: {}".format(od_model_name)))
    print(Logging.s("\tscore threshold: {}".format(score_threshold)))
    print(Logging.s("\tnms threshold: {}".format(nms_threshold)))
    print(Logging.s("Tracker info"))
    print(Logging.s("\tBYTE Tracker"))
    print(Logging.s("\t\tscore threshold: {}".format(byte_tracker_params[0])))
    print(Logging.s("\t\ttrack threshold: {}".format(byte_tracker_params[1])))
    print(Logging.s("\t\ttrack buffer: {}".format(byte_tracker_params[2])))
    print(Logging.s("\t\tmatch threshold: {}".format(byte_tracker_params[3])))
    print(Logging.s("\t\tmin box area: {}".format(byte_tracker_params[4])))
    print(Logging.s("\t\tframe rate: {}".format(byte_tracker_params[5])))
    print(Logging.s("\tSort"))
    print(Logging.s("\t\tscore threshold: {}".format(sort_params[0])))
    print(Logging.s("\t\tmax age: {}".format(sort_params[1])))
    print(Logging.s("\t\tmin hits: {}".format(sort_params[2])))
    print(Logging.s("Event model info(model name/tracker/score threshold):".format(event_model_names)))
    for event_name in split_model_names(event_model_names):
        print(Logging.s("\t{}\t/ {}\t/ {}".format(event_name, event_model_tracker[event_name], event_model_score_thresholds[event_name])))
    tracker_params = [{
            "tracker_class": BYTETracker,
            "tracker_name": "byte_tracker",
            "score_threshold": float(byte_tracker_params[0]),
            "track_threshold": float(byte_tracker_params[1]),
            "track_buffer": int(byte_tracker_params[2]),
            "match_threshold": float(byte_tracker_params[3]),
            "min_box_area": int(byte_tracker_params[4]),
            "frame_rate": int(byte_tracker_params[5])
        },
        {
            "tracker_class": Sort,
            "tracker_name": "sort",
            "score_threshold": float(sort_params[0]),
            "max_age": int(sort_params[1]),
            "min_hits": int(sort_params[1])
        }
    ]

    # Load Video and json
    capture, frame_count, fps = load_video(video_path, extract_fps)
    cam_ids = load_label(json_path, frame_count)

    # Load Object Detection & Event Detection models
    od_model, trackers, event_detectors = load_models(od_model_name, tracker_params, score_threshold=0.1, nms_threshold=nms_threshold, event_model_names=event_model_names)

    # Result Directory info
    frame_dir, fram_bbox_dir, json_dir, event_dir = make_result_dir(result_dir, video_name, save_frame_result)

    # Run detection
    video_info = {"video_path": video_path, "fps": fps, "frame_count": frame_count, 'extract_fps': extract_fps, "cam_id": video_path}
    event_results, sequence_results = run_detection(video_info, cam_ids, od_model, trackers, event_model_score_thresholds, event_detectors, frame_dir, json_dir, bbox_video_path, save_frame_result, process_time)

    # Extract event result as csv
    if save_frame_result:
        extract_event_results(split_model_names(event_model_names), event_dir, video_name, event_detectors, event_results)

    # Extract sequence result as json
    sequence_json_path = extract_sequence_results(event_dir, video_name, sequence_results)

    # Draw event in bbox video
    draw_event(bbox_video_path, sequence_json_path, extract_fps, split_model_names(event_model_names))