import numpy as np

def getAverageArea(object_bboxes, text_bboxes):
    text_groups = []
    avg_areas = []
    for obj_bbox in object_bboxes:
        text_area_sum = 0.0
        text_group = []
        for n, text_bbox in enumerate(text_bboxes):
            if getInnerTextRegionRatio(obj_bbox, text_bbox) > 0.75:
                textBbox_area = (text_bbox[2] - text_bbox[0] + 1) * (text_bbox[3] - text_bbox[1] + 1)
                text_area_sum += textBbox_area
                text_group.append(n)
        if len(text_group) != 0:
            avg_areas.append(text_area_sum / len(text_group))
        else:
            avg_areas.append(0.0)
        text_groups.append(text_group)
    return text_groups, avg_areas


def getInnerTextRegionRatio(objectBbox, textBbox):
    textBbox_area = (textBbox[2] - textBbox[0] + 1) * (textBbox[3] - textBbox[1] + 1)

    x1 = max(objectBbox[0], textBbox[0])
    y1 = max(objectBbox[1], textBbox[1])
    x2 = min(objectBbox[2], textBbox[2])
    y2 = min(objectBbox[3], textBbox[3])

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h

    text_ratio = inter / textBbox_area
    return text_ratio


def merge_and_filter_bboxes(object_bboxes, text_bboxes,
                            text_detection_confidences):
    text_groups, avg_areas = getAverageArea(object_bboxes, text_bboxes)
    group_text_bboxes = []
    for object_bbox, object_text_group, object_avg_area in zip(object_bboxes,
                                                               text_groups, avg_areas):
        group_xmin = 987654321
        group_ymin = 987654321
        group_xmax = 0
        group_ymax = 0
        group_text_conf = 0.0
        group_num = 0
        for text_index in object_text_group:
            text_bbox = text_bboxes[text_index]
            text_area = (text_bbox[2] - text_bbox[0] + 1) * (text_bbox[3] - text_bbox[1] + 1)
            if text_area > 0.5 * object_avg_area:
                group_text_conf += text_detection_confidences[text_index]
                group_num += 1
                xmin, ymin, xmax, ymax = text_bbox
                if xmin < group_xmin:
                    group_xmin = xmin
                if ymin < group_ymin:
                    group_ymin = ymin
                if xmax > group_xmax:
                    group_xmax = xmax
                if ymax > group_ymax:
                    group_ymax = ymax
        if group_num != 0:
            group_text_conf = group_text_conf / group_num
            text_line = f'text {group_text_conf} {group_xmin} {group_ymin} {group_xmax} {group_ymax}\n'
            group_text_bbox = [group_xmin, group_ymin, group_xmax, group_ymax]
            group_text_bboxes.append(group_text_bbox)

    return group_text_bboxes

def getFilteredTextGroup(object_bboxes, text_bboxes):
    text_groups = []
    avg_areas = []
    filtered_text_groups = []
    for obj_bbox in object_bboxes:
        text_area_sum = 0.0
        text_group = []
        filtered_text_group = []
        for n, text_bbox in enumerate(text_bboxes):
            if getInnerTextRegionRatio(obj_bbox, text_bbox) > 0.75:
                textBbox_area = (text_bbox[2] - text_bbox[0] + 1) * (text_bbox[3] - text_bbox[1] + 1)
                text_area_sum += textBbox_area
                text_group.append(n)
        if len(text_group) != 0:
            avg_areas.append(text_area_sum / len(text_group))
        else:
            avg_areas.append(0.0)
        for idx in text_group:
            in_ganpan_text_bbox = text_bbox[idx]
            ganpan_text_bbox_area = (in_ganpan_text_bbox[2] - in_ganpan_text_bbox[0] + 1) *\
                                    (in_ganpan_text_bbox[3] - in_ganpan_text_bbox[1] + 1)
            if ganpan_text_bbox_area > text_area_sum / len(text_group):
                filtered_text_group.append(idx)
        filtered_text_groups.append(filtered_text_group)

    return filtered_text_groups

# def merge_text_recognition_result(result):
#     for query_img_text_group, pred_img_text_group, query_group_text, pred_group_text,\
#         query_group_text_confidence, pred_group_text_confidence, in zip(result['query_img_text_groups'],
#         result['pred_img_text_groups'], result['query_group_texts'], result['pred_group_text'],
#         result['query_group_text_confidences'], result['pred_group_text_confidence']):

