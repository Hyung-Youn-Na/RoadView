import time

class ObjectDetection:
    def __init__(self, model='Template'):
        """
        :param model: model name
        :param category_num:
        """
        self.model_name = model
        start_time = time.time()
        end_time = time.time()

    def inference_by_image(self, image, confidence_threshold=0.3):
        """
        :param image: input image
        :param confidence_threshold: confidence/score threshold for object detection.
        :return: bounding box(x1, y1, x2, y2), scores, classes
        """
        results = []

        results.append({
            "label" :[
                {
                    "description": "person",
                    "score": 1.0,
                    "class_idx": 0
                }
            ],
            "position": {
                "x": 30,
                "y": 30,
                "w": 100,
                "h": 100
            }
        })

        results.append({
            "label": [
                {
                    "description": "car",
                    "score": 1.0,
                    "class_idx": 1
                }
            ],
            "position": {
                "x": 200,
                "y": 200,
                "w": 100,
                "h": 100
            }
        })
        time.sleep(0.25)

        return results