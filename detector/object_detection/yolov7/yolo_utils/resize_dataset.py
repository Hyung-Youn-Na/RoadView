from PIL import Image
import os

origin_dir = '/mlsun/cctv/dataset/obstacle/yolov5-FHD/images'
dataset_types = os.listdir(origin_dir)
target_dir = '/dataset/yolov5-416/images'

for dataset_type in dataset_types:
    origin_image_dir = os.path.join(origin_dir, dataset_type)
    target_image_dir = os.path.join(target_dir, dataset_type)

    image_names = os.listdir(origin_image_dir)
    for i, image_name in enumerate(image_names):
        origin_image_path = os.path.join(origin_image_dir, image_name)
        target_image_path = os.path.join(target_image_dir, image_name)
        if not os.path.exists(target_image_path):
            try:
                if os.path.exists(origin_image_path):
                    image = Image.open(origin_image_path)
                    image = image.resize((416, 416))
                    image = image.convert("RGB")
                    image.save(target_image_path)
            except:
                print("{} is not exist".format(target_image_path))
        print("\r{} - {: 6}\t{} -> {}".format(dataset_type, i + 1, origin_image_path, target_image_path), end="")
