import os

import shutil

root_dir = "/dataset/yolov5-1280"
target_root_dir = root_dir + "_yj"

train_list_path = os.path.join("/dataset", "train.txt")
val_list_path = os.path.join("/dataset", "valid.txt")

dataset_types = [
    "val",
    "test",
    "train"
]
image_dir = os.path.join(root_dir, "images")
label_dir = os.path.join(root_dir, "labels")

image_paths = []

for dataset_type in dataset_types:
    tmp_image_list = os.listdir(os.path.join(image_dir, dataset_type))
    for image_name in tmp_image_list:
        image_paths.append(os.path.join(image_dir, dataset_type, image_name))

valid_list_file = open(val_list_path, "r")

index = 0
for line in valid_list_file:
    selected_image_name = line.replace("\n", "").split("/")[-1].replace(".jpg", "")
    for image_path in image_paths:
        if selected_image_name in image_path :
            origin_image_path = image_path
            origin_label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
            target_image_path = os.path.join(target_root_dir, "images", "val", selected_image_name + ".jpg")
            target_label_path = os.path.join(target_root_dir, "labels", "val", selected_image_name + ".txt")

            shutil.copy(origin_image_path, target_image_path)
            shutil.copy(origin_label_path, target_label_path)
            image_paths.remove(image_path)
            print("\rvalid - {}/{} {} -> {} | {} -> {}".format(index, len(image_paths), origin_image_path, target_image_path, origin_label_path, target_label_path), end="")

            index += 1
valid_list_file.close()

train_list_file = open(train_list_path, "r")
index = 0
for line in train_list_file:
    selected_image_name = line.replace("\n", "").split("/")[-1].replace(".jpg", "")
    for image_path in image_paths:
        if selected_image_name in image_path :
            origin_image_path = image_path
            origin_label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
            target_image_path = os.path.join(target_root_dir, "images", "train", selected_image_name + ".jpg")
            target_label_path = os.path.join(target_root_dir, "labels", "train", selected_image_name + ".txt")

            shutil.copy(origin_image_path, target_image_path)
            shutil.copy(origin_label_path, target_label_path)
            print("\rtrain - {}/{} {} -> {} | {} -> {}".format(index, len(image_paths), origin_image_path, target_image_path, origin_label_path, target_label_path), end="")

            index += 1
train_list_file.close()

print()
