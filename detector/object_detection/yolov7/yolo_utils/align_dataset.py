import os

import shutil

root_dir = "/mlsun/cctv/dataset/obstacle/"

target_root_dir = os.path.join(root_dir, "yolov5")

list_paths = [
    # os.path.join(root_dir, "val_list.txt"),
    # os.path.join(root_dir, "test_list.txt"),
    os.path.join(root_dir, "train_list.txt")
]
dataset_types = [
    # "val",
    # "test",
    "train"
]
image_dir = os.path.join(target_root_dir, "images")
label_dir = os.path.join(target_root_dir, "labels")

###

i = 0
for i, dataset_type in enumerate(dataset_types):
    dataset_image_dir = os.path.join(image_dir, dataset_types[i])
    dataset_label_dir = os.path.join(label_dir, dataset_types[i])

    print(dataset_image_dir, dataset_label_dir)

    list_file = open(list_paths[i], 'r')
    file_count = 0
    while True:
        origin_image_path = str(list_file.readline()).replace("\n", "")
        origin_label_path = origin_image_path.replace(".jpg", ".txt")

        image_name = origin_image_path.split("/")[-1]
        label_name = origin_label_path.split("/")[-1]

        target_image_path = os.path.join(dataset_image_dir, image_name)
        target_label_path = os.path.join(dataset_label_dir, label_name)

        if not os.path.exists(target_image_path):
            shutil.copy(origin_image_path, target_image_path)
        if not os.path.exists(target_label_path):
            shutil.copy(origin_label_path, target_label_path)

        file_count +=1
        print("\r{}:{: 6} - {}\t{}".format(dataset_type, file_count, image_name, label_name), end="")
        if not origin_image_path:
            break
    list_file.close()
    print()
