"""Custom class for LVT dataset."""

import glob
import re
import os
import json
import random
import cv2
import shutil

from nvidia_tao_pytorch.cv.re_identification.dataloader.datasets.bases import BaseImageDataset
from ultralytics import YOLO


class LVTReIDDataset(BaseImageDataset):
    """Custom class for the LVTReIDDataset dataset.

    This class provides an interface to the LVTReIDDataset dataset and inherits from the BaseImageDataset class.

    """

    def __init__(self, experiment_spec, prepare_for_training, create_dataset=False, yolo_pose_model=None,  verbose=False):
        """Initialize the LVTReIDDataset dataset.

        Args:
            experiment_spec (dict): Specification of the experiment.
            prepare_for_training (bool): If True, prepare the dataset for training.
            create_dataset (bool): If True, create the dataset.
            verbose (bool, optional): If True, print verbose information. Defaults to False.

        """
        super(LVTReIDDataset, self).__init__()
        self.prepare_for_training = prepare_for_training

        self.dataset_dir = experiment_spec["dataset"]["dataset_dir"]

        if self.prepare_for_training:
            self.train_dir = experiment_spec["dataset"]["train_dataset_dir"]
            self.query_dir = experiment_spec["dataset"]["query_dataset_dir"]
            self.gallery_dir = experiment_spec["dataset"]["test_dataset_dir"]
        elif experiment_spec["inference"]["query_dataset"] and experiment_spec["inference"]["test_dataset"]:
            self.query_dir = experiment_spec["inference"]["query_dataset"]
            self.gallery_dir = experiment_spec["inference"]["test_dataset"]
        elif experiment_spec["evaluate"]["query_dataset"] and experiment_spec["evaluate"]["test_dataset"]:
            self.query_dir = experiment_spec["evaluate"]["query_dataset"]
            self.gallery_dir = experiment_spec["evaluate"]["test_dataset"]
        self._check_before_run()

        self.yolo_pose_model = None
        if create_dataset:
            self.yolo_pose_model = yolo_pose_model
            self.crops_dir = os.path.join(self.dataset_dir, "reid_dataset", "crops")
            if len(os.listdir(self.crops_dir)) == 0:
                self._create_crops()
            if len(os.listdir(self.gallery_dir)) == 0:
                self._redistribute_crops()

        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        if self.prepare_for_training:
            train = self._process_dir(self.train_dir, relabel=True)
            self.print_dataset_statistics(train, query, gallery)
        else:
            self.print_dataset_statistics(query, gallery)
        if self.prepare_for_training:
            self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = None
        if self.prepare_for_training:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)

        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper."""
        if self.prepare_for_training and not os.path.exists(self.train_dir):
            raise FileNotFoundError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise FileNotFoundError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise FileNotFoundError("'{}' is not available".format(self.gallery_dir))

    def _create_crops(self):
        assert os.path.exists(self.yolo_pose_model)
        if not os.path.isdir(self.crops_dir):
            os.mkdir(self.crops_dir)
        object_key_name_map = {}
        unknown_count = 0
        scene_person_ids = {}
        # Load a model
        scene_id = 0
        pose_model = YOLO(self.yolo_pose_model)  # load an official model
        for data_dir in glob.glob(os.path.join(self.dataset_dir, "dataset_*")):
            if os.path.isdir(data_dir):
                video_dir = os.path.join(data_dir, "video")
                videos = glob.glob(os.path.join(video_dir, "*.mp4"))
                cameras = [os.path.splitext(os.path.basename(file_path))[0] for file_path in videos]
                annotation_dir = os.path.join(data_dir, "ann")

                for camera_id, camera_name in enumerate(cameras):
                    image_names = [image_path.split("/")[-1] for image_path in list(sorted(
                        glob.glob(os.path.join(video_dir, camera_name, "*.jpg"))))]
                    annotation_file = list(glob.glob(os.path.join(annotation_dir, camera_name + "*.json")))[0]
                    with open(annotation_file, "r") as f:
                        annotations = json.load(f)
                    for ann_object in annotations["objects"]:
                        if ann_object["key"] not in object_key_name_map.keys():
                            if len(ann_object["tags"]) > 0:
                                object_key_name_map[ann_object["key"]] = ann_object["tags"][0]["name"]
                            else:
                                object_key_name_map[ann_object["key"]] = "unknown #"+str(unknown_count)
                                unknown_count+=1

                    for frame in annotations["frames"]:
                        index = str(frame["index"])
                        image_filename = [x for x in image_names if re.search(index, x)][0]
                        image_path = os.path.join(video_dir, camera_name, image_filename)
                        image = cv2.imread(image_path)
                        for fig in frame["figures"]:
                            points = fig["geometry"]["points"]["exterior"]
                            top_left = points[0]
                            bottom_right = points[1]
                            if top_left[1]==bottom_right[1] or top_left[0]==bottom_right[0]:
                                continue
                            cropped = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                            poses = pose_model(cropped)
                            if poses[0].keypoints.shape[1] != 0:
                                obj_name = object_key_name_map[fig["objectKey"]]
                                if obj_name not in scene_person_ids.keys():
                                    scene_person_ids[obj_name] = {"person_id": len(scene_person_ids.keys()),
                                                                  "cam_counts": {camera_id: 0}}
                                if camera_id not in scene_person_ids[obj_name]["cam_counts"].keys():
                                    scene_person_ids[obj_name]["cam_counts"][camera_id] = 0
                                crop_image_path = os.path.join(self.crops_dir, "{:04d}_c{}s{}_{:06d}.jpg".format(
                                    scene_person_ids[obj_name]["person_id"]+2000,
                                    camera_id+10 ,
                                    scene_id,
                                    scene_person_ids[obj_name]["cam_counts"][camera_id]))
                                cv2.imwrite(crop_image_path, cropped)
                                scene_person_ids[obj_name]["cam_counts"][camera_id] += 1
                            del poses, cropped
                        del image
                    del image_names
                scene_id += 1

    def _redistribute_crops(self):
        crop_count = {"train": 100, "query": 100, "gallery": 100}
        min_crop_count = {"train": 5, "query": 5, "gallery": 2}
        image_names = [image_path.split("/")[-1] for image_path in list(sorted(glob.glob(os.path.join(self.crops_dir, "*.jpg"))))]
        pattern = re.compile(r'(\d+)_c(\d+)')
        image_map = {}
        pid_container = set()
        for img_name in image_names:
            pid, cam_id = map(int, pattern.search(img_name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
            if pid not in image_map.keys():
                image_map[pid] = [ [] for _ in range(cam_id+1) ]
            if len(image_map[pid]) < (cam_id+1):
                add_cams = cam_id + 1 - len(image_map[pid])
                [image_map[pid].append([]) for _ in range(add_cams)]
            image_map[pid][cam_id].append(img_name)

        test_pids = set(random.sample(pid_container, len(pid_container)//2))
        for pid, image_lists in image_map.items():
            image_lists.sort(key=len)
            num_cams = len(image_lists)
            if pid in test_pids:
                gallery_crops_per_cam = -(-crop_count["gallery"] // num_cams)
                query_crops_per_cam = -(-crop_count["query"] // num_cams)
                gallery_crops = []
                query_crops = []

                for index, crops in enumerate(image_lists):
                    random.shuffle(crops)
                    gallery_crops.extend(crops[:gallery_crops_per_cam])
                    query_crops.extend(crops[gallery_crops_per_cam:(gallery_crops_per_cam + query_crops_per_cam)])
                    if index < (num_cams - 1):
                        query_crops_per_cam = -(-(crop_count["query"] - len(query_crops)) // (num_cams - index - 1))
                        gallery_crops_per_cam = -(-(crop_count["gallery"] - len(gallery_crops)) // (num_cams - index - 1))
                if len(gallery_crops) < min_crop_count["gallery"] or len(query_crops) < min_crop_count["query"]:
                    continue
                [shutil.copy(os.path.join(self.crops_dir, image_name),
                             os.path.join(self.gallery_dir, image_name)) for image_name in gallery_crops]
                [shutil.copy(os.path.join(self.crops_dir, image_name),
                             os.path.join(self.query_dir, image_name)) for image_name in query_crops]
            else:
                train_crops_per_cam = -(-crop_count["train"] // num_cams)
                train_crops = []
                for index, crops in enumerate(image_lists):
                    random.shuffle(crops)
                    train_crops.extend(crops[:train_crops_per_cam])
                    if index < (num_cams - 1):
                        train_crops_per_cam = -(-(crop_count["train"] - len(train_crops)) // (num_cams - index - 1))
                if len(train_crops) < min_crop_count["train"] :
                    continue
                [shutil.copy(os.path.join(self.crops_dir, image_name),
                             os.path.join(self.train_dir, image_name)) for image_name in train_crops]


    def _process_dir(self, dir_path, relabel=False):
        """Check the directory and return a dataset.

        Args:
            dir_path (str): Path to the directory.
            relabel (bool, optional): If True, relabel the data. Defaults to False.

        Returns:
            list: A list of tuples containing the image path, person ID, and camera ID.

        """
        image_names = [image_path.split("/")[-1] for image_path in list(sorted(glob.glob(os.path.join(dir_path, "*.jpg"))))]
        pattern = re.compile(r'(\d+)_c(\d+)')

        pid_container = set()
        for img_name in image_names:
            pid, _ = map(int, pattern.search(img_name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_name in image_names:
            pid, camid = map(int, pattern.search(img_name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            # assert 0 <= pid <= 1501, "The number of person IDs should be between 0 and 1501."
            # assert 1 <= camid <= 6, "The number of camera IDs should be between 0 and 6."
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            dataset.append((os.path.join(dir_path, img_name), pid, camid))
        return dataset
