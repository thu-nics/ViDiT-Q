import csv
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from . import video_transforms
from .utils import center_crop_arr


def get_transforms_video(resolution=256):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform_video


def get_transforms_image(image_size=256):
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    return transform


class DatasetFromCSV(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        csv_path,
        num_frames=16,
        frame_interval=1,
        transform=None,
        root=None,
    ):
        self.csv_path = csv_path
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            self.samples = list(reader)

        ext = self.samples[0][0].split(".")[-1]
        if ext.lower() in ("mp4", "avi", "mov", "mkv"):
            self.is_video = True
        else:
            assert f".{ext.lower()}" in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            self.is_video = False

        self.transform = transform

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.temporal_sample = video_transforms.TemporalRandomCrop(num_frames * frame_interval)
        self.root = root

    def getitem(self, index):
        sample = self.samples[index]
        path = sample[0]
        if self.root:
            path = os.path.join(self.root, path)
        text = sample[1]

        if self.is_video:
            vframes, aframes, info = torchvision.io.read_video(filename=path, pts_unit="sec", output_format="TCHW")
            total_frames = len(vframes)

            # Sampling video frames
            start_frame_ind, end_frame_ind = self.temporal_sample(total_frames)
            assert (
                end_frame_ind - start_frame_ind >= self.num_frames
            ), f"{path} with index {index} has not enough frames."
            frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, self.num_frames, dtype=int)

            video = vframes[frame_indice]
            video = self.transform(video)  # T C H W
        else:
            image = pil_loader(path)
            image = self.transform(image)
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        return {"video": video, "text": text}

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                print(e)
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.samples)

from qdiff.utils import get_quant_calib_data
# INFO: loading calib_data
class QuantCalibDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, dataset_cfg):
        """
        Initialization method to prepare the data.

        Parameters:
        data_paths (list): List of paths to the data samples.
        transforms (callable, optional): A function/transform that takes in a 
                                         sample and returns a transformed version.
                                         E.g, data augmentations.
        """
        self.data_paths = data_paths
        self.dataset_cfg = dataset_cfg
        calib_data_ckpt = torch.load(self.data_path, map_location='cpu')
        self.calib_data = get_quant_calib_data(dataset_cfg, calib_data_ckpt, dataset_cfg.n_steps) # [calib_xs, calib_ts, calib_cs, calib_masks]
        # import ipdb; ipdb.set_trace()

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset.

        Parameters:
        idx (int): Index of the data sample to retrieve.

        Returns:
        torch.Tensor: A tensor corresponding to the data sample.
        """
        # Load data and get label

        return calib_data_
