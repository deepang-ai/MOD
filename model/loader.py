import json
import os
from typing import Tuple, List, Mapping, Hashable, Dict

import monai
import numpy as np
import torch
from easydict import EasyDict


class ConvertToMultiChannelBasedOnClasses(monai.transforms.transform.Transform):
    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(self, is2019=False, is_hepatic_vessel=False):
        super().__init__()
        self.is2019 = is2019
        self.is_hepatic_vessel = is_hepatic_vessel

    def __call__(
        self, img: monai.config.NdarrayOrTensor
    ) -> monai.config.NdarrayOrTensor:
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        if self.is_hepatic_vessel:
            result = [(img == 1), (img == 2)]
        elif self.is2019:
            # BraTs2019
            # TC WT ET
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # BraTs2021
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )


class ConvertToMultiChannelBasedOnClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = ConvertToMultiChannelBasedOnClasses.backend

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is_hepatic_vessel=False,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019
        self.is_hepatic_vessel = is_hepatic_vessel
        self.converter = ConvertToMultiChannelBasedOnClasses(is2019, is_hepatic_vessel)

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii.gz"
        t1_img = image_path + "_t1.nii.gz"
        t1ce_img = image_path + "_t1ce.nii.gz"
        t2_img = image_path + "_t2.nii.gz"
        seg_img = image_path + "_seg.nii.gz"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def load_brats2020_dataset_images(root):
    images_path = os.listdir(root)
    images_path.remove("name_mapping.csv")
    images_path.remove("survival_info.csv")
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii"
        t1_img = image_path + "_t1.nii"
        t1ce_img = image_path + "_t1ce.nii"
        t2_img = image_path + "_t2.nii"
        seg_img = image_path + "_seg.nii"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def load_brats2019_dataset_images(root):
    root_dir = root + "/dataset.json"
    # 读打开文件
    with open(root_dir, encoding="utf-8") as a:
        # 读取文件
        images_list = json.load(a)["training"]
        for image in images_list:
            image["image"] = image["image"].replace("./", root + "/")
            image["label"] = image["label"].replace("./", root + "/")
    return images_list


def get_transforms(image_size: int, is2019=False) -> Tuple[
    monai.transforms.Compose,
    monai.transforms.Compose,
    monai.transforms.Compose,
    monai.transforms.Compose,
]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnClassesd(keys=["label"], is2019=is2019),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # monai.transforms.CenterSpatialCropD(
            #     keys=["image", "label"],
            #     roi_size=monai.utils.ensure_tuple_rep(image_size, 3),
            # ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=1,
                spatial_size=monai.utils.ensure_tuple_rep(image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            # monai.transforms.RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=config.model.image_size,
            #     random_size=False,
            # ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnClassesd(keys="label", is2019=is2019),
            monai.transforms.Orientationd(
                keys=["image", "label"], axcodes="RAS"
            ), 
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # monai.transforms.Resized(
            #     keys=["image", "label"],
            #     spatial_size=ensure_tuple_rep(config.model.image_size, 3),
            # ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )

    pretrain_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnClassesd(keys="label", is2019=is2019),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=1,
                spatial_size=monai.utils.ensure_tuple_rep(image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
        ]
    )
    pretrain_val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnClassesd(keys="label", is2019=is2019),
        ]
    )
    return train_transform, val_transform, pretrain_transform, pretrain_val_transform


def get_dataloader(
    config: EasyDict, image_size: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.trainer.is_brats2019:
        train_images = load_brats2019_dataset_images(config.trainer.brats2019)
    else:
        train_images = load_brats2021_dataset_images(config.trainer.brats2021)

    train_transform, val_transform, _, _ = get_transforms(
        image_size, is2019=config.trainer.is_brats2019
    )

    train_dataset = monai.data.Dataset(
        data=train_images[: int(len(train_images) * config.trainer.train_ratio)],
        transform=train_transform,
    )
    val_dataset = monai.data.Dataset(
        data=train_images[int(len(train_images) * config.trainer.train_ratio) :],
        transform=val_transform,
    )
    # val_dataset = monai.data.Dataset(data=train_images[3:5], transform=val_transform)  # TODO

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )
    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=False,
    )
    return train_loader, val_loader
