import os
from datetime import datetime

import accelerate.utils
import monai
import pytz
import logging
import torch
from functools import partial
from monai.utils import ensure_tuple_rep
from objprint import objstr
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
import yaml
from easydict import EasyDict

from model.loader import get_dataloader
from model.loss import OTLoss, DPLoss
from model.model.swin_unetr import MODSwinUNETR
from model.utils import common_params, cosine_scheduler, MaskGenerator


class MODTrainer(object):
    def __init__(self, config: EasyDict):
        torch.multiprocessing.set_sharing_strategy("file_system")
        accelerate.utils.set_seed(config.trainer.seed)
        logging_dir = os.path.join(
            os.getcwd(),
            "output/swin_unetr/logs/",
            str(
                datetime.now(tz=pytz.timezone("Asia/Shanghai")).strftime(
                    "%Y-%m-%d-%H:%M:%S"
                )
            ),
        )
        os.makedirs(logging_dir, exist_ok=True)
        self.accelerator = Accelerator(
            log_with="tensorboard",
            project_dir=logging_dir,
            mixed_precision=config.trainer.mixed_precision,
        )
        self.accelerator.init_trackers(__name__)
        if self.accelerator.is_main_process:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(logging_dir + "/log.txt"),
                ],
                force=True,
            )
        self.logger = get_logger("trainer")
        self.logger.info(objstr(config))
        self.config = config
        self.train_step = 0
        self.val_step = 0
        self.starting_epoch = 0
        self.seg_only = self.config.trainer.seg_only
        self.student_model = MODSwinUNETR(**config.swin_unetr)

        if self.seg_only:
            self.student_model.decoder.requires_grad_(False)
            self.student_model.swinViT.masked_embed.requires_grad = False
        else:
            self.teacher_model = MODSwinUNETR(**config.swin_unetr)
            self.teacher_model.requires_grad_(False)
            self.teacher_model.to(self.accelerator.device)
        self.patch_size = 64
        self.img_size = config.swin_unetr.img_size
        self.train_loader, self.val_loader = get_dataloader(
            config, config.swin_unetr.img_size
        )
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=config.trainer.lr,
            betas=(0.9, 0.999),
            weight_decay=config.trainer.weight_decay,
            eps=1e-8,
        )
        self.scheduler = transformers.get_scheduler(
            "cosine_with_restarts",
            self.optimizer,
            num_warmup_steps=config.trainer.warmup * len(self.train_loader),
            num_training_steps=config.trainer.num_epochs * len(self.train_loader),
        )
        self.inference = monai.inferers.SlidingWindowInferer(
            roi_size=ensure_tuple_rep(config.swin_unetr.img_size, 3),
            overlap=0.5,
            sw_device=self.accelerator.device,
            device=self.accelerator.device,
        )
        self.post_trans = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.metrics = {
            "dice_metric": monai.metrics.DiceMetric(
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=False,
            ),
            "hd95_metric": monai.metrics.HausdorffDistanceMetric(
                percentile=95,
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=False,
            ),
        }
        self.eval_metric = {
            "dice_metric": monai.metrics.DiceMetric(
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=False,
            ),
            "hd95_metric": monai.metrics.HausdorffDistanceMetric(
                percentile=95,
                include_background=True,
                reduction=monai.utils.MetricReduction.MEAN_BATCH,
                get_not_nans=False,
            ),
        }
        self.loss_functions = {
            "focal_loss": monai.losses.FocalLoss(to_onehot_y=False),
            # "generalized_dice_loss": monai.losses.GeneralizedDiceLoss(
            #     smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
            # ),
            "dice_loss": monai.losses.DiceLoss(
                smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True
            ),
        }
        if not self.seg_only:
            self.rand_adjust_contrast = monai.transforms.RandAdjustContrast(prob=1)
            self.rand_gaussian_sharpen = monai.transforms.RandGaussianSharpen(prob=1)
            self.ot_loss = OTLoss(self.config.trainer.patch_ratio)
            self.dp_loss = DPLoss(2, self.config.trainer.dp_ratio)
            self.mask_generator = MaskGenerator(
                input_size=self.img_size,
                model_patch_size=self.student_model.patch_size,
                mask_ratio=config.trainer.mask_ratio,
            )
            self.momentum_schedule = cosine_scheduler(
                config.trainer.momentum_teacher,
                1,
                config.trainer.num_epochs,
                len(self.train_loader),
            )
            # find student and teacher common params
            self.params_q, self.params_k = common_params(
                self.student_model, self.teacher_model, self.accelerator
            )

    def train_one_epoch(self, epoch):
        self.student_model.train()
        if not self.seg_only:
            self.teacher_model.train()
        for i, image_batch in enumerate(self.train_loader):
            total_loss = 0
            log = ""
            if not self.seg_only:
                image_a = torch.cat(
                    [
                        self.rand_adjust_contrast(img).unsqueeze(0)
                        for img in image_batch["image"]
                    ]
                )
                image_b = torch.cat(
                    [
                        self.rand_gaussian_sharpen(img).unsqueeze(0)
                        for img in image_batch["image"]
                    ]
                )
                input_image = torch.cat((image_a, image_b))

                input_masks = torch.cat(
                    [self.mask_generator().unsqueeze(0) for _ in input_image]
                ).to(self.accelerator.device)
                student_output = self.student_model(input_image, mask=input_masks)
                with torch.no_grad():
                    teacher_output = self.teacher_model(input_image)
                # online tokenizer loss
                patch_loss = self.ot_loss(
                    student_output["patch"],
                    teacher_output["patch"],
                    torch.nn.functional.max_pool3d(
                        input_masks.unsqueeze(1).float(),
                        kernel_size=16,
                        stride=16,
                        padding=0,
                    )
                    .squeeze(1)
                    .long(),
                )
                dp_loss = self.dp_loss(
                    student_output["recon"], input_image, input_masks
                )
                log += f" patch_loss {float(patch_loss):1.5f} dp_loss {float(dp_loss):1.5f}"
                total_loss += patch_loss * 0.001 + dp_loss * 0.001
                self.accelerator.log(
                    {
                        "Train/patch_loss": float(patch_loss),
                        "Train/dp_loss": float(dp_loss),
                    },
                    step=self.train_step,
                )
            student_output_wo_mask = self.student_model(image_batch["image"])
            logits = student_output_wo_mask["seg"]

            # seg loss
            for loss_name, loss_fn in self.loss_functions.items():
                with self.accelerator.autocast():
                    loss = loss_fn(logits, image_batch["label"])
                self.accelerator.log(
                    {"Train/" + loss_name: float(loss)}, step=self.train_step
                )
                log += f" {loss_name} {float(loss):1.5f} "
                total_loss += loss
            val_outputs = [self.post_trans(i) for i in logits]
            for metric_name, metric_fn in self.metrics.items():
                metric_fn(y_pred=val_outputs, y=image_batch["label"])

            self.accelerator.log(
                {
                    "Train/Total Loss": float(total_loss),
                },
                step=self.train_step,
            )
            self.accelerator.backward(total_loss)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Training [{i + 1}/{len(self.train_loader)}] Loss: {float(total_loss)} {log}"
            )
            if not self.seg_only:
                # EMA update for the teacher
                with torch.no_grad():
                    m = self.momentum_schedule[self.train_step]  # momentum parameter
                    for param_q, param_k in zip(self.params_q, self.params_k):
                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            self.train_step += 1

        _metric = {}
        for metric_name, metric_fn in self.metrics.items():
            batch_acc = metric_fn.aggregate()
            if self.accelerator.num_processes > 1:
                batch_acc = (
                    self.accelerator.reduce(batch_acc) / self.accelerator.num_processes
                )
            metric_fn.reset()
            _metric.update(
                {
                    f"Train/mean {metric_name}": float(batch_acc.mean()),
                    f"Train/TC {metric_name}": float(batch_acc[0]),
                    f"Train/WT {metric_name}": float(batch_acc[1]),
                    f"Train/ET {metric_name}": float(batch_acc[2]),
                }
            )
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Training metric {_metric}"
        )
        self.accelerator.log(_metric, step=epoch)

    @torch.no_grad()
    def val_one_epoch(self, epoch):
        self.student_model.eval()
        for i, image_batch in enumerate(self.val_loader):
            logits = self.inference(
                image_batch["image"], partial(self.student_model, return_seg=True)
            )
            total_loss = 0
            log = ""
            for loss_name, loss_fn in self.loss_functions.items():
                with self.accelerator.autocast():
                    loss = loss_fn(logits, image_batch["label"])
                self.accelerator.log(
                    {"Val/" + loss_name: float(loss)}, step=self.val_step
                )
                log += f" {loss_name} {float(loss):1.5f} "
                total_loss += loss
            val_outputs = [self.post_trans(i) for i in logits]
            for metric_name, metric_fn in self.eval_metric.items():
                metric_fn(y_pred=val_outputs, y=image_batch["label"])
            self.accelerator.log(
                {
                    "Val/Total Loss": float(total_loss),
                },
                step=self.val_step,
            )
            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Validation [{i + 1}/{len(self.val_loader)}] Loss: {float(total_loss)} {log}"
            )
            self.val_step += 1

        _metric = {}
        for metric_name, metric in self.eval_metric.items():
            batch_acc = metric.aggregate()
            if self.accelerator.num_processes > 1:
                batch_acc = (
                    self.accelerator.reduce(batch_acc.to(self.accelerator.device))
                    / self.accelerator.num_processes
                )
            metric.reset()
            _metric.update(
                {
                    f"Val/mean {metric_name}": float(batch_acc.mean()),
                    f"Val/TC {metric_name}": float(batch_acc[0]),
                    f"Val/WT {metric_name}": float(batch_acc[1]),
                    f"Val/ET {metric_name}": float(batch_acc[2]),
                }
            )
        self.logger.info(
            f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] Validation metric {_metric}"
        )
        self.accelerator.log(_metric, step=epoch)
        return _metric["Val/mean dice_metric"]

    def _run(self):
        self.accelerator.wait_for_everyone()
        (
            self.student_model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        ) = self.accelerator.prepare(
            self.student_model,
            self.optimizer,
            self.scheduler,
            self.train_loader,
            self.val_loader,
        )

        self.starting_epoch, self.train_step, self.val_step = self.resume_train_state(
            f"{os.getcwd()}/output/swin_unetr/checkpoint",
            self.train_loader,
            self.val_loader,
        )

        self.logger.info("Start training!")
        best_acc = 0
        for epoch in range(self.starting_epoch, self.config.trainer.num_epochs):
            self.train_one_epoch(epoch)
            mean_acc = self.val_one_epoch(epoch)

            self.logger.info(
                f"Epoch [{epoch + 1}/{self.config.trainer.num_epochs}] lr = {self.scheduler.get_last_lr()} mean acc: {mean_acc}"
            )
            if mean_acc > best_acc:
                best_acc = mean_acc
                self.accelerator.save_state(
                    output_dir=f"{os.getcwd()}/output/swin_unetr/checkpoint/best"
                )

            self.accelerator.save_state(
                output_dir=f"{os.getcwd()}/output/swin_unetr/checkpoint1/epoch_{epoch}"
            )
            self.accelerator.log({"Train/lr": self.scheduler.get_last_lr()}, step=epoch)

        self.logger.info(f"Best acc: {best_acc}")
        exit(1)

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.logger.exception(e)
            self.logger.error("Train error!")
            self.accelerator.end_training()
            exit(-1)

    def resume_train_state(
        self,
        base_path: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        try:
            # Get the most recent checkpoint
            dirs = [
                base_path + "/" + f.name
                for f in os.scandir(base_path)
                if (f.is_dir() and f.name.startswith("epoch_"))
            ]
            dirs.sort(
                key=os.path.getctime
            )  # Sorts folders by date modified, most recent checkpoint is the last
            self.logger.info(f"Try to load epoch {dirs[-1]} train state")
            self.accelerator.load_state(dirs[-1])
            training_difference = os.path.splitext(dirs[-1])[0]
            starting_epoch = (
                int(training_difference.replace(f"{base_path}/epoch_", "")) + 1
            )
            step = starting_epoch * len(train_loader)
            if val_loader is not None:
                val_step = starting_epoch * len(val_loader)
            else:
                val_step = 0
            self.logger.info(
                f"Load train state success! Start from epoch {starting_epoch}"
            )
            return starting_epoch, step, val_step
        except Exception as e:
            self.logger.error(e)
            self.logger.error(f"Load train state fail!")
            return 0, 0, 0


if __name__ == "__main__":
    config = EasyDict(
        yaml.load(open("mod-config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    trainer = MODTrainer(config)
    trainer.run()
