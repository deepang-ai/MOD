import torch
import torch.nn.functional as F


class OTLoss(object):
    def __init__(self, patch_ratio):
        self.cos_similar = torch.nn.CosineSimilarity(dim=1)
        self.patch_ratio = patch_ratio

    def __call__(self, student_patch, teacher_patch, student_mask):
        batch_size, _, _ = student_patch.shape
        mask = student_mask.flatten(-3, -1).unsqueeze(-1)
        student_patch_ = (student_patch * mask).contiguous().view(batch_size, -1)
        teacher_patch_ = (teacher_patch * mask).contiguous().view(batch_size, -1)
        patch_loss = (1 - self.cos_similar(student_patch_, teacher_patch_)).sum()
        # cls_loss = (1 - self.cos_similar(student_cls, teacher_cls)).sum()
        return patch_loss * self.patch_ratio


class DPLoss(object):
    def __init__(self, patch_size, dp_ratio, modality=4):
        super().__init__()
        self.patch_size = patch_size
        self.modality = modality
        self.dp_ratio = dp_ratio

    def __call__(self, pred, imgs, mask):
        """
        imgs: [N, H, W, D]
        pred: [N, H, W, D]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        mask_ = (
            mask.repeat_interleave(self.patch_size, 1)
            .repeat_interleave(self.patch_size, 2)
            .repeat_interleave(self.patch_size, 3)
            .unsqueeze(1)
            .contiguous()
        )
        loss_recon = F.l1_loss(pred, imgs, reduction="none")
        loss = (loss_recon * mask_).sum() / (mask_.sum() + 1e-5) / self.modality
        return loss * self.dp_ratio
