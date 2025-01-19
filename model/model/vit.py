# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.layers import Conv
from monai.utils import ensure_tuple_rep, optional_import

rearrange, _ = optional_import("einops", name="rearrange")


class ViTDecoder(nn.Module):
    def __init__(
        self,
        image_size=128,
        out_channels=4,
        patch_size=16,
        decoder_embed_dim=768,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.path_size = patch_size
        self.out_channels = out_channels
        self.decoder_pred = nn.ModuleList(
            [
                nn.Linear(
                    decoder_embed_dim, patch_size**3, bias=True
                )  # decoder to patch，中间那个参数是patch的w*h*C（patch的像素）映射回图像
                for _ in range(self.out_channels)
            ]
        )

    def forward(self, embedding):
        # predictor projection
        x = torch.cat(
            [self.decoder_pred[i](embedding) for i in range(self.out_channels)], dim=2
        )
        h = w = d = self.image_size // self.path_size
        x = x.reshape(
            (
                x.shape[0],
                h,
                w,
                d,
                self.path_size,
                self.path_size,
                self.path_size,
                self.out_channels,
            )
        )  # [N, patch_h, patch_w, patch_d, patch_size, patch_size, patch_size, modality]
        x = torch.einsum(
            "nhwdopqm->nmhowpdq", x
        )  # [N, modality, patch_h, patch_size, patch_w, patch_size, patch_d, patch_size]
        x = x.reshape(
            (
                x.shape[0],
                self.out_channels,
                self.image_size,
                self.image_size,
                self.image_size,
            )
        )  # [N, modality, img_h, img_w, img_d]
        return x


class MODViTAutoEnc(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    Modified to also give same dimension outputs as the input size of the image
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        out_channels: int = 1,
        deconv_chns: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        proj_type: str = "conv",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels or the number of channels for input.
            img_size: dimension of input image.
            patch_size: dimension of patch size
            out_channels:  number of output channels. Defaults to 1.
            deconv_chns: number of channels for the deconvolution layers. Defaults to 16.
            hidden_size: dimension of hidden layer. Defaults to 768.
            mlp_dim: dimension of feedforward layer. Defaults to 3072.
            num_layers:  number of transformer blocks. Defaults to 12.
            num_heads: number of attention heads. Defaults to 12.
            proj_type: position embedding layer type. Defaults to "conv".
            dropout_rate: fraction of the input units to drop. Defaults to 0.0.
            spatial_dims: number of spatial dimensions. Defaults to 3.
            qkv_bias: apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn: to make accessible the attention in self attention block. Defaults to False. Defaults to False.

        .. deprecated:: 1.4
            ``pos_embed`` is deprecated in favor of ``proj_type``.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            # It will provide an output of same size as that of the input
            >>> net = ViTAutoEnc(in_channels=1, patch_size=(16,16,16), img_size=(96,96,96), proj_type='conv')

            # for 3-channel with image size of (128,128,128), output will be same size as of input
            >>> net = ViTAutoEnc(in_channels=3, patch_size=(16,16,16), img_size=(128,128,128), proj_type='conv')

        """

        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.spatial_dims = spatial_dims
        for m, p in zip(self.img_size, self.patch_size):
            if m % p != 0:
                raise ValueError(
                    f"patch_size={patch_size} should be divisible by img_size={img_size}."
                )

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            proj_type=proj_type,
            dropout_rate=dropout_rate,
            spatial_dims=self.spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

        conv_trans = Conv[Conv.CONVTRANS, self.spatial_dims]
        # self.conv3d_transpose* is to be compatible with existing 3d model weights.
        up_kernel_size = [int(math.sqrt(i)) for i in self.patch_size]
        self.conv3d_transpose = conv_trans(
            hidden_size, deconv_chns, kernel_size=up_kernel_size, stride=up_kernel_size
        )
        self.conv3d_transpose_1 = conv_trans(
            in_channels=deconv_chns,
            out_channels=out_channels,
            kernel_size=up_kernel_size,
            stride=up_kernel_size,
        )

        self.decoder = ViTDecoder(
            image_size=img_size,
            decoder_embed_dim=hidden_size,
            out_channels=in_channels,
            patch_size=patch_size,
        )
        self.masked_embed = nn.Parameter(torch.zeros(1, hidden_size))
        self.embed_dim = hidden_size

    def mask_model(self, x, mask):
        b, c, h, w, d = x.size()
        mask = mask.bool()
        idx = torch.nonzero(mask.view(-1), as_tuple=False).squeeze()
        x_flat = x.view(-1, c)
        x_flat[idx] = self.masked_embed.expand(idx.size(0), -1).to(x.dtype)
        x = x_flat.view(b, c, h, w, d)
        return x.flatten(2).transpose(-1, -2)

    def forward(self, x, mask=None, return_seg=False):
        """
        Args:
            x: input tensor must have isotropic spatial dimensions,
                such as ``[batch_size, channels, sp_size, sp_size[, sp_size]]``.
        """
        spatial_size = x.shape[2:]
        x = self.patch_embedding(x)  # [b,768,8,8,8]
        if mask is not None:
            x = self.mask_model(
                rearrange(x, "n (h w d) c -> n c h w d", d=8, h=8, w=8), mask
            )
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        embedding = x

        x = x.transpose(1, 2)
        d = [s // p for s, p in zip(spatial_size, self.patch_size)]
        x = torch.reshape(x, [x.shape[0], x.shape[1], *d])
        x = self.conv3d_transpose(x)
        seg_result = self.conv3d_transpose_1(x)
        pred = self.decoder(embedding)

        if return_seg:
            return seg_result
        else:
            return {
                "recon": pred,
                "patch": embedding,
                "seg": seg_result,
            }
