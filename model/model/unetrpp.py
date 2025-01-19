import torch
import math
import einops
from torch import nn
from typing import Tuple, Union, Sequence
from timm.models.layers import trunc_normal_
from monai.networks.layers.utils import get_norm_layer
from monai.networks.blocks.dynunet_block import (
    UnetResBlock,
    UnetOutBlock,
    get_conv_layer,
)
from monai.networks.blocks.unetr_block import UnetrUpBlock


class TransformerBlock(nn.Module):
    """
    A transformer block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        proj_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.epa_block = EPA(
            input_size=input_size,
            hidden_size=hidden_size,
            proj_size=proj_size,
            num_heads=num_heads,
            channel_attn_drop=dropout_rate,
            spatial_attn_drop=dropout_rate,
        )
        self.conv51 = UnetResBlock(
            3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch"
        )
        self.conv8 = nn.Sequential(
            nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1)
        )

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape
        # print(x.shape)
        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.epa_block(self.norm(x))

        # (B, C, H, W, D)
        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)

        return attn_skip + self.conv8(self.conv51(attn_skip))


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class EPA(nn.Module):
    """
    Efficient Paired Attention Block, based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        proj_size,
        num_heads=4,
        qkv_bias=False,
        channel_attn_drop=0.1,
        spatial_attn_drop=0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature2 = nn.Parameter(torch.ones(num_heads, 1, 1))

        # qkvv are 4 linear layers (query_shared, key_shared, value_spatial, value_channel)
        self.qkvv = nn.Linear(hidden_size, hidden_size * 4, bias=qkv_bias)

        # E and F are projection matrices with shared weights used in spatial attention module to project
        # keys and values from HWD-dimension to P-dimension
        self.EF = nn.Parameter(init_(torch.zeros(input_size, proj_size)))

        self.attn_drop = nn.Dropout(channel_attn_drop)
        self.attn_drop_2 = nn.Dropout(spatial_attn_drop)

    def forward(self, x):
        B, N, C = x.shape

        qkvv = self.qkvv(x).reshape(B, N, 4, self.num_heads, C // self.num_heads)
        qkvv = qkvv.permute(2, 0, 3, 1, 4)
        q_shared, k_shared, v_CA, v_SA = qkvv[0], qkvv[1], qkvv[2], qkvv[3]

        q_shared = q_shared.transpose(-2, -1)
        k_shared = k_shared.transpose(-2, -1)
        v_CA = v_CA.transpose(-2, -1)
        v_SA = v_SA.transpose(-2, -1)

        proj_e_f = lambda args: torch.einsum("bhdn,nk->bhdk", *args)
        k_shared_projected, v_SA_projected = map(
            proj_e_f, zip((k_shared, v_SA), (self.EF, self.EF))
        )

        q_shared = torch.nn.functional.normalize(q_shared, dim=-1)
        k_shared = torch.nn.functional.normalize(k_shared, dim=-1)

        attn_CA = (q_shared @ k_shared.transpose(-2, -1)) * self.temperature
        attn_CA = attn_CA.softmax(dim=-1)
        attn_CA = self.attn_drop(attn_CA)
        x_CA = (attn_CA @ v_CA).permute(0, 3, 1, 2).reshape(B, N, C)

        attn_SA = (
            q_shared.permute(0, 1, 3, 2) @ k_shared_projected
        ) * self.temperature2
        attn_SA = attn_SA.softmax(dim=-1)
        attn_SA = self.attn_drop_2(attn_SA)
        x_SA = (
            (attn_SA @ v_SA_projected.transpose(-2, -1))
            .permute(0, 3, 1, 2)
            .reshape(B, N, C)
        )

        return x_CA + x_SA

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"temperature", "temperature2"}


class UnetrPPEncoder(nn.Module):
    def __init__(
        self,
        input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4],
        dims=[32, 64, 128, 256],
        proj_size=[64, 64, 64, 32],
        depths=[3, 3, 3, 3],
        num_heads=4,
        spatial_dims=3,
        in_channels=4,
        dropout=0.0,
        transformer_dropout_rate=0.1,
        **kwargs,
    ):
        super().__init__()

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(
                spatial_dims,
                in_channels,
                dims[0],
                kernel_size=(4, 4, 4),
                stride=(4, 4, 4),
                dropout=dropout,
                conv_only=True,
            ),
            get_norm_layer(name=("group", {"num_groups": 1}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(
                    spatial_dims,
                    dims[i],
                    dims[i + 1],
                    kernel_size=(2, 2, 2),
                    stride=(2, 2, 2),
                    dropout=dropout,
                    conv_only=True,
                ),
                get_norm_layer(
                    name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=input_size[i],
                        hidden_size=dims[i],
                        proj_size=proj_size[i],
                        num_heads=num_heads,
                        dropout_rate=transformer_dropout_rate,
                        pos_embed=True,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
        self.masked_embed = nn.Parameter(torch.zeros(1, dims[0]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def mask_model(self, x, mask):
        b, c, h, w, d = x.size()
        mask = mask.bool()
        idx = torch.nonzero(mask.view(-1), as_tuple=False).squeeze()
        x_flat = x.view(-1, c)
        x_flat[idx] = self.masked_embed.expand(idx.size(0), -1).to(x.dtype)
        x = x_flat.view(b, c, h, w, d)
        return x

    def forward_features(self, x, mask=None):
        hidden_states = []

        x = self.downsample_layers[0](x)
        if mask is not None:
            x = self.mask_model(x, mask)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.stages[i](self.downsample_layers[i](x))
            hidden_states.append(x)
        return hidden_states

    def forward(self, x, mask=None):
        hidden_states = self.forward_features(x, mask)
        return hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        proj_size: int = 64,
        num_heads: int = 4,
        out_size: int = 0,
        depth: int = 3,
        conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block
        # (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(
                    spatial_dims,
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    norm_name=norm_name,
                )
            )
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    TransformerBlock(
                        input_size=out_size,
                        hidden_size=out_channels,
                        proj_size=proj_size,
                        num_heads=num_heads,
                        dropout_rate=0.1,
                        pos_embed=True,
                    )
                )
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class UNETRPPDecoder(nn.Module):
    def __init__(
        self,
        image_size=128,
        out_channels=4,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels
        self.decoder_pred = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.ConvTranspose3d(
                            in_channels=256, out_channels=128, kernel_size=2, stride=2
                        ),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose3d(
                            in_channels=128, out_channels=32, kernel_size=2, stride=2
                        ),
                        nn.Upsample(scale_factor=2),
                        nn.ConvTranspose3d(
                            in_channels=32,
                            out_channels=1,
                            kernel_size=2,
                            stride=2,
                        ),
                    ]
                )
                for _ in range(self.out_channels)
            ]
        )

    def forward(self, embedding):  # [b,384,4,4,4]
        # predictor projection
        x = torch.cat(
            [self.decoder_pred[i](embedding) for i in range(self.out_channels)], dim=1
        )
        return x


class MODUNETR_PP(nn.Module):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int = 128,
        feature_size: int = 16,
        hidden_size: int = 256,
        num_heads: int = 4,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        dropout_rate: float = 0.0,
        depths=[3, 3, 3, 3],
        dims=[32, 64, 128, 256],
        conv_op=nn.Conv3d,
        do_ds=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(
                f"Position embedding layer of type {pos_embed} is not supported."
            )

        self.hidden_size = hidden_size
        self.image_size = image_size
        if image_size == 128:
            self.feat_size = (4, 4, 4)
            input_size = [32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4]
            out_size = [128 * 128 * 128, 32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8]
        elif image_size == 96:
            self.feat_size = (3, 3, 3)
            input_size = [24 * 24 * 24, 12 * 12 * 12, 6 * 6 * 6, 3 * 3 * 3]
            out_size = [128 * 128 * 128, 24 * 24 * 24, 12 * 12 * 12, 6 * 6 * 6]

        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=input_size,
            dims=dims,
            depths=depths,
            num_heads=num_heads,
            in_channels=in_channels,
        )

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=out_size[-1],
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=out_size[-2],
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=out_size[-3],
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=out_size[-4],
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(
            spatial_dims=3, in_channels=feature_size, out_channels=out_channels
        )
        if self.do_ds:
            self.out2 = UnetOutBlock(
                spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels
            )
            self.out3 = UnetOutBlock(
                spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels
            )

        self.decoder = UNETRPPDecoder(
            image_size=image_size,
            out_channels=in_channels,
        )
        self.patch_size = feature_size

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in, mask=None, return_seg=False):
        hidden_states = self.unetr_pp_encoder(x_in, mask)
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        # pred = self.decoder(hidden_states[-1])
        if return_seg:
            return logits
        else:
            return {
                # "recon": pred,
                "patch": einops.rearrange(
                    hidden_states[-1], "n c h w d -> n (h w d) c"
                ),
                "seg": logits,
            }
