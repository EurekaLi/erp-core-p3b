"""项目内EEGNet架构适配和严格加载CBraMod；只返回logits，不调用作者训练入口。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _temporal_conv(
    in_channels: int, out_channels: int, kernel: int, groups: int = 1
) -> nn.Sequential:
    """显式SAME补零维持时间长度；偶数核两侧差1点，与卷积输出几何匹配。"""
    return nn.Sequential(
        nn.ZeroPad2d(((kernel - 1) // 2, kernel // 2, 0, 0)),
        nn.Conv2d(in_channels, out_channels, (1, kernel), groups=groups, bias=False),
    )


class EEGNet(nn.Module):
    """时间卷积→depthwise空间滤波→separable卷积的PyTorch适配，不宣称逐位复现Keras。

    架构出处：https://github.com/vlawhern/arl-eegmodels ，Lawhern et al., 2018。
    输入为(batch,1,EEG,time)，256Hz下采用128点时间核，池化等参数在配置中显式记录。
    """

    def __init__(self, channels: int, samples: int, config: dict[str, Any]):
        super().__init__()
        f1, d, f2 = (int(config[key]) for key in ("F1", "D", "F2"))
        pools = config["pool_sizes"]
        if min(channels, samples, f1, d, f2, *pools) < 1 or not 0 <= config["dropout"] < 1:
            raise ValueError("EEGNet维度/池化/dropout参数无效")
        final_samples = samples // pools[0] // pools[1]
        if final_samples < 1:
            raise ValueError("池化后没有时间点")
        self.temporal = _temporal_conv(1, f1, int(config["temporal_kernel"]))
        # Keras默认BN保留系数.99对应PyTorch更新系数.01，不直接照抄同名参数。
        self.bn1 = nn.BatchNorm2d(f1, eps=1e-3, momentum=0.01)
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(f1 * d, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, pools[0])),
            nn.Dropout(config["dropout"]),
        )
        self.block2 = nn.Sequential(
            _temporal_conv(f1 * d, f1 * d, int(config["separable_kernel"]), groups=f1 * d),
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2, eps=1e-3, momentum=0.01),
            nn.ELU(),
            nn.AvgPool2d((1, pools[1])),
            nn.Dropout(config["dropout"]),
        )
        self.classifier = nn.Linear(f2 * final_samples, 2)
        self.config = config
        self.expected_shape = (1, channels, samples)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != self.expected_shape:
            raise ValueError("EEGNet输入维度不符合原生缓存")
        features = self.block2(self.block1(self.spatial(self.bn1(self.temporal(inputs)))))
        return self.classifier(features.flatten(start_dim=1))

    @torch.no_grad()
    def constrain_weights(self) -> None:
        """每次优化后投影空间滤波及分类器范数；保留EEGNet常用max-norm约束。"""
        for parameter, axes, maximum in (
            (self.spatial.weight, (1, 2, 3), self.config["spatial_max_norm"]),
            (self.classifier.weight, (1,), self.config["classifier_max_norm"]),
        ):
            norms = torch.linalg.vector_norm(parameter, dim=axes, keepdim=True)
            parameter.mul_((float(maximum) / norms.clamp_min(1e-12)).clamp(max=1))


def load_cbramod_encoder(provenance: dict[str, Any]) -> nn.Module:
    """完整strict载入预训练网络后移除重建头，不复用08临时更新过的实例。"""
    source = Path(provenance["source_root"]).resolve()
    existing = sys.modules.get("models")
    if existing is not None and str(source / "models") not in list(
        getattr(existing, "__path__", [])
    ):
        raise RuntimeError("进程中已有其他models包，请使用独立CLI进程")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("models.cbramod")
    if Path(module.__file__).resolve() != source / "models/cbramod.py":
        raise RuntimeError("CBraMod导入位置与固定资源不同")
    model = module.CBraMod()
    state = torch.load(provenance["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.proj_out = nn.Identity()
    return model


class CBraClassifier(nn.Module):
    """同一线性头分别做冻结表示对照和全量微调，避免同时改变分类头结构。"""

    def __init__(self, encoder: nn.Module, channels: int, *, frozen: bool):
        super().__init__()
        self.encoder, self.frozen = encoder, frozen
        for name, parameter in self.encoder.named_parameters():
            parameter.requires_grad_(not frozen and not name.endswith("mask_encoding"))
        self.head = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(channels * 200, 2))
        self.channels = channels

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结不只指requires_grad=False：编码器dropout也必须关闭，固定特征才有意义。
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != (self.channels, 1, 200):
            raise ValueError("CBraMod输入必须为30通道/一个200点patch的当前协议")
        if self.frozen:
            with torch.no_grad():
                features = self.encoder(inputs)
        else:
            features = self.encoder(inputs)
        return self.head(features)


def build_model(
    name: str,
    channels: int,
    samples: int,
    config: dict[str, Any],
    provenance: dict[str, Any] | None,
) -> nn.Module:
    if name == "eegnet":
        return EEGNet(channels, samples, config["eegnet"])
    if name not in ("cbramod_linear", "cbramod_finetune") or provenance is None:
        raise ValueError("模型名称/预训练资源无效")
    return CBraClassifier(
        load_cbramod_encoder(provenance), channels, frozen=name == "cbramod_linear"
    )
