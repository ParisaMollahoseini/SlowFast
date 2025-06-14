#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import MODEL_REGISTRY, build_model  # noqa
from .contrastive import ContrastiveModel  # noqa
from .custom_video_model_builder import *  # noqa
from .masked import MaskMViT  # noqa
from .video_model_builder import MViT, ResNet, SlowFast  # noqa
from .clip_video_model import *
from .clip_image_model import *
from .temporalclip_video_model import TemporalClipVideo
from .temporalclip_promptpool import TemporalClipPromptPool  # noqa


try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVX3D,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
