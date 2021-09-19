# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build as build_detr
from .gsrtr import build as build_gsrtr


def build_model(args):
    if args.dataset_file == "swig":
        return build_gsrtr(args)
    else:
        return build_detr(args)
