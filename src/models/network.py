# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

from . import get_encoder, get_decoder, get_classification
import tensorflow as tf
import models


class PCBSegNet:
    def __init__(self, opt):
        self.model_type = opt["model_type"]
        self.image_height = opt["datasets"]["train"]["img_size_h"]
        self.image_width = opt["datasets"]["train"]["img_size_w"]
        self.num_classes = opt["train"]["num_classes"] + 1

    def build(self):
        encoder = get_encoder(self.image_height, self.image_width)
        model = get_decoder(encoder, self.num_classes)
        return model


class PCBClassNet:
    def __init__(self, opt):
        self.model_type = opt["model_type"]
        self.image_height = opt["datasets"]["train"]["img_size_h"]
        self.image_width = opt["datasets"]["train"]["img_size_w"]
        self.num_classes = opt["train"]["num_classes"]

    def build(self):
        encoder = get_encoder(self.image_height, self.image_width)
        model = get_classification(encoder, self.num_classes)
        return model


def get_model(opt):
    if opt["model_type"] == "SegmentationModel":
        seg_model = PCBSegNet(opt)
        model = seg_model.build()
        optimizer = getattr(tf.keras.optimizers, opt["train"]["optim"]["type"])(
            learning_rate=opt["train"]["optim"]["lr"],
            beta_1=opt["train"]["optim"]["betas"][0],
            beta_2=opt["train"]["optim"]["betas"][1],
        )
        loss = getattr(models, opt["train"]["loss"]["type"])
        metrics = [
            getattr(models, opt["train"]["metric"][item]["type"])
            for item in opt["train"]["metric"]
        ]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    elif opt["model_type"] == "ClassificationModel":
        class_model = PCBClassNet(opt)
        model = class_model.build()
        optimizer = getattr(tf.keras.optimizers, opt["train"]["optim"]["type"])(
            learning_rate=opt["train"]["optim"]["lr"],
            beta_1=opt["train"]["optim"]["betas"][0],
            beta_2=opt["train"]["optim"]["betas"][1],
        )
        # tf.keras.metrics.Precision()
        metrics = [
            getattr(tf.keras.metrics, opt["train"]["metric"][item]["type"])()
            for item in opt["train"]["metric"]
        ] + ["accuracy"]
        model.compile(
            optimizer=optimizer, loss=opt["train"]["loss"]["type"], metrics=metrics
        )
        return model

    else:
        assert (
            False
        ), f"Found model type as {opt['model_type']} but it should be one of SegmentationModel/ClassificationModel"
