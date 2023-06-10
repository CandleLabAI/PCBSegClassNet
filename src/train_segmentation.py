# ------------------------------------------------------------------------
# Copyright (c) 2023 CandleLabAI. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Training code for segmentation with given configuration params

usage:

python train_segmentation.py -opt cfs/pscn_seg.yml -epoch 100
"""

import argparse
import logging
import tensorflow as tf
import numpy as np

from utils import dict2str, parse, get_msg
from models import get_model
from data import get_data

def parse_config():
    """
    Helper function to parse config
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt",
        type=str,
        required=True,
        help="Path to option YAML file."
    )
    parser.add_argument("-epoch",
                        type=int,
                        default=1,
                        help="number of epochs.")
    args = parser.parse_args()

    opt = parse(args.opt)
    opt["train"]["total_epochs"] = args.epoch
    return opt

def init_log(opt):
    """
    Helper function for logging
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(opt["path"]["log_file"], mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    logger.info(get_msg())
    logger.info(dict2str(opt))

def main():
    """
    main function for segmentation training
    """
    opt = parse_config()
    init_log(opt)

    # get dataloaders
    train_dataset, val_dataset = get_data(opt)
    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(train_dataset)} batches for training")
    logger.info(f"Found {len(val_dataset)} batches for validation")

    # get model
    model = get_model(opt)
    logger.info(model.summary())

    callbacks = []
    if "modelcheckpoint" in opt["train"]["callbacks"]:
        logger.info("Using modelcheckpoint callback")
        callbacks.append(
            getattr(
                tf.keras.callbacks, opt["train"]["callbacks"]["modelcheckpoint"]["type"]
            )(
                filepath=opt["path"]["checkpoint_network"],
                monitor=opt["train"]["callbacks"]["modelcheckpoint"]["monitor"],
                mode=opt["train"]["callbacks"]["modelcheckpoint"]["mode"],
                verbose=opt["train"]["callbacks"]["modelcheckpoint"]["verbose"],
                save_best_only=opt["train"]["callbacks"]["modelcheckpoint"][
                    "save_best_only"
                ],
                save_weights_only=opt["train"]["callbacks"]["modelcheckpoint"][
                    "save_weights_only"
                ],
            )
        )

    if "reducelronplateau" in opt["train"]["callbacks"]:
        logger.info("Using reducelronplateau callback")
        callbacks.append(
            getattr(
                tf.keras.callbacks,
                opt["train"]["callbacks"]["reducelronplateau"]["type"],
            )(
                monitor=opt["train"]["callbacks"]["reducelronplateau"]["monitor"],
                mode=opt["train"]["callbacks"]["reducelronplateau"]["mode"],
                verbose=opt["train"]["callbacks"]["reducelronplateau"]["verbose"],
                factor=opt["train"]["callbacks"]["reducelronplateau"]["factor"],
                patience=opt["train"]["callbacks"]["reducelronplateau"]["patience"],
                min_lr=opt["train"]["callbacks"]["reducelronplateau"]["min_lr"],
            )
        )

    # training model
    if opt["train"]["total_epochs"] > 0:
        logger.info(f"Training for {opt['train']['total_epochs']} epochs")
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=opt["train"]["total_epochs"],
            verbose=1,
            callbacks=callbacks,
        )

    # testing the model
    # load best saved weights
    logger.info("Testing model with best weights")
    model.load_weights(opt["path"]["checkpoint_network"])
    model.evaluate(val_dataset)


if __name__ == "__main__":
    main()
