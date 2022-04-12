import os
from typing import Any, Mapping

import torch

import numpy as np


from learning.sgmodel import Trainer
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from ruamel.yaml import YAML

from callbacks import SegmentationVisualizationCallback, MlflowCallback
from data.sequoia import SequoiaDatasetInterface

from loss import LOSSES as LOSSES_DICT
from metrics import metrics_factory
from models import MODELS as MODELS_DICT
from utils.utils import MLRun
from utils.grid import make_grid

torch.manual_seed(42)
np.random.seed(42)
EXP_DIR = 'mlflow'
logger = get_logger(__name__)


def parse_params(params: dict) -> (dict, dict, dict, list):
    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params'].pop('loss')
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    train_metrics = metrics_factory(params['train_metrics'])
    test_metrics = metrics_factory(params['test_metrics'])

    # dataset params
    dataset_params = params['dataset']

    train_params = {
        "greater_metric_to_watch_is_better": True,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss_logging_items_names": ["loss"],
        "loss": loss,
        **input_train_params
    }

    test_params = {
        "test_metrics": test_metrics,
    }

    # early stopping
    early_stop = [EarlyStop(Phase.VALIDATION_EPOCH_END, **params['early_stopping']['params'])] \
        if params['early_stopping']['enabled'] else []

    return train_params, test_params, dataset_params, early_stop


def init_model(sg_model: Trainer, params: Mapping, phase: str, mlflowclient: MLRun):
    # init model
    model_params = params['model']
    if model_params['name'] in MODELS_DICT.keys():
        model = MODELS_DICT[model_params['name']](**model_params['params'],
                                                  in_chn=len(params['dataset']['channels']),
                                                  out_chn=params['dataset']['num_classes']
                                                  )
    else:
        model = model_params['name']

    if not (phase == 'train'):
        checkpoint_folder = mlflowclient.run.info.artifact_uri + '/SG'
    else:
        checkpoint_folder = None
    sg_model.build_model(model,
                         source_ckpt_folder_name=checkpoint_folder)


def experiment(params: Mapping):
    exp = params['experiment']
    exp_name = exp['name']
    description = exp['description']
    phase = exp['phase']

    params = params['parameters']
    train_params, test_params, dataset_params, early_stop = parse_params(params)

    # Mlflow
    if not (phase == 'train'):
        exp_hash = exp['exp_hash']
    else:
        exp_hash = None
    mlclient = MLRun(exp_name, description, exp_hash)

    sg_model = Trainer(experiment_name='SG', ckpt_root_dir=mlclient.run.info.artifact_uri)
    dataset = SequoiaDatasetInterface(dataset_params)
    sg_model.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])

    init_model(sg_model, params, phase, mlclient)

    if phase == 'train':
        # Callbacks
        cbcks = [
            MlflowCallback(Phase.TRAIN_EPOCH_END, freq=1, client=mlclient, params=train_params),
            MlflowCallback(Phase.VALIDATION_EPOCH_END, freq=1, client=mlclient),
            SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                              freq=5,
                                              batch_idxs=[0, len(sg_model.train_loader) - 1],
                                              last_img_idx_in_batch=4,
                                              num_classes=len(dataset.classes),
                                              undo_preprocessing=dataset.undo_preprocess),
            *early_stop
        ]
        train_params["phase_callbacks"] = cbcks

        sg_model.train(train_params)

    test_metrics = sg_model.test(**test_params)

    # log test metrics
    mlclient.log_metrics(test_metrics)


if __name__ == '__main__':
    param_path = 'parameters.yaml'
    with open(param_path, 'r') as param_stream:
        grids = YAML().load(param_stream)

    logger.info(f'Loaded parameters from {param_path}')
    experiments = make_grid(grids)
    logger.info(f'Found {len(experiments)} experiments')

    for i, params in enumerate(experiments):
        try:
            logger.info(f'Running experiment {i} out of {len(experiments)}')
            experiment(params)
        except Exception as e:
            logger.error(f'Experiment {i} failed with error {e}')
            raise e
