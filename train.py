import os
from datetime import datetime
from os.path import join
import numpy as np
import json
import logging
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from model.networks import SPD
from trainer.trainer import TransformerTrainer
from utils.data_augmentation import Compose, RandomRotationFlip, RandomCrop, CenterCrop
from dataloader import build_dataset
from utils import *

logging.basicConfig(level=logging.INFO, format='')

def main(config, resume, initial_checkpoint=None):
    train_logger = None

    L = config['trainer']['sequence_length']
    assert (L > 0)

    dataset_type, base_folder, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    scale_factor = {}
    step_size = {}
    every_x_rgb_frame = {}
    reg_factor = {}
    baseline = {}
    clip_distance = {}
    dataset_idx_flag = {}

    use_phased_arch = config['use_phased_arch']

    for split in ['train', 'validation']:
        dataset_type[split] = config['data_loader'][split]['type']
        base_folder[split] = join(config['data_loader'][split]['base_folder'])
        event_folder[split] = config['data_loader'][split]['event_folder']
        depth_folder[split] = config['data_loader'][split]['depth_folder']
        frame_folder[split] = config['data_loader'][split]['frame_folder']
        proba_pause_when_running[split] = config['data_loader'][split]['proba_pause_when_running']
        proba_pause_when_paused[split] = config['data_loader'][split]['proba_pause_when_paused']
        scale_factor[split] = config['data_loader'][split]['scale_factor']
        dataset_idx_flag[split] = config['data_loader'][split]['dataset_idx_flag']

        try:
            step_size[split] = config['data_loader'][split]['step_size']
        except KeyError:
            step_size[split] = 1

        try:
            clip_distance[split] = config['data_loader'][split]['clip_distance']
        except KeyError:
            clip_distance[split] = 100.0

        try:
            every_x_rgb_frame[split] = config['data_loader'][split]['every_x_rgb_frame']
        except KeyError:
            every_x_rgb_frame[split] = 1

        try:
            baseline[split] = config['data_loader'][split]['baseline']
        except KeyError:
            baseline[split] = False

        try:
            reg_factor[split] = config['data_loader'][split]['reg_factor']
        except KeyError:
            reg_factor[split] = False

    np.random.seed(21)

    loss_composition = config['trainer']['loss_composition']
    normalize = config['data_loader'].get('normalize', True)

    train_dataset = build_dataset(set="train", base_folder=base_folder['train'],
                                  dataset_type=dataset_type['train'],
                                  event_folder=event_folder['train'],
                                  depth_folder=depth_folder['train'],
                                  frame_folder=frame_folder['train'],
                                  sequence_length=L,
                                  transform=Compose([RandomRotationFlip(0.0, 0.5, 0.0),
                                                     RandomCrop(224)]),
                                  proba_pause_when_running=proba_pause_when_running['train'],
                                  proba_pause_when_paused=proba_pause_when_paused['train'],
                                  step_size=step_size['train'],
                                  clip_distance=clip_distance['train'],
                                  every_x_rgb_frame=every_x_rgb_frame['train'],
                                  normalize=normalize,
                                  scale_factor=scale_factor['train'],
                                  use_phased_arch=use_phased_arch,
                                  baseline=baseline['train'],
                                  dataset_idx_flag=dataset_idx_flag['train'],
                                  loss_composition=loss_composition,
                                  reg_factor=reg_factor['train'],
                                  )

    val_dataset = build_dataset(set="validation", base_folder=base_folder['validation'],
                                dataset_type=dataset_type['validation'],
                                event_folder=event_folder['validation'],
                                depth_folder=depth_folder['validation'],
                                frame_folder=frame_folder['validation'],
                                sequence_length=L,
                                transform=CenterCrop(224),
                                proba_pause_when_running=proba_pause_when_running['validation'],
                                proba_pause_when_paused=proba_pause_when_paused['validation'],
                                step_size=step_size['validation'],
                                clip_distance=clip_distance['validation'],
                                every_x_rgb_frame=every_x_rgb_frame['validation'],
                                normalize=normalize,
                                scale_factor=scale_factor['validation'],
                                use_phased_arch=use_phased_arch,
                                baseline=baseline['validation'],
                                dataset_idx_flag=dataset_idx_flag['validation'],
                                loss_composition=loss_composition,
                                reg_factor=reg_factor['validation'],
                                )

    kwargs = {'num_workers': config['data_loader']['num_workers'],
              'pin_memory': config['data_loader']['pin_memory']} if config['cuda'] else {}
    data_loader = DataLoader(train_dataset, batch_size=config['data_loader']['batch_size'],
                             shuffle=config['data_loader']['shuffle'], **kwargs)

    valid_data_loader = DataLoader(val_dataset, batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'], **kwargs)
    config['model']['gpu'] = config['gpu']

    config["lr_scheduler"]["steps_per_epoch"] = int(len(data_loader))

    torch.manual_seed(21)

    model = SPD(img_size_s1=config['model']['img_size_s1'], img_size_s2=config['model']['img_size_s2'],
                    model_scale=config['model']['model_scale'], pre_load=config['model']['preload'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    loss = eval(config['loss']['type'])
    loss_params = config['loss']['config'] if 'config' in config['loss'] else None
    print("Using %s with config %s" % (config['loss']['type'], config['loss']['config']))

    trainer = TransformerTrainer(model, loss, loss_params,
                                 resume=resume,
                                 config=config,
                                 data_loader=data_loader,
                                 valid_data_loader=valid_data_loader,
                                 train_logger=train_logger)

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Monocular Depth Prediction')
    parser.add_argument('-c', '--config', default="./configs/default.json", type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-i', '--initial_checkpoint',
                        default=None, type=str,
                        help='path to the checkpoint with which to initialize the model weights (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        if args.initial_checkpoint is not None:
            logger.warning(
                'Warning: --initial_checkpoint overriden by --resume')
        config = torch.load(args.resume)['config']
    if args.config is not None:

        with open(args.config, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(config['trainer']['save_dir'], f"{config['name']}_{current_time}")

        if args.resume is None:
            assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume, args.initial_checkpoint)
