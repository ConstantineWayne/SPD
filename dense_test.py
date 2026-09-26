import os
import json
import logging
import argparse
import matplotlib.pyplot as plt
import torch
import cv2

from metric import *
import torch.nn.functional as F

from model.networks import SPD

from utils.data_augmentation import CenterCrop
from os.path import join
import numpy as np
from dataloader import build_dataset

logging.basicConfig(level=logging.INFO, format='')


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def eval_metrics(output, target):
    metrics = [mse, abs_rel_diff, scale_invariant_error, median_error, mean_error, rms_linear]
    acc_metrics = np.zeros(len(metrics))
    output = output.cpu().data.numpy()
    target = target.cpu().data.numpy()
    for i, metric in enumerate(metrics):
        acc_metrics[i] += metric(output, target)
    return acc_metrics

def make_heatmap(img, cmap='magma'):
    img = np.squeeze(img)
    img = np.nan_to_num(img)

    if img.ndim != 2:
        raise ValueError(f"make_heatmap expects [H,W], got {img.shape}")

    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)

    color = plt.get_cmap(cmap)(img)
    color = (color[:, :, :3] * 255).astype(np.uint8)
    color = color[..., ::-1]  # RGB -> BGR
    return np.ascontiguousarray(color)


def make_depth_colormap(img, cmap='turbo'):
    img = np.squeeze(img)
    img = np.nan_to_num(img)

    if img.ndim != 2:
        raise ValueError(f"make_depth_colormap expects [H,W], got {img.shape}")

    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)

    color = plt.get_cmap(cmap)(img)
    color = (color[:, :, :3] * 255).astype(np.uint8)
    color = color[..., ::-1]  # RGB -> BGR
    return np.ascontiguousarray(color)


def make_colormap(img, cmap='magma'):
    img = np.nan_to_num(img)
    vmin, vmax = np.percentile(img, 2), np.percentile(img, 98)
    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)

    color = plt.get_cmap(cmap)(img)
    color = (color[:, :, :3] * 255).astype(np.uint8)
    color = color[..., ::-1]  # RGB to BGR for cv2
    return color


def main(config, initial_checkpoint):
    train_logger = None

    L = 1

    dataset_type, base_folder, event_folder, depth_folder, frame_folder = {}, {}, {}, {}, {}
    proba_pause_when_running, proba_pause_when_paused = {}, {}
    step_size = {}

    scale_factor = {}
    every_x_rgb_frame = {}
    baseline = {}
    reg_factor = {}

    clip_distance = {}

    use_phased_arch = config['use_phased_arch']

    dataset_type['test'] = config['data_loader']['test']['type']

    event_folder['test'] = config['data_loader']['test']['event_folder']
    depth_folder['test'] = config['data_loader']['test']['depth_folder']
    frame_folder['test'] = config['data_loader']['test']['frame_folder']
    proba_pause_when_running['test'] = config['data_loader']['test']['proba_pause_when_running']
    proba_pause_when_paused['test'] = config['data_loader']['test']['proba_pause_when_paused']
    scale_factor['test'] = config['data_loader']['test']['scale_factor']

    base_folder['test'] = join(config['data_loader']['test']['base_folder'])
    print(base_folder['test'])

    try:
        step_size['test'] = 1
    except KeyError:
        step_size['test'] = 1

    try:
        clip_distance['test'] = config['data_loader']['test']['clip_distance']
    except KeyError:
        clip_distance['test'] = 100.0
        print("Clip distance not loaded properly!")

    try:
        every_x_rgb_frame['test'] = config['data_loader']['test']['every_x_rgb_frame']
    except KeyError:
        every_x_rgb_frame['test'] = 1
        print("Every_x_rgb_frame not loaded properly!")

    try:
        baseline['test'] = config['data_loader']['test']['baseline']
    except KeyError:
        baseline['test'] = False
        print("Baseline not loaded properly!")

    try:
        reg_factor['test'] = config['data_loader']['test']['reg_factor']
    except KeyError:
        reg_factor['test'] = False

    normalize = config['data_loader'].get('normalize', True)
    loss_composition = config['trainer']['loss_composition']


    noise = None
    severity = 0

    test_dataset = build_dataset(set="validation", base_folder=base_folder['test'],
                                 dataset_type=dataset_type['test'],
                                 event_folder=event_folder['test'],
                                 depth_folder=depth_folder['test'],
                                 frame_folder=frame_folder['test'],
                                 sequence_length=L,
                                 transform=CenterCrop(224),
                                 proba_pause_when_running=proba_pause_when_running['test'],
                                 proba_pause_when_paused=proba_pause_when_paused['test'],
                                 step_size=step_size['test'],
                                 clip_distance=clip_distance['test'],
                                 every_x_rgb_frame=every_x_rgb_frame['test'],
                                 normalize=normalize,
                                 scale_factor=scale_factor['test'],
                                 use_phased_arch=use_phased_arch,
                                 baseline=baseline['test'],
                                 dataset_idx_flag=False,
                                 loss_composition=loss_composition,
                                 reg_factor=reg_factor['test'],
                                 noise=noise,
                                 severity=severity
                                 )

    config['model']['gpu'] = config['gpu']

    model = SPD(img_size_s1=config['model']['img_size_s1'], img_size_s2=config['model']['img_size_s2'],
                    model_scale=config['model']['model_scale'], pre_load=config['model']['preload'])

    if initial_checkpoint is not None:
        print('Loading initial model weights from: {}'.format(initial_checkpoint))
        checkpoint = torch.load(initial_checkpoint)
        model.load_state_dict(checkpoint['state_dict'])

    gpu = torch.device('cuda:' + str(config['gpu']))
    model.to(gpu)

    model.eval()

    N = len(test_dataset)

    print('N: ' + str(N))


    with torch.no_grad():
        idx = 0

        while idx < N:
            item = test_dataset[idx]

            events = item[0]['events'].unsqueeze(dim=0)
            rgb = item[0]['image'].unsqueeze(dim=0)

            target = item[0]['depth_image'].cpu().numpy()

            events = events.float().to(gpu)
            rgb = rgb.float().to(gpu)

            pred_depth_out = model(rgb, events)

            if isinstance(pred_depth_out, tuple):
                pred_depth_out = pred_depth_out[0]

            # tensor version
            pred_depth_tensor = pred_depth_out.detach().float()

            if pred_depth_tensor.dim() == 3:
                pred_depth_tensor = pred_depth_tensor.unsqueeze(0)

            # numpy version for saving
            pred_depth = pred_depth_tensor.squeeze().cpu().numpy()
            gt = target.squeeze()





            target_tensor = item[0]['depth_image'].unsqueeze(0).float().to(gpu)

            if target_tensor.dim() == 3:
                target_tensor = target_tensor.unsqueeze(0)


            # ===== predicted depth colormap =====
            pred_depth_np = pred_depth_tensor.squeeze().cpu().numpy()

            # ===== unified entropy from local depth distribution =====
            pred = pred_depth_tensor.clone()

            # normalize
            pred = (pred - pred.amin(dim=[2, 3], keepdim=True)) / (
                    pred.amax(dim=[2, 3], keepdim=True) - pred.amin(dim=[2, 3], keepdim=True) + 1e-8
            )
            folder_name = os.path.basename(base_folder['test'])
            base_folder_ = os.path.join(
                args.base_save_folder,
                folder_name
            )

            pred_depth_folder = os.path.join(base_folder_, 'pred_depth')
            pred_depth_img_folder = os.path.join(base_folder_, 'pred_depth_img')

            gt_folder = os.path.join(base_folder_, 'gt')

            error_folder = os.path.join(base_folder_, 'error')
            os.makedirs(error_folder, exist_ok=True)

            os.makedirs(pred_depth_folder, exist_ok=True)
            os.makedirs(pred_depth_img_folder, exist_ok=True)
            os.makedirs(gt_folder, exist_ok=True)

            if 'img_path' in item[0]:
                name = os.path.splitext(os.path.basename(item[0]['img_path']))[0]
            else:
                name = '{:010d}'.format(idx)

            np.save(join(pred_depth_folder, name + '.npy'), pred_depth)
            np.save(join(gt_folder, name + '.npy'), target)

            print(idx, name)
            idx += 1
if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(
        description='Learning DVS Image Reconstruction')
    parser.add_argument('--path_to_model', type=str,
                        help='path to the model weights',
                        default='')
    parser.add_argument('--config', type=str,
                        help='path to config. If not specified, config from model folder is taken',
                        default='')

    parser.add_argument('--base_save_folder', type=str,
                        help='path to the model weights',
                        default='')

    args = parser.parse_args()

    if args.config is None:
        head_tail = os.path.split(args.path_to_model)
        config = json.load(open(os.path.join(head_tail[0], 'config.json')))
    else:
        config = json.load(open(args.config))

    main(config, args.path_to_model)