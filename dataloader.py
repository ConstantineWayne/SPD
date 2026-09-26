
import os
from torch.utils.data import ConcatDataset
from data_loader.dataset import *
import bisect
""" DATASET Structure
eventscape_dataset
- train
  -- Town01
      -- sequence0
          -- rgb
          -- events
          -- depth
  -- Town02
  -- Town03
- val
  -- Town05
- test
  -- Town05
"""
class ConcatDatasetCustom(ConcatDataset):
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx
        
def concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=None, proba_pause_when_running=0.0, proba_pause_when_paused=0.0, step_size=1,
                           clip_distance=100.0, every_x_rgb_frame=1, normalize=True, scale_factor=1.0,
                           use_phased_arch=False, baseline=False, reg_factor=3.7, loss_composition=None, dataset_idx_flag = False,noise=None,severity=None):
    print("entered concatenate_subfolders")
    subfolders = os.listdir(base_folder)
    print('Found {} samples in {}'.format(len(subfolders), base_folder))

    train_datasets = []

    for dataset_name in subfolders:
        train_datasets.append(eval(dataset_type)(base_folder=join(base_folder, dataset_name), event_folder=event_folder,
                                                 depth_folder=depth_folder,
                                                 frame_folder=frame_folder, 
                                                 sequence_length=sequence_length,
                                                 transform=transform,
                                                 proba_pause_when_running=proba_pause_when_running,
                                                 proba_pause_when_paused=proba_pause_when_paused,
                                                 step_size=step_size,
                                                 clip_distance=clip_distance,
                                                 every_x_rgb_frame=every_x_rgb_frame,
                                                 normalize=normalize,
                                                 scale_factor=scale_factor,
                                                 use_phased_arch=use_phased_arch,
                                                 baseline=baseline,loss_composition=loss_composition,
                                                 reg_factor=reg_factor,
                                                 noise=noise,
                                                 severity=severity))
          
    if dataset_idx_flag == False:
        concat_dataset = ConcatDataset(train_datasets)
    elif dataset_idx_flag == True:
        concat_dataset = ConcatDatasetCustom(train_datasets)
    return concat_dataset



            
def build_dataset(set, base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform, proba_pause_when_running, proba_pause_when_paused, step_size,
                           clip_distance, every_x_rgb_frame, normalize, scale_factor,
                           use_phased_arch, baseline, reg_factor, loss_composition,
                           dataset_idx_flag,noise=None,severity=None):
   if(set=='train'):
     dataset = concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=transform, proba_pause_when_running=proba_pause_when_running, proba_pause_when_paused=proba_pause_when_paused, step_size=step_size,
                           clip_distance=clip_distance, every_x_rgb_frame=every_x_rgb_frame, normalize=normalize, scale_factor=scale_factor,
                           use_phased_arch=use_phased_arch, baseline=baseline, reg_factor=reg_factor, loss_composition=loss_composition,
                           dataset_idx_flag = dataset_idx_flag
                                           )
   elif(set=='validation'):
     dataset = concatenate_subfolders(base_folder, dataset_type, event_folder, depth_folder, frame_folder, sequence_length,
                           transform=transform, proba_pause_when_running=proba_pause_when_running, proba_pause_when_paused=proba_pause_when_paused, step_size=step_size,
                           clip_distance=clip_distance, every_x_rgb_frame=every_x_rgb_frame, normalize=normalize, scale_factor=scale_factor,
                           use_phased_arch=use_phased_arch, baseline=baseline, reg_factor=reg_factor, loss_composition=loss_composition,
                            dataset_idx_flag = dataset_idx_flag,noise=noise,severity=severity
                                           )

   return dataset