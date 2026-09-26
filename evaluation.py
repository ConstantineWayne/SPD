import numpy as np

import argparse
import glob
from os.path import join
import tqdm
from pprint import pprint
import matplotlib.pyplot as plt



def FLAGS():
    parser = argparse.ArgumentParser("""Event Depth Data Evaluation.""")

    # training / validation dataset
    parser.add_argument("--target_dataset", default="xxx")
    parser.add_argument("--predictions_dataset", default="xxx")
    parser.add_argument("--event_masks", default="")
    parser.add_argument("--crop_ymax", default=260, type=int)
    parser.add_argument("--debug", default=True)
    parser.add_argument("--idx", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=-1)
    parser.add_argument("--prediction_offset", type=int, default=0)
    parser.add_argument("--target_offset", type=int, default=0)
    parser.add_argument("--clip_distance", type=float, default=80.0)
    parser.add_argument("--output_folder", type=str, default=None)
    parser.add_argument("--inv", action="store_true")

    flags = parser.parse_args()

    return flags

metrics_keywords = [
    f"_abs_rel_diff",
    f"_squ_rel_diff",
    f"_RMS_linear",
    f"_RMS_log",
    f"_SILog",
    f"_mean_target_depth",
    f"_median_target_depth",
    f"_mean_prediction_depth",
    f"_median_prediction_depth",
    f"_mean_depth_error",
    f"_median_diff",
    f"_threshold_delta_1.25",
    f"_threshold_delta_1.25^2",
    f"_threshold_delta_1.25^3",
    f"_10_mean_target_depth",
    f"_10_median_target_depth",
    f"_10_mean_prediction_depth",
    f"_10_median_prediction_depth",
    f"_10_abs_rel_diff",
    f"_10_squ_rel_diff",
    f"_10_RMS_linear",
    f"_10_RMS_log",
    f"_10_SILog",
    f"_10_mean_depth_error",
    f"_10_median_diff",
    f"_10_threshold_delta_1.25",
    f"_10_threshold_delta_1.25^2",
    f"_10_threshold_delta_1.25^3",
    f"_20_abs_rel_diff",
    f"_20_squ_rel_diff",
    f"_20_RMS_linear",
    f"_20_RMS_log",
    f"_20_SILog",
    f"_20_mean_target_depth",
    f"_20_median_target_depth",
    f"_20_mean_prediction_depth",
    f"_20_median_prediction_depth",
    f"_20_mean_depth_error",
    f"_20_median_diff",
    f"_20_threshold_delta_1.25",
    f"_20_threshold_delta_1.25^2",
    f"_20_threshold_delta_1.25^3",
    f"_30_abs_rel_diff",
    f"_30_squ_rel_diff",
    f"_30_RMS_linear",
    f"_30_RMS_log",
    f"_30_SILog",
    f"_30_mean_target_depth",
    f"_30_median_target_depth",
    f"_30_mean_prediction_depth",
    f"_30_median_prediction_depth",
    f"_30_mean_depth_error",
    f"_30_median_diff",
    f"_30_threshold_delta_1.25",
    f"_30_threshold_delta_1.25^2",
    f"_30_threshold_delta_1.25^3",
]


def inv_depth_to_depth(prediction, reg_factor=3.70378):
    # convert to normalize depth (target is coming in log inverse depth)
    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32)))

    # Perform inverse depth (so now is normalized depth)
    prediction = 1/prediction
    prediction = prediction/np.amax(prediction)

    # Convert back to log depth (but now it is log  depth)
    prediction = np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32) + np.log(prediction)/reg_factor
    return prediction

def prepare_depth_data(target, prediction, clip_distance, reg_factor=3.70378):
    # normalize prediction (0 - 1)
    prediction = np.exp(reg_factor * (prediction - np.ones((prediction.shape[0], prediction.shape[1]), dtype=np.float32)))
    target = np.exp(reg_factor * (target - np.ones((target.shape[0], target.shape[1]), dtype=np.float32)))

    # Get back to the absolute values
    target *= clip_distance
    prediction *= clip_distance
    
    min_depth = np.exp(-1 * reg_factor) * clip_distance
    max_depth = clip_distance
    
    prediction[np.isinf(prediction)] = max_depth
    prediction[np.isnan(prediction)] = min_depth

    depth_mask = (np.ones_like(target)>0)
    valid_mask = np.logical_and(target > min_depth, target < max_depth)
    valid_mask = np.logical_and(depth_mask, valid_mask)
    
    

    return target, prediction, valid_mask





def add_to_metrics(idx, metrics, target_, prediction_, mask, event_frame = None, prefix="", debug = False, output_folder=None):
    if len(metrics) == 0:
        metrics = {k: 0 for k in metrics_keywords}

    eps = 1e-5

    target = target_[mask]
    prediction = prediction_[mask]

   
    # thresholds
    ratio = np.maximum((target/(prediction+eps)), (prediction/(target+eps)))

    new_metrics = {}
    new_metrics[f"{prefix}threshold_delta_1.25"] = np.mean(ratio <= 1.25)
    new_metrics[f"{prefix}threshold_delta_1.25^2"] = np.mean(ratio <= 1.25**2)
    new_metrics[f"{prefix}threshold_delta_1.25^3"] = np.mean(ratio <= 1.25**3)


    log_diff = np.log(prediction+eps) - np.log(target+eps)
    abs_diff = np.abs(target-prediction)

    new_metrics[f"{prefix}abs_rel_diff"] = (abs_diff/(target+eps)).mean()
    new_metrics[f"{prefix}squ_rel_diff"] = (abs_diff**2/(target**2+eps)).mean()
    new_metrics[f"{prefix}RMS_linear"] = np.sqrt((abs_diff**2).mean())
    new_metrics[f"{prefix}RMS_log"] = np.sqrt((log_diff**2).mean())
    new_metrics[f"{prefix}SILog"] = (log_diff**2).mean()-(log_diff.mean())**2
    new_metrics[f"{prefix}mean_target_depth"] = target.mean()
    new_metrics[f"{prefix}median_target_depth"] = np.median(target)
    new_metrics[f"{prefix}mean_prediction_depth"] = prediction.mean()
    new_metrics[f"{prefix}median_prediction_depth"] = np.median(prediction)
    new_metrics[f"{prefix}mean_depth_error"] = abs_diff.mean()
    new_metrics[f"{prefix}median_diff"] = np.abs(np.median(target) - np.median(prediction))

    for k, v in new_metrics.items():
        metrics[k] += v

    if debug:
        pprint(new_metrics)
        {print ("%s : %f" % (k, v)) for k,v in new_metrics.items()}
        fig, ax = plt.subplots(ncols=3, nrows=4)
        ax[0, 0].imshow(target_, vmin=0, vmax=200)
        ax[0, 0].set_title("target depth")
        ax[0, 1].imshow(prediction_, vmin=0, vmax=200)
        ax[0, 1].set_title("prediction depth")
        target_debug = target_.copy()
        target_debug[~mask] = 0
        ax[0, 2].imshow(target_debug, vmin=0, vmax=200)
        ax[0, 2].set_title("target depth masked")

        ax[1, 0].imshow(np.log(target_+eps),vmin=0,vmax=np.log(200))
        ax[1, 0].set_title("log target")
        ax[1, 1].imshow(np.log(prediction_+eps),vmin=0,vmax=np.log(200))
        ax[1, 1].set_title("log prediction")
        ax[1, 2].imshow(np.max(np.stack([target_ / (prediction_ + eps), prediction_ / (target_ + eps)]), axis=0))
        ax[1, 2].set_title("max ratio")

        ax[2, 0].imshow(np.abs(np.log(target_ + eps) - np.log(prediction_ + eps)))
        ax[2, 0].set_title("abs log diff")
        ax[2, 1].imshow(np.abs(target_ - prediction_))
        ax[2, 1].set_title("abs diff")
        if event_frame is not None:
            a = np.zeros(event_frame.shape)
            a[:,:,0]= (np.sum(event_frame.astype("float32"), axis=-1)>0)
            a[:,:,1]= np.clip(target_.copy(), 0, 1) 
            ax[2, 2].imshow(a)
            ax[2, 2].set_title("event frame")

        log_diff_ = np.abs(np.log(target_ + eps) - np.log(prediction_ + eps))
        log_diff_[~mask] = 0
        ax[3, 0].imshow(log_diff_)
        ax[3, 0].set_title("abs log diff masked")
        abs_diff_ = np.abs(target_ - prediction_)
        abs_diff_[~mask] = 0
        ax[3, 1].imshow(abs_diff_)
        ax[3, 1].set_title("abs diff masked")
        ax[3, 2].imshow(mask)
        ax[3, 2].set_title("mask frame")
        fig.canvas.manager.set_window_title(prefix+"_Depth_Evaluation")
        plt.show()

    return metrics


if __name__ == "__main__":
    flags = FLAGS()

    # predicted labels
    prediction_files = sorted(glob.glob(join(flags.predictions_dataset, '*.npy')))
    prediction_files = prediction_files[flags.prediction_offset:]

    target_files = sorted(glob.glob(join(flags.target_dataset, '*.npy')))
    target_files = target_files[flags.target_offset:]

    if flags.event_masks is not "":
        event_frame_files = sorted(glob.glob(join(flags.event_masks, '*png')))
        event_frame_files = event_frame_files[flags.prediction_offset:]

    # Information about the dataset length
    print("len of prediction files", len(prediction_files))
    print("len of target files", len(target_files))

    if flags.event_masks is not "":
        print("len of events files", len(event_frame_files))

    assert len(prediction_files)>0
    assert len(target_files)>0

    if flags.event_masks is not "":
        use_event_masks = len(event_frame_files)>0
    else:
        use_event_masks = False

    metrics = {}

    num_it = len(target_files)
    for idx in tqdm.tqdm(range(num_it)):
        p_file, t_file = prediction_files[idx], target_files[idx]

        # Read absolute scale ground truth
        target_depth = np.load(t_file)
        target_depth = np.squeeze(target_depth)
        # print(target_depth.shape)

        # Crop depth height according to argument
        target_depth = target_depth[:flags.crop_ymax]
        # print(target_depth.shape)

        # Read predicted depth data
        predicted_depth = np.load(p_file)
        predicted_depth = np.squeeze(predicted_depth)

        # Crop depth height according to argument
        predicted_depth = predicted_depth[:flags.crop_ymax]

        # Check if prediction is coming in inverse log depth
        if flags.inv:
            predicted_depth = inv_depth_to_depth(predicted_depth)

        # Convert to the correct scale
        target_depth, predicted_depth, valid_mask = prepare_depth_data(target_depth, predicted_depth, flags.clip_distance)



        assert predicted_depth.shape == target_depth.shape

        debug = flags.debug and idx == flags.idx

        metrics = add_to_metrics(idx, metrics, target_depth, predicted_depth, valid_mask, event_frame=None, prefix="_", debug=debug, output_folder=flags.output_folder)

        for depth_threshold in [10, 20, 30]:
            depth_threshold_mask = (np.nan_to_num(target_depth) < depth_threshold)
            add_to_metrics(-1, metrics, target_depth, predicted_depth, valid_mask & depth_threshold_mask,
                            prefix=f"_{depth_threshold}_", debug=debug)

       


    pprint({k: v/num_it for k,v in metrics.items()})
    {print ("%s : %f" % (k, v/num_it)) for k,v in metrics.items()}