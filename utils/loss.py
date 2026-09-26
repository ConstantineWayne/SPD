import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters.sobel import spatial_gradient, sobel

def scale_invariant_loss(y_input, y_target, weight=1.0, n_lambda=1.0):
    mask = (y_target > 0) & (y_target < 1.0)
    y_input = y_input[mask]
    y_target = y_target[mask]
    log_diff = y_input - y_target
    return weight * (((log_diff ** 2).mean() - (n_lambda * (log_diff.mean()) ** 2)) ** 0.5)


def scale_invariant_log_loss(y_input, y_target, n_lambda=1.0):
    log_diff = torch.log(y_input) - torch.log(y_target)
    is_nan = torch.isnan(log_diff)
    return (log_diff[~is_nan] ** 2).mean() - (n_lambda * (log_diff[~is_nan].mean()) ** 2)


def mse_loss(y_input, y_target):
    return F.mse_loss(y_input[~torch.isnan(y_target)], y_target[~torch.isnan(y_target)])


class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale=1, num_scales=4):
        super(MultiScaleGradient, self).__init__()
        print('Setting up Multi Scale Gradient loss...')

        self.start_scale = start_scale
        self.num_scales = num_scales

        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2 ** scale), self.start_scale * (2 ** scale)) for
                             scale in range(self.num_scales)]
        print('Done')

    def forward(self, prediction, target, preview=False):
        # helper to remove potential nan in labels

        loss_value = 0
        diff = prediction - target
        _, _, H, W = target.shape
        upsample = torch.nn.Upsample(size=(2 * H, 2 * W), mode='bicubic', align_corners=True)
        record = []
        nbr_ignored_scales = 0

        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            if preview:
                record.append(upsample(sobel(m(diff))))
            else:
                # Use kornia spatial gradient computation
                delta_diff = spatial_gradient(m(diff))
                is_nan = torch.isnan(delta_diff)
                is_not_nan_sum = (~is_nan).sum()
                # output of kornia spatial gradient is [B x C x 2 x H x W]
                new_loss = torch.abs(delta_diff[~is_nan]).sum() / is_not_nan_sum * target.shape[0] * 2

                # ignore losses that are nan due to target = nan
                if new_loss != new_loss:
                    nbr_ignored_scales += 1
                else:
                    loss_value += new_loss

        if preview:
            return record
        else:
            return loss_value / (self.num_scales - nbr_ignored_scales)


multi_scale_grad_loss_fn = MultiScaleGradient()


def multi_scale_grad_loss(prediction, target, preview=False):
    return multi_scale_grad_loss_fn.forward(prediction, target, preview)

def temporal_consistency_loss(image0, image1, processed0, processed1, flow01, alpha=50.0, output_images=False):
    """ Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to estimate a visibility mask.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param processed0: [N x C x H x W] processed image (reconstruction) 0
        :param processed1: [N x C x H x W] processed image (reconstruction) 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
        :param alpha: used for computation of the visibility mask (default: 50.0)
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    #xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx = xx.to(image0.device)
    yy = yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)
    visibility_mask = torch.exp(-alpha * (image1 - image0_warped_to1) ** 2)
    processed0_warped_to1 = F.grid_sample(processed0, warping_grid)

    tc_map = visibility_mask * torch.abs(processed1 - processed0_warped_to1) \
             / (torch.abs(processed1) + torch.abs(processed0_warped_to1) + 1e-5)

    tc_loss = tc_map.mean()

    if output_images:
        additional_output = {'image0': image0,
                             'image1': image1,
                             'image0_warped_to1': image0_warped_to1,
                             'processed0_warped_to1': processed0_warped_to1,
                             'visibility_mask': visibility_mask,
                             'error_map': tc_map}
        return tc_loss, additional_output

    else:
        return tc_loss


def warping_flow_loss(image0, image1, flow01):
    """ Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        The temporal loss is the warping error between two processed frames (image reconstructions in E2VID),
        after the images have been aligned using the flow `flow01`.
        The input (ground truth) images `image0` and `image1` are used to derive a flow loss.

        :param image0: [N x C x H x W] input image 0
        :param image1: [N x C x H x W] input image 1
        :param flow01: [N x 2 x H x W] displacement map from image1 to image0
    """
    t_width, t_height = image0.shape[3], image0.shape[2]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(image0.device), yy.to(image0.device)
    xx.transpose_(0, 1)
    yy.transpose_(0, 1)
    xx, yy = xx.float(), yy.float()

    flow01_x = flow01[:, 0, :, :]  # N x H x W
    flow01_y = flow01[:, 1, :, :]  # N x H x W

    warping_grid_x = xx + flow01_x  # N x H x W
    warping_grid_y = yy + flow01_y  # N x H x W

    # normalize warping grid to [-1,1]
    warping_grid_x = (2 * warping_grid_x / (t_width - 1)) - 1
    warping_grid_y = (2 * warping_grid_y / (t_height - 1)) - 1

    warping_grid = torch.stack(
        [warping_grid_x, warping_grid_y], dim=3)  # 1 x H x W x 2

    image0_warped_to1 = F.grid_sample(image0, warping_grid)

    tc_map = torch.abs(image1 - image0_warped_to1)

    tc_loss = tc_map.mean()

    return tc_loss


def voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=False):
    """ Adapted from:
        Temporal loss, as described in Eq. (2) of the paper 'Learning Blind Video Temporal Consistency',
        Lai et al., ECCV'18.

        This function takes an optic flow tensor and uses this to warp the channels of an
        event voxel grid to an image. The variance of this image is the resulting loss.
        :param voxel: [N x C x H x W] input voxel
        :param displacement: [N x 2 x H x W] displacement map from previous flow tensor to current flow tensor
    """
    if reverse_time:
        displacement = -displacement
    v_shape = voxel.size()
    t_width, t_height, t_channels = v_shape[3], v_shape[2], v_shape[1]
    xx, yy = torch.meshgrid(torch.arange(t_width), torch.arange(t_height))  # xx, yy -> WxH
    xx, yy = xx.to(voxel.device).float(), yy.to(voxel.device).float()

    displacement_x = displacement[:, 1, :, :]  # N x H x W
    displacement_y = displacement[:, 0, :, :]  # N x H x W
    displacement_increment = 1.0/(t_channels-1.0)

    voxel_grid_warped = torch.zeros((v_shape[0], 1, t_height, t_width), dtype=voxel.dtype, device=voxel.device) 
    for i in range(t_channels):
        warp_magnitude_ratio = (1.0-i*displacement_increment) if reverse_time else i*displacement_increment
        #Add displacement to the x coords
        warping_grid_x = xx + displacement_x*warp_magnitude_ratio # N x H x W
        #Add displacement to the y coords
        warping_grid_y = yy + displacement_y*warp_magnitude_ratio # N x H x W
        warping_grid = torch.stack([warping_grid_y, warping_grid_x], dim=3)  # 1 x H x W x 2
        #Normalize the warping grid to between -1 and 1 (necessary for grid_sample API)
        warping_grid[:,:,:,0] = (2.0*warping_grid[:,:,:,0])/(t_height-1)-1.0
        warping_grid[:,:,:,1] = (2.0*warping_grid[:,:,:,1])/(t_width-1)-1.0
        voxel_channel_warped = F.grid_sample(voxel, warping_grid)
        voxel_grid_warped+=voxel_channel_warped[:, i:i+1, :, :]

    variance = voxel_grid_warped.var()
    tc_loss = -variance
    if not reverse_time:
        reverse_tc_loss = voxel_warping_flow_loss(voxel, displacement, output_images=False, reverse_time=True)
        tc_loss += reverse_tc_loss
    if output_images:
        additional_output = {'voxel_grid': voxel,
                             'voxel_grid_warped': voxel_grid_warped}
        return tc_loss, additional_output
    else:
        return tc_loss
