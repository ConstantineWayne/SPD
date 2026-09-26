import numpy as np
import torch
import torch.nn as nn
from base.base_trainer import BaseTrainer
from torchvision import utils
from utils.loss import multi_scale_grad_loss
from utils.training_utils import plot_grad_flow


def quick_norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img) + 1e-5)


class TransformerTrainer(BaseTrainer):

    def __init__(self, model, loss, loss_params, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):
        super(TransformerTrainer, self).__init__(model, loss,
                                                 loss_params, resume, config, train_logger)
        self.config = config
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
        # self.log_step = int(np.sqrt(self.batch_size))
        self.log_step = 1
        self.added_tensorboard_graph = False
        self.weight_loss = config['loss']['weight']

        if config['use_phased_arch']:
            self.use_phased_arch = True
            print("Using phased architecture")
        else:
            self.use_phased_arch = False
            print("Will not use phased architecture")

        # Parameters for multi scale gradiant loss
        if 'grad_loss' in config:
            self.use_grad_loss = True
            try:
                self.weight_grad_loss = config['grad_loss']['weight']
            except KeyError:
                self.weight_grad_loss = 1.0

            print('Using Multi Scale Gradient loss with weight={:.2f}'.format(
                self.weight_grad_loss))
        else:
            print('Will not use Multi Scale Gradiant loss')
            self.use_grad_loss = False

    def _to_input_and_target(self, item):
        events = item['event'].float().to(self.gpu)
        target = item['depth'].float().to(self.gpu)
        image = item['image'].float().to(self.gpu)
        flow = item['flow'].float().to(self.gpu)
        semantic = item['semantic'].float().to(self.gpu) if self.use_semantic_loss else None
        times = item['times'].float().to(self.gpu) if self.use_phased_arch else None
        return events, image, flow, target, semantic, times

    @staticmethod
    def make_preview(event_previews, predicted_targets, groundtruth_targets):
        # event_previews: a list of [1 x 1 x H x W] event previews
        # predicted_frames: a list of [1 x 1 x H x W] predicted frames
        # for make_grid, we need to pass [N x 1 x H x W] where N is the number of images in the grid
        return utils.make_grid(torch.cat(event_previews + predicted_targets + groundtruth_targets, dim=0),
                               normalize=False, scale_each=True,
                               nrow=len(predicted_targets))

    @staticmethod
    def make_grad_loss_preview(grad_loss_frames):
        return utils.make_grid(torch.cat(grad_loss_frames, dim=0),
                               normalize=True, scale_each=True,
                               nrow=len(grad_loss_frames))

    @staticmethod
    def make_movie(event_previews, predicted_frames, groundtruth_targets):
        video_tensor = None
        for i in torch.arange(len(event_previews)):
            voxel = event_previews[i]
            predicted_frame = predicted_frames[i]
            movie_frame = torch.cat([voxel,
                                     predicted_frame,
                                     groundtruth_targets[i]],
                                    dim=-1)
            movie_frame.unsqueeze_(dim=0)
            video_tensor = movie_frame if video_tensor is None else \
                torch.cat((video_tensor, movie_frame), dim=1)
        return video_tensor

    def calculate_total_batch_loss(self, loss_dict, total_loss_dict, L):
        nominal_loss = self.weight_loss * sum(loss_dict['losses']) / float(L)

        losses = []
        losses.append(nominal_loss)

        if self.use_grad_loss:
            grad_loss = self.weight_grad_loss * sum(loss_dict['grad_losses']) / float(L)
            losses.append(grad_loss)

        loss = sum(losses)

        with torch.no_grad():
            if not total_loss_dict:
                total_loss_dict = {'loss': loss, 'L_si': nominal_loss}
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] = grad_loss


            else:
                total_loss_dict['loss'] += loss
                total_loss_dict['L_si'] += nominal_loss
                if self.use_grad_loss:
                    total_loss_dict['L_grad'] += grad_loss

        return total_loss_dict



    def forward_pass_sequence(self, sequence):

        L = len(sequence)

        assert (L > 0)

        total_batch_losses = {}

        loss_dict = {'losses': [], 'grad_losses': []}

        for i, batch_item in enumerate(sequence):

            events = batch_item['events']
            target = batch_item['depth_image']
            rgb = batch_item['image']

            # print(events.shape)
            events = events.float().to(self.gpu)
            target = target.float().to(self.gpu)
            rgb = rgb.float().to(self.gpu)

            pred_dict = self.model(rgb, events)
            pred_depth = pred_dict

            nig_loss = self.model.get_loss()

            # calculate loss
            if self.loss_params is not None:
                loss_dict['losses'].append(
                    self.loss(pred_depth, target, **self.loss_params) + 0.05*nig_loss)

            else:
                loss_dict['losses'].append(self.loss(pred_depth, target)+0.05*nig_loss)

            if self.use_grad_loss:
                grad_loss = multi_scale_grad_loss(pred_depth, target)
                loss_dict['grad_losses'].append(grad_loss)

        total_batch_losses = self.calculate_total_batch_loss(loss_dict, total_batch_losses, L)

        return total_batch_losses

    def _train_epoch(self, epoch):

        self.model.train()

        all_losses_in_batch = {}
        for batch_idx, sequence in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            losses = self.forward_pass_sequence(sequence)
            loss = losses['loss']
            loss.backward()
            if batch_idx % 25 == 0:
                plot_grad_flow(self.model.named_parameters())

            coeffs = self.model.get_coeff()
            coeff_map = {
                'int_head11': 'int_head11',
                'int_head12': 'int_head12',
                'int_head13': 'int_head13',
                'int_head14': 'int_head14',
                'int_head21': 'int_head21',
                'int_head22': 'int_head22',
                'int_head23': 'int_head23',
                'int_head24': 'int_head24',
            }
            for name, param in self.model.named_parameters():

                if param.grad is None:
                    continue

                if param.grad.ndim < 4:
                    continue

                for key in coeff_map:
                    if key in name:
                        weight = 1.5
                        coeff =weight * coeffs[coeff_map[key]].mean()
                        param.grad *= coeff
                        break



            nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()

            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    loss_str = ''
                    for loss_name, loss_value in losses.items():
                        loss_str += '{}: {:.4f} '.format(loss_name, loss_value.item())
                    self.logger.info('Train Epoch: {}, batch_idx: {}, [{}/{} ({:.0f}%)] {}'.format(
                        epoch, batch_idx,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        loss_str))

                    # compute average losses over the batch
        total_losses = {loss_name: sum(loss_values) / len(self.data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}
        log = {
            'loss': total_losses['loss'],
            'losses': total_losses
        }

        if self.valid:
            val_log = self._valid_epoch(epoch=epoch)
            log = {**log, **val_log}

        return log

    def _valid_epoch(self, epoch=0):
        self.model.eval()
        all_losses_in_batch = {}
        with torch.no_grad():
            for batch_idx, sequence in enumerate(self.valid_data_loader):
                losses = self.forward_pass_sequence(sequence)
                for loss_name, loss_value in losses.items():
                    if loss_name not in all_losses_in_batch:
                        all_losses_in_batch[loss_name] = []
                    all_losses_in_batch[loss_name].append(loss_value.item())

                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self.logger.info('Validation: [{}/{} ({:.0f}%)]'.format(
                        batch_idx * self.valid_data_loader.batch_size,
                        len(self.valid_data_loader) * self.valid_data_loader.batch_size,
                        100.0 * batch_idx / len(self.valid_data_loader)))

        total_losses = {loss_name: sum(loss_values) / len(self.valid_data_loader)
                        for loss_name, loss_values in all_losses_in_batch.items()}

        return {'val_loss': total_losses['loss'],
                'val_losses': total_losses}
