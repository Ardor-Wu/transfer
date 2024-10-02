import os
import time
from statistics import median

import torch
import numpy as np
import utils
import logging
from collections import defaultdict
import random
import math
import torchvision

from options import *
from model.hidden import Hidden
# from model.hidden_finetune import Hidden_ft
from average_meter import AverageMeter
from noise_layers.identity import Identity
# import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
# from skimage.metrics import structural_similarity as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure
# from torchmetrics.image import PeakSignalNoiseRatio
from noise_layers.identity import Identity
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.crop import Crop
from noise_layers.brightness import Brightness
from noise_layers.gaussian_blur import GaussianBlur
# import imgaug.augmenters as iaa
from pytorch_msssim import ssim
import imageio
from PIL import Image
import torchvision.transforms as transforms

# from targets.MBRS.MBRS_repo.test import message

torch.autograd.set_detect_anomaly(True)


def get_val_data_loader(data_name, hidden_config, train_options, val_dataset, train=False):
    if 'DB' in data_name:
        val_data = utils.get_data_loaders_DB(hidden_config, train_options, dataset=val_dataset, train=train)
        data_type = 'batch_dict'
        max_batches = None
    elif 'midjourney' in data_name:
        val_data = utils.get_data_loaders_midjourney(hidden_config, train_options, dataset=val_dataset, train=train)
        data_type = 'image_label'
        max_batches = None
    elif 'nlb' in data_name:
        val_data = utils.get_data_loaders_nlb(hidden_config, train_options, dataset=val_dataset)
        data_type = 'image_label'
        max_batches = 1
    else:
        raise ValueError(f"Unknown data_name: {data_name}")
    return val_data, data_type, max_batches


def test_tfattk_hidden(
        model,
        model_list,
        device,
        hidden_config,
        train_options,
        val_dataset,
        train_type,
        model_type,
        data_name,
        wm_method,
        target,
        smooth,
        attk_param=None,
        pp=None,
        watermark_length=30,
        target_length=30,
        num_models=1,
        fixed_message=False,

        optimization=True,
        PA='mean',
        budget=None
):
    dir_string = (
            'results/' +
            wm_method + '_to_' + target + '_' + model_type + '_' +
            str(watermark_length) + '_to_' + str(target_length) + 'bits' + train_type + '_' + str(num_models) + 'models'
    )
    if fixed_message:
        dir_string += '_fixed_message'

    if not optimization:
        dir_string += '_no_optimization'
        if PA == 'mean':
            dir_string += '_mean'
        elif PA == 'median':
            dir_string += '_median'
        if budget is not None:
            dir_string += f'_budget_{budget}'

    if not os.path.exists('results'):
        os.makedirs('results')

    # Configure logging to write messages with level INFO or higher to a file
    logging.basicConfig(filename=dir_string + '.log', filemode='w', level=logging.INFO)

    # Get the validation data loader and data_type
    val_data, data_type, max_batches = get_val_data_loader(data_name, hidden_config, train_options, val_dataset,
                                                           train=False)

    validation_losses = defaultdict(AverageMeter)
    logging.info('Running validation for transfer attack')
    num = 0

    for data in tqdm(iter(val_data)):
        num += 1
        if data_type == 'batch_dict':
            image = data['image'].to(device)
        elif data_type == 'image_label':
            image, _ = data
            image = image.to(device)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")

        if fixed_message:
            watermark_message = '0001101101000010001110111010101010001100010111001011100110011010101010100100101110010101110000011000000110101001000100101011011010111100011010000010000010101111100111010101111101010010110000101101100110101110000011100010001110101010010001110000001101001101'
            # Convert each bit character to a float (0.0 or 1.0)
            message_bits = [float(bit) for bit in watermark_message[:hidden_config.message_length]]

            # Create a PyTorch tensor from the list of floats
            message = torch.tensor(message_bits, dtype=torch.float).to(device)
            batch_size = image.shape[0]  # Assuming image is a tensor with shape [batch_size, ...]
            # Repeat the message and flatten
            message = message.repeat(batch_size)
            message = message.view(batch_size, -1)
        else:
            if hidden_config.message_length == 30:
                message_path = './message/' + str(hidden_config.message_length) + 'bits_message_' + str(num) + '.pth'
                message = torch.load(message_path).to(device)
            else:
                message = torch.randint(0, 2, (image.size(0), hidden_config.message_length)).float().to(device)

        losses, (encoded_images, attk_images, decoded_messages), num = tfattk_validate_on_batch(
            model, [image, message], model_list, num, train_type, model_type,
            attk_param, pp, wm_method, target, smooth=smooth, fixed_message=fixed_message, optimization=optimization,
            PA=PA, budget=budget
        )

        for name, loss in losses.items():
            validation_losses[name].update(loss)

        if max_batches is not None and num >= max_batches:
            break

    utils.log_progress(validation_losses)


def compute_tdr_avg(decoded_rounded, decoded_rounded_attk, messages, smooth, median_bitwise_errors=None):
    threshold2_numerators = {
        20: 2,
        30: 5,
        32: 5,
        48: 11,
        64: 17,
        100: 31  # <1e-4 false positive rate
    }

    message_length = messages.size(1)
    if message_length in threshold2_numerators:
        threshold2_numerator = threshold2_numerators[message_length]
        threshold1_numerator = message_length - threshold2_numerator
        threshold1 = threshold1_numerator / message_length
        threshold2 = threshold2_numerator / message_length
    else:
        raise ValueError(f"Unsupported message length: {message_length}")

    messages_numpy = messages.detach().cpu().numpy()
    decoded_numpy = decoded_rounded
    bit_errors = np.sum(np.abs(decoded_numpy - messages_numpy), axis=1)
    per_message_accuracy = 1 - bit_errors / message_length
    tdr_avg_1 = np.mean(per_message_accuracy > threshold1)
    tdr_avg_2 = np.mean(per_message_accuracy < threshold2)

    if smooth:
        if median_bitwise_errors is None:
            raise ValueError("median_bitwise_errors must be provided when smooth is True")
        per_message_accuracy_attk = 1 - median_bitwise_errors / message_length
    else:
        decoded_attk_numpy = decoded_rounded_attk
        bit_errors_attk = np.sum(np.abs(decoded_attk_numpy - messages_numpy), axis=1)
        per_message_accuracy_attk = 1 - bit_errors_attk / message_length

    tdr_avg_attk_1 = np.mean(per_message_accuracy_attk > threshold1)
    tdr_avg_attk_2 = np.mean(per_message_accuracy_attk < threshold2)

    return tdr_avg_1, tdr_avg_2, tdr_avg_attk_1, tdr_avg_attk_2


def tfattk_validate_on_batch(model, batch: list, model_list: list, num: int, train_type: str, model_type: str,
                             attk_param: float, pp: str, wm_method, target, encode_wm=True, white=False, smooth=False,
                             fixed_message=False, optimization=True, PA='mean', budget=None):
    # model must have encoder and decoder

    images, messages = batch

    batch_size = images.shape[0]

    # self.encoder_decoder.eval()
    ###
    if encode_wm:
        model.encoder.eval()
        model.decoder.eval()
        ###
        # model.discriminator.eval()
        with torch.no_grad():
            # d_target_label_cover = torch.full((batch_size, 1), model.cover_label, device=model.device)
            # d_target_label_encoded = torch.full((batch_size, 1), model.encoded_label, device=model.device)
            # g_target_label_encoded = torch.full((batch_size, 1), model.cover_label, device=model.device)

            # d_on_cover = model.discriminator(images)
            # d_loss_on_cover = model.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

            encoded_images = model.encoder(images, messages)
    else:
        if model is not None:
            model.eval()
        encoded_images = images

    noise = wevade_transfer_batch(encoded_images.clone(), messages.size(1), model_list,
                                  watermark_length=30, iteration=5000,
                                  lr=2, r=0.5, epsilon=0.2, num=num,
                                  name='flipratio_batch', train_type=train_type,
                                  model_type=model_type, batch_size=len(model_list), wm_method=wm_method, target=target,
                                  white=white, fixed_message=fixed_message, optimization=optimization, PA=PA,
                                  budget=budget)

    with torch.no_grad():
        attk_images = encoded_images + noise

        if encode_wm:
            attk_images = utils.transform_image(attk_images, model.device)
        else:
            if model is not None:
                encoded_images_tdr = utils.transform_image_stablesign(encoded_images, images.device)
                attk_images_tdr = utils.transform_image_stablesign(attk_images, images.device)
            encoded_images = utils.transform_image(encoded_images, images.device)
            attk_images = utils.transform_image(attk_images, images.device)

        if encode_wm:
            decoded_messages = model.decoder(encoded_images)
            if smooth:
                decoded_messages_attk_all = []
                for image in attk_images:
                    noised_images = []
                    for i in range(100):
                        gaussian_noise = torch.randn(image.shape).to(images.device)
                        noised_image = image + 0.0015 * gaussian_noise
                        noised_images.append(noised_image)
                    noised_images = torch.stack(noised_images)
                    decoded_messages_attk = model.decoder(noised_images)
                    decoded_messages_attk_all.append(decoded_messages_attk)
                decoded_messages_attk_all = torch.stack(decoded_messages_attk_all)
            else:
                decoded_messages_attk = model.decoder(attk_images)

            # d_on_encoded = model.discriminator(encoded_images)
            # d_loss_on_encoded = model.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            # d_on_encoded_for_enc = model.discriminator(encoded_images)
            # g_loss_adv = model.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            # d_on_attk = model.discriminator(attk_images)
            # d_loss_on_attk = model.bce_with_logits_loss(d_on_attk, d_target_label_encoded.float())

            # d_on_attk_for_enc = model.discriminator(attk_images)
            # g_loss_adv_attk = model.bce_with_logits_loss(d_on_attk_for_enc, g_target_label_encoded.float())
            '''
            if model.vgg_loss is None:
                g_loss_enc = model.mse_loss(encoded_images, images)
                g_loss_enc_attk = model.mse_loss(attk_images, images)
            else:
                vgg_on_cov = model.vgg_loss(images)
                vgg_on_enc = model.vgg_loss(encoded_images)
                vgg_on_enc_attk = model.vgg_loss(attk_images)
                g_loss_enc = model.mse_loss(vgg_on_cov, vgg_on_enc)
                g_loss_enc_attk = model.mse_loss(vgg_on_cov, vgg_on_enc_attk)
    
            g_loss_dec = model.mse_loss(decoded_messages, messages)
            g_loss_dec_attk = model.mse_loss(decoded_messages_attk, messages)
            '''
        else:
            if model is not None:
                decoded_messages = model(encoded_images_tdr) + 0.5
                decoded_messages_attk = model(attk_images_tdr) + 0.5

    if model is not None:
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        if smooth:
            decoded_rounded_attk_all = decoded_messages_attk_all.detach().cpu().numpy().round().clip(0, 1)
        else:
            decoded_rounded_attk = decoded_messages_attk.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])
        if smooth:
            messages_np = messages.detach().cpu().numpy()
            messages_expanded = messages_np[:, None, :]
            bitwise_errors = np.abs(decoded_rounded_attk_all - messages_expanded)
            bitwise_error_sums = np.sum(bitwise_errors, axis=2)
            median_bitwise_errors = np.median(bitwise_error_sums, axis=1)
            bitwise_avg_err_attk = np.mean(median_bitwise_errors) / messages.shape[1]
        else:
            bitwise_avg_err_attk = np.sum(np.abs(decoded_rounded_attk - messages.detach().cpu().numpy())) / (
                    batch_size * messages.shape[1])
            median_bitwise_errors = None

        tdr_avg_1, tdr_avg_2, tdr_avg_attk_1, tdr_avg_attk_2 = compute_tdr_avg(
            decoded_rounded,
            decoded_rounded_attk,
            messages,
            smooth,
            median_bitwise_errors
        )

    ssim_value = []
    for i in range(encoded_images.shape[0]):
        SSIM_metric = StructuralSimilarityIndexMeasure(data_range=attk_images[i].max() - attk_images[i].min()).to(
            encoded_images.device)
        ssim_value.append(SSIM_metric(encoded_images[i].unsqueeze(0), attk_images[i].unsqueeze(0)).cpu().numpy())
    ssim_value = [0 if x == -math.inf else x for x in ssim_value]
    ssim_value = np.array(ssim_value)
    ssim_value[np.isnan(ssim_value)] = 0
    # print(np.sum(np.abs(decoded_rounded_attk - decoded_rounded)) / (
    #         batch_size * messages.shape[1]))

    # print(np.sum(np.abs(decoded_rounded_attk - messages.detach().cpu().numpy()), axis=1) / (
    #         messages.shape[1]))

    if encode_wm:
        losses = {
            # 'encoder_mse         ': g_loss_enc.item(),
            # 'encoder_mse_attk    ': g_loss_enc_attk.item(),
            'noise (L-infinity)  ': torch.mean(
                torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=float('inf'),
                           dim=1)).item(),
            'noise (L-2)         ': torch.mean(
                torch.norm((encoded_images - attk_images).reshape(encoded_images.size(0), -1), p=2, dim=1)).item(),
            # 'dec_mse             ': g_loss_dec.item(),
            # 'dec_mse_attk        ': g_loss_dec_attk.item(),
            'bitwise-acc         ': 1 - bitwise_avg_err,
            'bitwise-acc_attk    ': 1 - bitwise_avg_err_attk,
            'tdr                 ': tdr_avg_1 + tdr_avg_2,
            'tdr_attk            ': tdr_avg_attk_1 + tdr_avg_attk_2,
            'ssim                ': np.mean(ssim_value),
            # 'adversarial_bce     ': g_loss_adv.item(),
            # 'adversarial_bce_attk': g_loss_adv_attk.item(),
            # 'discr_cover_bce     ': d_loss_on_cover.item(),
            # 'discr_encod_bce     ': d_loss_on_encoded.item(),
            # 'discr_attk_bce      ': d_loss_on_attk.item()
        }
    else:
        if model is not None:
            losses = {
                'noise (L-infinity)  ': torch.mean(
                    torch.norm((encoded_images - attk_images).view(encoded_images.size(0), -1), p=float('inf'),
                               dim=1)).item(),
                'noise (L-2)         ': torch.mean(
                    torch.norm((encoded_images - attk_images).view(encoded_images.size(0), -1), p=2, dim=1)).item(),
                'bitwise-acc         ': 1 - bitwise_avg_err,
                'bitwise-acc_attk    ': 1 - bitwise_avg_err_attk,
                'tdr                 ': tdr_avg_1 + tdr_avg_2,
                'tdr_attk            ': tdr_avg_attk_1 + tdr_avg_attk_2,
                'ssim                ': np.mean(ssim_value),
            }
        else:
            attk_images_transform = (attk_images.clone() + 1) / 2
            attk_images_transform = attk_images_transform.clamp_(0, 1)
            save_folder = '/home/yh351/code/tree-ring-watermark-main/' + wm_method + '_' + str(
                len(model_list)) + '_attk_images'
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            for i in range(attk_images_transform.size(0)):
                torchvision.utils.save_image(attk_images_transform[i], save_folder + f'/attk_image_{i}.png')
            losses = {
                'noise (L-infinity)  ': torch.mean(
                    torch.norm((encoded_images - attk_images).view(encoded_images.size(0), -1), p=float('inf'),
                               dim=1)).item(),
                'noise (L-2)         ': torch.mean(
                    torch.norm((encoded_images - attk_images).view(encoded_images.size(0), -1), p=2, dim=1)).item(),
                'ssim                ': np.mean(ssim_value),
            }
            return losses, num
    # print(losses)
    return losses, (encoded_images, attk_images, decoded_messages), num


def normalize_image(image):
    # Convert image to float type for the division operation
    image = image.astype(np.float32)

    # # Find the maximum and minimum values of the image
    # min_val = np.min(image)
    # max_val = np.max(image)

    # # Normalize the image to the range [0, 1]
    # image_normalized = (image - min_val) / (max_val - min_val)

    image_normalized = (image + 1) / 2

    # Scale to the range [0, 255]
    image_scaled = np.clip(image_normalized * 255, 0, 255)

    # Make sure to round the values and convert it back to uint8
    image_output = np.round(image_scaled).astype(np.uint8)

    return image_output


def project(param_data, backup, epsilon):
    # If the perturbation exceeds the upper bound, project it back.
    r = param_data - backup
    r = epsilon * r

    return backup + r


def wevade_transfer_batch(all_watermarked_image, target_length, model_list, watermark_length, iteration, lr, r, epsilon,
                          num, name, train_type, model_type, batch_size, wm_method, target, white=False,
                          fixed_message=False, optimization=True, PA='mean'
                          , budget=None):
    watermarked_image_cloned = all_watermarked_image.clone()
    criterion = nn.MSELoss(reduction='mean')
    # Construct base directory
    base_dir = f'./wevade_perturb_{wm_method}_to_{target}_{model_type}_{watermark_length}_to_{target_length}bits/{train_type}'

    # Ensure the directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Construct filename with 'fixed_message' information
    filename = (
        f'ensemble_model{len(model_list)}_{name}_{batch_size}_'
        f'{int((1 - epsilon) * 100)}_batch_{num}'
    )

    if fixed_message:
        filename += '_fixed'

    if white:
        filename += '_white'

    if not optimization:
        filename += '_no_optimization'
        if PA == 'mean':
            filename += '_mean'
        elif PA == 'median':
            filename += '_median'

    filename += '.pth'

    # Combine base directory and filename to get the full path
    path = os.path.join(base_dir, filename)

    if optimization == False:
        watermarks = []
        for idx, sur_model in enumerate(model_list):
            sur_model.encoder.eval()

            with torch.no_grad():
                sur_model.decoder.eval()
                decoded_messages = sur_model.decoder(all_watermarked_image)
                decoded_rounded = torch.clamp(torch.round(decoded_messages), 0, 1)
                target_watermark = 1 - decoded_rounded
                watermark = sur_model.encoder(all_watermarked_image, target_watermark) - all_watermarked_image
                watermarks.append(watermark)
        watermarks = torch.stack(watermarks)
        if PA == 'mean':
            all_perturbations = watermarks.mean(dim=0)
        elif PA == 'median':
            all_perturbations = watermarks.median(dim=0).values
        else:
            raise ValueError(f"Unsupported PA: {PA}")
        if budget is not None:
            l_inf = torch.norm(all_perturbations, p=float('inf'), dim=(1, 2, 3))
            l_inf = l_inf.reshape(-1, 1, 1, 1)
            all_perturbations = all_perturbations / l_inf * budget
    elif os.path.exists(path):
        all_perturbations = torch.load(path, map_location='cpu').to(all_watermarked_image.device)
    else:
        target_watermark_list = []

        with torch.no_grad():
            watermarked_image = watermarked_image_cloned.clone()
            all_perturbations = torch.zeros_like(watermarked_image)
            continue_processing_mask = torch.ones(watermarked_image.shape[0], dtype=torch.bool,
                                                  device=watermarked_image.device)

        for idx, sur_model in enumerate(model_list):
            with torch.no_grad():
                sur_model.decoder.eval()
                if white:
                    decoded_messages = sur_model.decoder(watermarked_image) + 0.5
                else:
                    decoded_messages = sur_model.decoder(watermarked_image)
                decoded_rounded = torch.clamp(torch.round(decoded_messages), 0, 1)
                target_watermark = 1 - decoded_rounded
                target_watermark_list.append(target_watermark)
        target_watermark_list = torch.stack(target_watermark_list)

        # for _ in tqdm(range(iteration)):
        for _ in range(iteration):
            if continue_processing_mask.all() == False:
                # Mask the watermarked images that don't need further processing
                watermarked_image = all_watermarked_image[continue_processing_mask]
            watermarked_image = watermarked_image.requires_grad_(True)
            min_value, _ = torch.min(watermarked_image.reshape(watermarked_image.size(0), -1), dim=1, keepdim=True)
            max_value, _ = torch.max(watermarked_image.reshape(watermarked_image.size(0), -1), dim=1, keepdim=True)
            min_value = min_value.view(watermarked_image.size(0), 1, 1, 1)
            max_value = max_value.view(watermarked_image.size(0), 1, 1, 1)

            grads = 0
            idx_list = random.sample(range(0, len(model_list)), batch_size)
            for idx in idx_list:
                sur_model = model_list[idx]
                if white:
                    decoded_watermark = sur_model.decoder(watermarked_image) + 0.5
                else:
                    decoded_watermark = sur_model.decoder(watermarked_image)
                loss = criterion(decoded_watermark,
                                 target_watermark_list[idx][continue_processing_mask]) * watermarked_image.size(0)
                if wm_method == 'hidden+stega':
                    if idx % 2 == 0:
                        grads += torch.autograd.grad(loss, watermarked_image)[0]
                    else:
                        grads += torch.autograd.grad(loss, watermarked_image)[0] / 400
                else:
                    grads += torch.autograd.grad(loss, watermarked_image)[0]
                sur_model.decoder.zero_grad()
            grads /= len(idx_list)

            with torch.no_grad():
                watermarked_image = watermarked_image - lr * grads
                watermarked_image = torch.clamp(watermarked_image, min_value, max_value)

                # Projection.
                perturbation_norm = torch.norm(watermarked_image - watermarked_image_cloned[continue_processing_mask],
                                               p=float('inf'), dim=(1, 2, 3), keepdim=True)
                exceeding_indices = perturbation_norm > r
                c = torch.where(exceeding_indices, r / perturbation_norm, torch.ones_like(perturbation_norm))
                if torch.sum(exceeding_indices) > 0:
                    watermarked_image = project(watermarked_image, watermarked_image_cloned[continue_processing_mask],
                                                c)

                bit_acc_target = torch.zeros((len(model_list), len(watermarked_image))).to(watermarked_image.device)
                for idx, sur_model in enumerate(model_list):
                    sur_model.decoder.eval()
                    if white:
                        decoded_watermark = sur_model.decoder(watermarked_image) + 0.5
                    else:
                        decoded_watermark = sur_model.decoder(watermarked_image)
                    rounded_decoded_watermark = decoded_watermark.detach().round().clamp(0, 1)
                    bit_acc_target[idx] = 1 - (torch.abs(
                        rounded_decoded_watermark - target_watermark_list[idx][continue_processing_mask]).sum(
                        dim=1) / watermark_length)
                # bit_acc_target, _ = torch.min(bit_acc_target, dim=0)
                # bit_acc_target = torch.mean(bit_acc_target, dim=0)
                bit_acc_target = torch.mean(bit_acc_target).item()
                # print('bit_acc_target:', bit_acc_target)
                # print(bit_acc_target)

                perturbation = watermarked_image - watermarked_image_cloned[continue_processing_mask]
                all_perturbations[continue_processing_mask] = perturbation
                all_watermarked_image[continue_processing_mask] = watermarked_image
                ssim_value = ssim((watermarked_image + 1) / 2,
                                  (watermarked_image_cloned[continue_processing_mask] + 1) / 2, data_range=1,
                                  size_average=False)

                conditions_met = (ssim_value <= 0.9).view(-1) | (bit_acc_target >= 1 - epsilon)
                continue_processing_mask[continue_processing_mask.clone()] &= ~conditions_met

                # If all images in the batch have met the condition, stop processing
                if not continue_processing_mask.any():
                    break

        torch.save(all_perturbations, path)

    return all_perturbations
