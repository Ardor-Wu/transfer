from numpy.ma.core import argsort

from model.hidden import Hidden
# from model.stega import stegamodel
from noise_layers.noiser import Noiser
import torch
import argparse
from noise_argparser import NoiseArgParser
from options import *
import os
import utils
from attack_func import test_tfattk_hidden
# from attack_theory_flip_all import test_tfattk_DB_theory
import logging
import torchvision.transforms as transforms
from targets.MBRS.MBRS import MBRS
from targets.stegastamp.stega_stamp import StegaStampModel
from targets.RivaGAN_inference.RivaGAN_inference import RivaGAN


def main():
    '''
    ========================
    set all parameters here!
    ========================
    '''
    parser = argparse.ArgumentParser(description='Transfer Attack')
    parser.add_argument('--num_models', type=int, default=10, help='number of surrogate models')
    parser.add_argument('--target', type=str, default='hidden', help='target model')
    parser.add_argument('--target_length', type=int, default=-1, help='target message length')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--no_optimization', action='store_true',
                        help='Disable optimization (set to True when specified)')
    parser.add_argument('--normalized', action='store_true',
                        help='Normalize the watermark (set to True when specified)')
    parser.add_argument('--PA', type=str, default='mean', help='PA for non-optimization method')
    parser.add_argument('--model_type', type=str, default='cnn', help='cnn or resnet')
    parser.add_argument('--resnet_same_encoder', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    # device = 'cuda:' + str(device)
    # seed = 42
    data_dir = ''
    batch_size = 100
    epochs = 200
    num_models = args.num_models

    # num_models = 10
    # change num models to read from arguments

    name = '30bits_AT_DALLE_200epochs_50maintrain'
    size = 128
    train_dataset = 'large_random_10k'
    val_dataset = 'large_random_1k'
    tensorboard = False
    enable_fp16 = False
    noise = None
    train_type = 'AT'
    data_name = 'DB'
    wm_method = 'hidden'
    model_type = args.model_type
    white = False
    smooth = False

    # surrogate message length
    message = 30

    target_length = args.target_length
    if target_length == -1:
        if args.target == 'hidden':
            target_length = 30
        elif args.target == 'mbrs':
            target_length = 64
        elif args.target == 'stega':
            target_length = 100
        elif args.target == 'RivaGAN':
            target_length = 32

    fixed_message = False

    target = args.target

    PA = args.PA
    optimization = not args.no_optimization
    if args.normalized:
        assert not optimization
        budget = 0.25
    else:
        budget = None

    start_epoch = 1
    train_options = TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,
        train_folder=os.path.join(data_dir, 'train'),
        validation_folder=os.path.join(data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=start_epoch,
        experiment_name=name)

    noise_config = noise if noise is not None else []
    if wm_method == 'hidden':
        sur_config = HiDDenConfiguration(H=size, W=size,
                                         message_length=message,
                                         encoder_blocks=4, encoder_channels=64,
                                         decoder_blocks=7, decoder_channels=64,
                                         use_discriminator=True,
                                         use_vgg=False,
                                         discriminator_blocks=3, discriminator_channels=64,
                                         decoder_loss=1,
                                         encoder_loss=0.7,
                                         adversarial_loss=1e-3,
                                         enable_fp16=enable_fp16
                                         )

    if target in ['hidden', 'mbrs', 'stega', 'RivaGAN']:  # mbrs also needs this config
        if model_type == 'cnn':
            target_config = HiDDenConfiguration(H=size, W=size,
                                                message_length=target_length,
                                                encoder_blocks=4, encoder_channels=64,
                                                decoder_blocks=7, decoder_channels=64,
                                                use_discriminator=True,
                                                use_vgg=False,
                                                discriminator_blocks=3, discriminator_channels=64,
                                                decoder_loss=1,
                                                encoder_loss=0.7,
                                                adversarial_loss=1e-3,
                                                enable_fp16=enable_fp16
                                                )
        elif model_type == 'resnet':
            if args.resnet_same_encoder:
                encoder_blocks = 4
            else:
                encoder_blocks = 7
            target_config = HiDDenConfiguration(H=size, W=size,
                                                message_length=target_length,
                                                encoder_blocks=encoder_blocks,
                                                encoder_channels=64,
                                                decoder_blocks=7, decoder_channels=64,
                                                use_discriminator=True,
                                                use_vgg=False,
                                                discriminator_blocks=3, discriminator_channels=64,
                                                decoder_loss=1,
                                                encoder_loss=0.7,
                                                adversarial_loss=1e-3,
                                                enable_fp16=enable_fp16
                                                )
    else:
        target_config = None

    # Model
    noiser = Noiser(noise_config, device)
    if target == 'hidden':
        model = Hidden(target_config, device, noiser, model_type)

    if target == 'hidden':
        if 'DB' in data_name:
            target_cp_file = f'./target model/{target}_{train_type}/{target_length}bits_{model_type}_{train_type}.pth'
        elif 'midjourney' in data_name:
            target_cp_file = f'./target model/{target_length}bits_{model_type}_AT_midjourney.pth'
        if args.resnet_same_encoder:
            target_cp_file = target_cp_file.replace('.pth', '_resnet_same_encoder.pth')

    sur_cp_folder = './surrogate model/' + wm_method + '/' + train_type + '/'

    sur_cp_list = []
    for idx in range(num_models):
        sur_cp_list.append(sur_cp_folder + 'model_' + str(idx + 1) + '.pth')

    if target == 'hidden':
        target_cp = torch.load(target_cp_file, map_location='cpu')
        utils.model_from_checkpoint(model, target_cp)

    if target == 'mbrs':
        model = MBRS(H=size, W=size, message_length=target_length, device=device)

    if target == 'stega':
        model_path = '/scratch/qilong3/transferattack/targets/checkpoints/stegaStamp/stegastamp_pretrained'
        model = StegaStampModel(model_path, device=device)

    if target == 'RivaGAN':
        model = RivaGAN(device=device)

    sur_model_list = []
    for idx in range(len(sur_cp_list)):
        if wm_method == 'hidden':
            sur_model_list.append(Hidden(sur_config, device, noiser, 'cnn'))
        if white:
            linear = torch.nn.Linear(30, 30, bias=True)
            sur_model_list[idx].decoder = torch.nn.Sequential(sur_model_list[idx].decoder, linear.to(device))
            cp = torch.load(sur_cp_list[idx].replace(".pth", "_whit.pth"), map_location='cpu')
        else:
            cp = torch.load(sur_cp_list[idx], map_location='cpu')
        utils.model_from_checkpoint(sur_model_list[idx], cp)

    test_tfattk_hidden(model, sur_model_list, device, target_config, train_options, val_dataset, train_type, model_type,
                       data_name, wm_method, target, smooth, target_length=target_length, num_models=num_models,
                       fixed_message=fixed_message, optimization=optimization, PA=PA, budget=budget,
                       resnet_same_encoder=args.resnet_same_encoder)


if __name__ == '__main__':
    utils.setup_seed(42)
    main()
