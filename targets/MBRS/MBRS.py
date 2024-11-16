# from targets.MBRS.MBRS_repo.MBRS_utils.load_test_setting import noise_layers, with_diffusion
# from targets.MBRS.MBRS_repo.MBRS_utils.load_train_setting import batch_size
from targets.MBRS.MBRS_repo.network.Network import *

# from targets.MBRS.MBRS_repo.MBRS_utils.load_test_setting import *


import torch.nn as nn
import torch


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super(EncoderWrapper, self).__init__()
        self.encoder = encoder

    def forward(self, images, messages):
        # Check if the input image size is 128x128
        if images.shape[-2:] == (128, 128):
            # Rescale to 256x256
            images_to_encode = nn.functional.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        encoded_images=self.encoder(images_to_encode, messages)
        # scale back to 128*128
        encoded_images = nn.functional.interpolate(encoded_images, size=(128, 128), mode='bilinear', align_corners=False)
        return encoded_images


class DecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super(DecoderWrapper, self).__init__()
        self.decoder = decoder

    def forward(self, x):
        # Check if the input image size is 128x128
        if x.shape[-2:] == (128, 128):
            # Rescale to 256x256
            x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return self.decoder(x)


class MBRS():
    def __init__(self, H, W, message_length, device):
        noise_layers = ["Combined([Identity()])"]
        batch_size = 1
        lr = 1e-3
        with_diffusion = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(H, W, message_length, noise_layers, self.device, batch_size, lr, with_diffusion)
        EC_path = f"/scratch/qilong3/transferattack/target_model/mbrs_AT/{message_length}bits_cnn_AT.pth"
        self.network.load_model_ed(EC_path)
        # Wrap the encoder and decoder with the resizing functionality
        self.encoder = EncoderWrapper(self.network.encoder_decoder.module.encoder)
        self.decoder = DecoderWrapper(self.network.encoder_decoder.module.decoder)
        self.encoder.eval()
        self.decoder.eval()
        # Move to device
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device
