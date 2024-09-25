from . import *
from .Encoder_MP import Encoder_MP, Encoder_MP_Diffusion
from .Decoder import Decoder, Decoder_Diffusion
from .Noise import Noise
import random
import sys
sys.path.append('/scratch/qilong3/watermark_robustness')
from noise_layers.identity import Identity
from noise_layers.diff_jpeg import DiffJPEG
from noise_layers.gaussian import Gaussian
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.brightness import Brightness
from noise_layers.gaussian_blur import GaussianBlur
from noise_layers.adversarial import Adversarial


class EncoderDecoder(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder_MP(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder(H, W, message_length)
        self.device = device

    def forward(self, image, message):
        encoded_images = self.encoder(image, message)
        noise_list = [0, 1, 2, 3, 4]
        for i in range(encoded_images.shape[0]):
            choice = random.choice(noise_list)
            if choice == 0:
                noise_layers = Identity()
            elif choice == 1:
                noise_layers = DiffJPEG(random.randint(10, 99), self.device)
            elif choice == 2:
                noise_layers = Gaussian(random.uniform(0, 0.1))
            elif choice == 3:
                noise_layers = GaussianBlur(std=random.uniform(0, 2.0))
            # noise_layers = Crop(random.uniform(0.3, 0.7))
            elif choice == 4:
                # noise_layers = Resize(random.uniform(0.3, 0.7))
                noise_layers = Brightness(random.uniform(1.0, 3))
            img_noised = noise_layers(encoded_images[i:i + 1, :, :, :])
            if i == 0:
                noised_images = img_noised
            else:
                noised_images = torch.cat((noised_images, img_noised), 0)
        # noised_image = self.noise([encoded_images, image])
        decoded_message = self.decoder(noised_images)
        return encoded_images, noised_images, decoded_message


class EncoderDecoder_Diffusion(nn.Module):
    '''
    A Sequential of Encoder_MP-Noise-Decoder
    '''

    def __init__(self, H, W, message_length, noise_layers):
        super(EncoderDecoder_Diffusion, self).__init__()
        self.encoder = Encoder_MP_Diffusion(H, W, message_length)
        self.noise = Noise(noise_layers)
        self.decoder = Decoder_Diffusion(H, W, message_length)

    def forward(self, image, message):
        encoded_image = self.encoder(image, message)
        noised_image = self.noise([encoded_image, image])
        decoded_message = self.decoder(noised_image)

        return encoded_image, noised_image, decoded_message
