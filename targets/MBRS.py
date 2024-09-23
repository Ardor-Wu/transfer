from torch.utils.data import DataLoader
from utils import *
from torchvision.utils import save_image

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "MBRS_repo"))

from network.Network import *

from MBRS_utils.load_test_setting import *

from tqdm import tqdm


class MBRS():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(H, W, message_length, noise_layers, self.device, batch_size, lr, with_diffusion)
        EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
        self.network.load_model_ed(EC_path)