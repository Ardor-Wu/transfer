# from targets.MBRS.MBRS_repo.MBRS_utils.load_test_setting import noise_layers, with_diffusion
# from targets.MBRS.MBRS_repo.MBRS_utils.load_train_setting import batch_size
from targets.MBRS.MBRS_repo.network.Network import *


# from targets.MBRS.MBRS_repo.MBRS_utils.load_test_setting import *


class MBRS():

    def __init__(self, H, W, message_length, device):
        noise_layers = ["Combined([Identity()])"]
        batch_size = 1
        lr = 1e-3
        with_diffusion = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = Network(H, W, message_length, noise_layers, self.device, batch_size, lr, with_diffusion)
        EC_path = "/scratch/qilong3/transferattack/target_model/mbrs_AT/64bits_cnn_AT.pth"
        self.network.load_model_ed(EC_path)
        self.encoder = self.network.encoder_decoder.module.encoder
        self.decoder = self.network.encoder_decoder.module.decoder
        self.encoder.eval()
        self.decoder.eval()
        # move to device
        self.encoder.to(device)
        self.decoder.to(device)
        self.device = device
