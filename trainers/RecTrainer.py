import torch
import numpy as np

from .loss import l1_loss
from batch_modification import Eraser, Implanter
from TorchIO import load_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AdvTrainer:

    def __init__(self, G, optimiser_G, data_loader, batch_size, num_iters, max_erase_size=16, metric_logger=None, sample_logger=None):
        self.G = G.to(DEVICE)

        self.optimiser_G = optimiser_G

        self.num_iters = num_iters
        self.batch_size = batch_size

        self.data_loader = data_loader

        self.max_erase_size = max_erase_size

        self.log_metrics = metric_logger
        self.log_samples = sample_logger

        self.G_loss_arr = np.zeros(0)


    def load_state(self, state_path_G):
        load_model(state_path_G, self.G, self.optimiser_G)

    def _train_G(self):
        x = next(self.data_loader).float().to(DEVICE)
        x̂, sizes, positions = Eraser.erase_random_size_location(x, self.max_erase_size)

        self.x̂ = x̂ # save to a field to be accessed for logging

        gen = self.G(x̂)
        fake_images = Implanter.implant(x, gen, positions, sizes)
        self.gen = fake_images # save to a field to be accessed for logging

        self.optimiser_G.zero_grad()

        loss = l1_loss(fake_images, x)

        loss.backward()
        self.optimiser_G.step()

        self.G_loss_arr = np.append(self.G_loss_arr, loss.item())
    
    def train(self, epochs, base=0):
        epoch = base

        self.G.train()

        while epoch < base + epochs:
            
            for step in range(self.num_iters):
                # train generator
                self._train_G()

            # record metrics
            if self.log_metrics:
                self.log_metrics(self.G_loss_arr, epoch)
            self.G_loss_arr = np.zeros(0)

            # record samples
            if epoch % 10 == 0 and self.log_samples:
                self.log_samples(self.x̂, self.gen, 4, epoch)

            epoch += 1
