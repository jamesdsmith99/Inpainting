import torch
import numpy as np

from .loss import size_discriminator_loss, size_generator_loss
from batch_modification import Eraser, Implanter
from TorchIO import load_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AcaiTrainer:

    def __init__(self, G, D, optimiser_G, optimiser_D, data_loader, batch_size, num_iters, overtrain_D=1, max_erase_size=16, metric_logger=None, sample_logger=None):
        self.G, self.D = G.to(DEVICE), D.to(DEVICE)

        self.optimiser_G, self.optimiser_D = optimiser_G, optimiser_D

        self.num_iters = num_iters
        self.batch_size = batch_size

        self.data_loader = data_loader

        self.overtrain_D = overtrain_D

        self.max_erase_size = max_erase_size

        self.log_metrics = metric_logger
        self.log_samples = sample_logger

        self.G_loss_arr, self.D_loss_arr = np.zeros(0), np.zeros(0)

    def load_state(self, state_path_G, state_path_D):
        load_model(state_path_G, self.G, self.optimiser_G)
        load_model(state_path_D, self.D, self.optimiser_D)
        
    def _train_D(self):
        
        x = next(self.data_loader).float().to(DEVICE)
        x̂, sizes, positions = Eraser.erase_random_size_location(x, self.max_erase_size)

        fake = self.G(x̂)
        fake_images = Implanter.implant(x, fake, positions, sizes)

        self.optimiser_D.zero_grad()

        real_images = next(self.data_loader).float().to(DEVICE)

        size_vec = torch.tensor(sizes).to(DEVICE)
        
        D_real = self.D(real_images).view(-1)
        D_fake = self.D(fake_images).view(-1)
        loss = size_discriminator_loss(D_real, D_fake, size_vec)

        loss.backward()
        self.optimiser_D.step()


        # append metrics to array
        self.D_loss_arr = np.append(self.D_loss_arr, loss.item())

    def _train_G(self):
        x = next(self.data_loader).float().to(DEVICE)
        x̂, sizes, positions = Eraser.erase_random_size_location(x, self.max_erase_size)

        self.x̂ = x̂ # save to a field to be accessed for logging

        gen = self.G(x̂)
        fake_images = Implanter.implant(x, gen, positions, sizes)
        self.gen = fake_images # save to a field to be accessed for logging

        self.optimiser_G.zero_grad()

        discriminator_feedback = self.D(fake_images).view(-1)

        loss = size_generator_loss(discriminator_feedback)

        loss.backward()
        self.optimiser_G.step()

        
        self.G_loss_arr = np.append(self.G_loss_arr, loss.item())
    
    def train(self, epochs, base=0):
        epoch = base

        self.G.train()
        self.D.train()

        while epoch < base + epochs:
            
            for step in range(self.num_iters):
                # overtrain discriminator
                for k in range(self.overtrain_D):
                    self._train_D()

                # train generator
                self._train_G()

            # record metrics
            if self.log_metrics:
                self.log_metrics(self.G_loss_arr, self.D_loss_arr, epoch)
            self.G_loss_arr = np.zeros(0)
            self.D_loss_arr = np.zeros(0)

            # record samples
            if epoch % 25 == 0 and self.log_samples:
                self.log_samples(self.x̂, self.gen, 4, epoch)

            epoch += 1
