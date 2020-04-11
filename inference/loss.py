import torch

def pixel_loss(x, p, x_pos, y_pos, crop_size):
    return (x - p)[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]

def l1_loss(x, p, x_pos, y_pos, crop_size):
    return (x - p).abs().mean()

def l2_loss(x, p, x_pos, y_pos, crop_size):
    return ((x - p)**2).mean()

class DiscriminatorLoss:

    def __init__(self, D):
        D.eval()
        self.D = D

    def calc_loss(self, x, p, x_pos, y_pos, crop_size):
        with torch.no_grad():
            size_pred = self.D(p.unsqueeze(0))

        return size_pred**2 # l2_loss betwen predicted and 0, no need for mean as single image

class SkipGANLoss:

    def __init__(self, D, α, β):
        self.DLoss = DiscriminatorLoss(D)
        self.α, self.β = α, β

    def calc_loss(self, x, p, x_pos, y_pos, crop_size):
        l_rec = l1_loss(x, p, x_pos, y_pos, crop_size)
        l_adv = self.DLoss.calc_loss(x, p, x_pos, y_pos, crop_size)
        return self.α * l_rec + self.β * l_adv