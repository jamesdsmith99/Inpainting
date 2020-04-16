import torch
import numpy as np

from scipy.spatial.distance import braycurtis, jensenshannon
from scipy.stats import energy_distance, wasserstein_distance

from skimage.metrics import structural_similarity


'''
    convert a tensor with values in range [-1, 1] x to a numpy histogram with n bins
'''
def _convert_to_hist(x, n):
    x = x.detach().cpu().numpy()
    return np.histogram(x, bins=n, range=(-1, 1), density=True)

'''
    calculate the distance between the histograms for the inpainted patches, using the distance metric
'''
def _patch_hist_dist(x, p, x_pos, y_pos, crop_size, metric):
    x = x[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]
    p = p[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]

    x_hist, _ = _convert_to_hist(x, 40)
    p_hist, _ = _convert_to_hist(p, 40)

    return metric(x_hist, p_hist)

'''
    calculate the standard deviation of the elements in a demeaned (mean=0) tensor x
    using bessles correction (dividing by N-1 rather than N)
'''
def _sample_sd(x):
    N = x.numel()
    return ((x**2).sum() / (N - 1)).sqrt()


'''
    calculate the ssim between images x and y. Assumes that x and y have the same dimensions and are non-negative.
'''
def ssim(x, y, c_1, c_2):
    μ_x = x.mean()
    μ_y = y.mean()

    x = x - μ_x
    y = y - μ_y

    σ_x = _sample_sd(x)
    σ_y = _sample_sd(y)

    N = x.numel()

    σ_xy = (x * y).sum() / (N - 1)

    numerator = (2*μ_x*μ_y + c_1) * (2*σ_xy + c_2)
    denominator = (μ_x**2 + μ_y**2 + c_1) * (σ_x**2 + σ_y**2 + c_2)

    return numerator / denominator

def _ssim_loss(x, y):
    #k_1 = 0.01
    #k_2 = 0.03
    # c_i = (2*k_i)**2
    c_1 = (2*0.01)**2
    c_2 = (2*0.03)**2
    # make images non-negative and get ssim
    similarity = ssim(x+1, y+1, c_1, c_2)
    # make this a loss where higher value is worse
    return (1 - similarity).cpu()


def _sk_ssim_loss(x, y):
    x = (x + 1) * 0.5
    y = (y + 1) * 0.5

    x, y = x.detach().cpu().numpy(), y.detach().cpu().numpy()

    return 1 - structural_similarity(x, y)

def sk_patch_ssim_loss(x, p, x_pos, y_pos, crop_size):
    x = x[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]
    p = p[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]

    return _sk_ssim_loss(x, p)

def patch_ssim_loss(x, p, x_pos, y_pos, crop_size):
    x = x[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]
    p = p[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size]

    return _ssim_loss(x, p)

def sk_global_ssim_loss(x, p, x_pos, y_pos, crop_size):
    return _sk_ssim_loss(x[0, :, :], p[0, :, :])

def global_ssim_loss(x, p, x_pos, y_pos, crop_size):
    return _ssim_loss(x, p)

def wasserstein(x, p, x_pos, y_pos, crop_size):
    return _patch_hist_dist(x, p, x_pos, y_pos, crop_size, wasserstein_distance)

def energy(x, p, x_pos, y_pos, crop_size):
    return _patch_hist_dist(x, p, x_pos, y_pos, crop_size, energy_distance)

def jensen_shannon(x, p, x_pos, y_pos, crop_size):
    return _patch_hist_dist(x, p, x_pos, y_pos, crop_size, jensenshannon)

def pixel_loss(x, p, x_pos, y_pos, crop_size):
    return (x - p)[0, y_pos:y_pos+crop_size, x_pos:x_pos+crop_size].cpu()

def l1_loss(x, p, x_pos, y_pos, crop_size):
    return (x - p).abs().mean()

def l2_loss(x, p, x_pos, y_pos, crop_size):
    return ((x - p)**2).mean()

class DiscriminatorLoss:

    def __init__(self, D, device):
        D.eval()
        self.D = D
        self.device = device

    def calc_loss(self, x, p, x_pos, y_pos, crop_size):
        with torch.no_grad():
            size_pred = self.D(p.unsqueeze(0).to(self.device)).cpu()

        return size_pred**2 # l2_loss betwen predicted and 0, no need for mean as single image

class SkipGANLoss:

    def __init__(self, D, α, β, device):
        self.DLoss = DiscriminatorLoss(D, device)
        self.α, self.β = α, β

    def calc_loss(self, x, p, x_pos, y_pos, crop_size):
        l_rec = l1_loss(x, p, x_pos, y_pos, crop_size)
        l_adv = self.DLoss.calc_loss(x, p, x_pos, y_pos, crop_size)
        return self.α * l_rec + self.β * l_adv