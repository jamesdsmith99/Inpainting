import numpy as np
from matplotlib import pyplot as plt
import torchvision
import wandb

def mean_GAN_loss_wandb(G_loss_arr, D_loss_arr, epoch):
    G_loss = np.mean(G_loss_arr)
    D_loss = np.mean(D_loss_arr)

    wandb.log({
        'Generator Loss': G_loss,
        'Discriminator Loss': D_loss
    }, step=epoch)

def mean_skip_GAN_loss_wandb(G_loss_arr, D_loss_arr, loss_rec_arr, loss_adv_arr, epoch):
    G_loss = np.mean(G_loss_arr)
    D_loss = np.mean(D_loss_arr)
    l_rec = np.mean(loss_rec_arr)
    l_adv = np.mean(loss_adv_arr)

    wandb.log({
        'Generator Loss': G_loss,
        'Discriminator Loss': D_loss,
        'Reconstruction Loss': l_rec,
        'Adversarial Loss': l_adv
    }, step=epoch)

def log_samples(x, p, n, epoch):
    '''
        plot n predictions below the blacked out image
    '''
    fig, ax = plt.subplots(2, n)
    fig.set_size_inches(9, 12.5)

    # put images in range [0, 1] from [-1, 1]
    p = (p + 1) * 0.5
    x = (x + 1) * 0.5

    for i in range(n):
        ax[0][i].imshow(p[i].squeeze().cpu().detach().numpy(), cmap='gray')
        ax[1][i].imshow(x[i].squeeze().cpu().detach().numpy(), cmap='gray')

    wandb.log({"Image Examples": fig}, step=epoch)
    plt.close()