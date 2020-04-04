import torch
from torch.nn import functional as F

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

'''
    given the discriminators output for the real and fake data calculate the discriminator loss

    maximise:  log(D(x)) + log(1 - D(G(z)))
    minimise: -log(D(x)) - log(1 - D(G(z)))

    real data has label 1
    fake data has label 0
'''
def discriminator_loss(real, fake):
    loss_real = F.binary_cross_entropy(real, torch.ones_like(real).to(DEVICE))   # -log(real)
    loss_fake = F.binary_cross_entropy(fake, torch.zeros_like(fake).to(DEVICE))  # -log(1 - fake)
    return (loss_real.mean() + loss_fake.mean()) / 2.0 # mean approximates 𝔼

'''
    given the discriminators output for the generated data calculate the generator loss

    log(D(G(z)))

    real data has label 1 and we want to fool the discriminator into thinking our generated data is real
'''

def generator_loss(fake):
    return F.binary_cross_entropy(fake, torch.ones_like(fake).to(DEVICE)) # 𝔼[-log(fake)]