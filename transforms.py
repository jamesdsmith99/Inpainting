import random
import torch
from torch.nn import functional as F

class DCGANNormalize(object):

    '''
        transform an image in [0, 1] to be in [-1, 1]
    '''
    
    def __call__(self, sample):
        return (sample - 0.5) * 2

class RandomCrop(object):

    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, t):
        h, w = t.shape

        if h < self.height:
            t = F.pad(t, (self.height - h, 0))

        if w < self.width:
            t = F.pad(t, (0, self.width - w))

        h, w = t.shape

        y = random.randrange(h - self.height) if h > self.height else 0
        x = random.randrange(w - self.width) if w > self.width else 0

        return t[y:y+self.height, x:x+self.width]

class TensorToImage(object):
    def __call__(self, sample):
        return sample.reshape(1, *sample.shape)