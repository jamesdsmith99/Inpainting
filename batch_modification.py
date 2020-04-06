import torch
import random

class Eraser:

    @staticmethod
    def _erase_at(x, positions, sizes):
        '''
            x = Tensor - batch of images
            positions = [(int, int)] - list of position coordinates in the form (x, y)
            sizes = [int] - list of sizes

            Given the batch of single channel images x, ∀ i ∈ range(x.size[0]):
            set the pixels in a square whose top left corner is located at positions[i]of size sizes[i] to 1.
        '''
        x̂ = x.clone()

        for i in range(x.shape[0]):
            x_coord, y_coord = positions[i]
            size = sizes[i]

            x̂[i, 0, y_coord:y_coord+size, x_coord:x_coord+size] = torch.ones(size, size)

        return x̂

    @staticmethod
    def _randomly_generate_positions(shape, sizes):
        '''
            shape = torch.Size - size of a tensor representing a batch of single channel images
            sizes = [int] - list of ints representing the sizes of the regions to white out

            Given a batch with a shape, randomly generate the location of the top-left corner to white out
            so that the whited out region lies within the image properly.

            The coordinates are in the form (x, y)
        '''
        batch_size, _, h, w = shape
        return [
            (random.randrange(w+1-sizes[i]), random.randrange(h+1-sizes[i])) for i in range(batch_size)
        ]

    @staticmethod
    def erase_random_location(x, size):
        sizes = [size] * x.shape[0]
        positions = Eraser._randomly_generate_positions(x.shape, sizes)

        return Eraser._erase_at(x, positions, sizes), sizes, positions

    @staticmethod
    def erase_random_size_location(x, max_size):
        '''
            x = Tensor - batch of images
            max_size = int - maximum size square that can be erased (inclusive)

            Given a batch of images x erase a square of random size at a random location.
        '''
        sizes = [random.randint(0, max_size) for _ in range(x.shape[0])]
        positions = Eraser._randomly_generate_positions(x.shape, sizes)

        return Eraser._erase_at(x, positions, sizes), sizes, positions


class Implanter:

    @staticmethod
    def implant(x̂, p, positions, sizes):
        '''
            for each image place its inpainted prediction into the batch of original images
        '''
        result = x̂.clone()

        for i in range(x̂.shape[0]):
            x_coord, y_coord = positions[i]
            size = sizes[i]

            p_slice = p[i, :, y_coord:y_coord+size, x_coord:x_coord+size]

            result[i, :, y_coord:y_coord+size, x_coord:x_coord+size] = p_slice

        return result