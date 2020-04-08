import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

from trainers.loss import l2_norm

class HeatmapMaker:

    def __init__(self, model, crop_size, device, stride=1, vis=False):
        model.eval()
        self.model = model.to(device)
        self.device = device

        self.crop_size = crop_size
        self.stride = stride

        self.vis = vis

        self.ones = torch.ones(crop_size, crop_size)

    def _erase_region_at(self, img, y, x):
        img = img.clone()
        img[y:y+self.crop_size, x:x+self.crop_size] = self.ones
        return img
    
    def _create_row_batch(self, img, y):
        w = img.shape[1]
        row = [self._erase_region_at(img, y, x).unsqueeze(0) for x in range(0, w+1-self.crop_size, self.stride)]
        return torch.stack(row)

    def _create_inpainted_image(self, x, p, x_pos, y_pos):
        res = x.clone()
        res[0, y_pos:y_pos+self.crop_size, x_pos:x_pos+self.crop_size] = p[0, y_pos:y_pos+self.crop_size, x_pos:x_pos+self.crop_size]
        return res
    
    def _save_vis_frame(self, x, p, loss, x_pos, y_pos, fname):
        fig, ax = plt.subplots(ncols=2)
        pos = (x_pos, y_pos)
        inpainted = self._create_inpainted_image(x, p, x_pos, y_pos)

        ax[0].imshow(x.squeeze().cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
        ax[1].imshow(inpainted.squeeze().cpu().numpy(), cmap='gray', vmin=-1, vmax=1)

        fig.suptitle(f'Loss: {loss}')
        plt.savefig(f'{fname}.png')
        plt.close()

    def calculate_map(self, scan):
        heatmap, counts = torch.zeros(scan.shape), torch.zeros(scan.shape)
        
        h, w = scan.shape

        for j, y in tqdm(enumerate(range(0, h+1-self.crop_size, self.stride))):
            batch = self._create_row_batch(scan, y).to(self.device)

            with torch.no_grad():
                pred = self.model(batch)

            for i, x in enumerate(range(0, w+1-self.crop_size, self.stride)):
                inpainted_diff = (pred[i] - batch[i])[0, y:y+self.crop_size, x:x+self.crop_size]
                loss = l2_norm(inpainted_diff)
                
                if self.vis:
                    fname = j*((h+1-self.crop_size)//self.stride) + i
                    self._save_vis_frame(batch[i], pred[i], loss, x, y, fname)
                
                heatmap[y:y+self.crop_size, x:x+self.crop_size] = heatmap[y:y+self.crop_size, x:x+self.crop_size] + loss * self.ones
                counts[y:y+self.crop_size, x:x+self.crop_size] = counts[y:y+self.crop_size, x:x+self.crop_size] + self.ones

        counts = counts + torch.ones_like(counts) * 0.000000001 # to prevent division by 0
        return heatmap / counts
