import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def ssim(
        img1, img2, max_val=2.0, filter_size=11, filter_sigma=1.5, k1=0.01,
        k2=0.03):
    def gaussian(window_size, sigma=1.5):
        gauss = torch.Tensor([
            exp(-(x - window_size//2)**2/float(2*sigma**2))
            for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel, sigma):
        _1D_window = gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(img1, img2, window, window_size, channel, max_val, k1, k2):
        padding = window_size//2
        mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(
            img1*img1, window, padding=padding, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2*img2, window, padding=padding, groups=channel) - mu2_sq
        sigma12 = F.conv2d(
            img1*img2, window, padding=padding, groups=channel) - mu1_mu2

        L = 2 ** max_val - 1
        C1 = (k1 * L)**2
        C2 = (k2 * L)**2

        ssim_map = ((
            (2*mu1_mu2 + C1)*(2*sigma12 + C2)) /
            ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)))

        return ssim_map.mean()

    (_, channel, _, _) = img1.size()
    window = create_window(filter_size, channel, filter_sigma)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, filter_size, channel, max_val, k1, k2)


def psnr(y_true, y_pred, max_pixel_value=2.0):
    mae = torch.mean(torch.abs(y_true - y_pred))
    if mae == 0:
        return 100
    return 20 * torch.log10(max_pixel_value / mae)
