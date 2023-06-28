import os
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

join = os.path.join

evaluatePSNR = lambda xtrue, x: 10 * np.log10(1 / np.mean((xtrue.flatten('F')-x.flatten('F'))**2))
evaluateSSIM = lambda xtrue, x: ssim(xtrue, x, channel_axis=-1)


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res

def maybe_mkdir_p(directory: str):
    os.makedirs(directory, exist_ok=True)

def to_double_all(img, clip=False):
    """
    Normalize img to 0~1.
    Calculate min and max over 10 phases.
    :param img_norm: img to normalize
    :param clip: clip to 0 ~ INF
    """     
    img_norm = img.copy()
    img_norm = np.clip(img_norm,0,np.inf) if clip else img_norm
    if len(img.shape) == 3: # img.shape = nx*ny*nz
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax
    else:
        img_norm[np.isnan(img_norm)] = 0
        img_norm_amin = np.amin(img_norm,keepdims=True)
        img_norm -= img_norm_amin
        img_norm_amax = np.amax(img_norm, keepdims=True)
        img_norm /= img_norm_amax
    return img_norm, img_norm_amin, img_norm_amax


def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()


# def plot_image_grid(images_np, nrow=3, factor=1, interpolation='lanczos', show=False):
#     """Draws images in a grid
    
#     Args:
#         images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
#         nrow: how many images will be in one row
#         factor: size if the plt.figure 
#         interpolation: interpolation used in plt.imshow
#     """
#     n_channels = max(x.shape[0] for x in images_np)
#     assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
#     images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

#     grid = get_image_grid(images_np, nrow)
    
#     return grid

def max_norm(img):
    img -= np.amin(img)
    img /= np.amax(img)

    return img