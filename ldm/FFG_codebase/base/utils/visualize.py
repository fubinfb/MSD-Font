"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import torch
from torchvision import utils as tv_utils
from PIL import Image
import scipy.io as io

def refine(imgs):
    refined = torch.ones_like(imgs)
    refined[:, :, 2:-2, 2:-2] = imgs[:, :, 2:-2, 2:-2]
    refined[refined > 0.8] = 1.
    return refined


def make_comparable_grid(*batches, nrow):
    assert all(len(batches[0]) == len(batch) for batch in batches[1:])
    N = len(batches[0])
    # print("the size of the batches")
    # print(len(batches))
    batches = [b.detach().cpu() for b in batches]

    grids = []
    for i in range(0, N, nrow):
        rows = [batch[i:i+nrow] for batch in batches]
        row = torch.cat(rows)
        grid = to_grid(row, 'torch', nrow=nrow)
        grids.append(grid)

        C, _H, W = grid.shape
        sep_bar = torch.zeros(C, 10, W)
        grids.append(sep_bar)

    return torch.cat(grids[:-1], dim=1)


def normalize(tensor, eps=1e-5):
    """ Normalize tensor to [0, 1] """
    # eps=1e-5 is same as make_grid in torchvision.
    minv, maxv = tensor.min(), tensor.max()
    tensor = (tensor - minv) / (maxv - minv + eps)

    return tensor


def to_grid(tensor, to, **kwargs):
    """ Integrated functions of make_grid and save_image
    Convert-able to torch tensor [0, 1] / ndarr [0, 255] / PIL image / file save
    """
    to = to.lower()

    grid = tv_utils.make_grid(tensor, **kwargs, normalize=True)
    if to == 'torch':
        return grid

    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    if to == 'numpy':
        return ndarr

    im = Image.fromarray(ndarr)
    if to == 'pil':
        return im

    im.save(to)


def save_tensor_to_image(tensor, filepath, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    tensor = normalize(tensor)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # for grey image, transfer h,w,1 to h,w
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)

def get_center_crop_bbox(unisize, img_size):
        """Randomly get a crop bounding box."""
        # crop_sizeh = 10
        # crop_sizew = 10
        margin_h1 = (img_size - unisize) // 2
        # margin_h2 = img_size - unisize - margin_h1
        margin_w1 = (img_size - unisize) // 2
        # margin_w2 = img_size - unisize - margin_w1
        # margin_h = max(128 - crop_sizeh, 0)
        # margin_w = max(128 - crop_sizew, 0)
        # offset_h = np.random.randint(0, margin_h + 1)
        # offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = margin_h1, margin_h1 + unisize
        crop_x1, crop_x2 = margin_w1, margin_w1 + unisize

        return crop_y1, crop_y2, crop_x1, crop_x2

def crop(img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[:,crop_y1:crop_y2, crop_x1:crop_x2]
        return img

def save_tensor_to_image_withCrop(tensor, filepath, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    crop_bbox = get_center_crop_bbox(unisize=96, img_size=128)
    tensor = crop(tensor, crop_bbox)
    tensor = normalize(tensor)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # for grey image, transfer h,w,1 to h,w
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)

def save_tensor_to_image_with_thr(tensor, filepath, thr=0.5, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    # thr = 0.5 # value>thr -> value=1.0 -> value=255 -> render into white
    tensor = normalize(tensor)
    tensor[tensor>thr] = 1.0
    tensor[tensor<=thr] = 0.0 
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # for grey image, transfer h,w,1 to h,w
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)


def save_tensor_as_mat(tensor, filepath, scale=None):
    # tensor = normalize(tensor)
    # ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = tensor.permute(1, 2, 0).cpu().numpy()
    # Note: change the filepath to /.../../filename.mat
    io.savemat(filepath, {'DistanceField': ndarr})


def save_tensor_to_image_MS(tensor, filepath, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    # print(tensor.shape)
    # tensor = torch.tensor(tensor).unsqueeze(dim=0)
    tensor = torch.tensor(tensor).view(1,128,128)
    tensor = normalize(tensor)
    # print(tensor.shape)
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # for grey image, transfer h,w,1 to h,w
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)

def save_tensor_to_image_with_thr_MS(tensor, filepath, thr=0.5, scale=None):
    """ Save torch tensor to filepath
    Same as torchvision.save_image; only scale factor is difference.
    """
    # thr = 0.5 # value>thr -> value=1.0 -> value=255 -> render into white
    tensor = torch.tensor(tensor).view(1,128,128)
    tensor = normalize(tensor)
    tensor[tensor>thr] = 1.0
    tensor[tensor<=thr] = 0.0 
    ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # for grey image, transfer h,w,1 to h,w
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(-1)
    im = Image.fromarray(ndarr)
    if scale:
        size = tuple(map(lambda v: int(v*scale), im.size))
        im = im.resize(size, resample=Image.BILINEAR)
    im.save(filepath)