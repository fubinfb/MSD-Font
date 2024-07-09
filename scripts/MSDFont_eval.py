import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from einops import rearrange, repeat
from torchvision.utils import make_grid

class Compose(object):
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        for t in self.tf:
            img = t(img)
        return img

mean = [0.5]
std = [0.5]

transform = Compose([transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])

def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font

def render(font, char, size=(128, 128), pad=20):
    width, height = font.getsize(char)
    max_size = max(width, height)

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)

    return img

def render_RGB(font, char):
    img = render(font, char)
    img = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
    img = transform(img)
    return img
    
    
def render_Grey(font, char):
    img = render(font, char)
    img = transform(img)
    return img

def sample(population, k):
    if len(population) < k:
        sampler = random.choices
    else:
        sampler = random.sample
    sampled = sampler(population, k=k)
    return sampled

def load_ttf_data_from_single_dir(data_dir, char_filter=None, extension="ttf", n_font=None):
    data_dir = Path(data_dir)
    font_paths = sorted(data_dir.glob(f"*.{extension}"))
    if n_font is not None:
        font_paths = sample(font_paths, n_font)

    key_font_dict = {}
    key_char_dict = {}

    for font_path in font_paths:
        key = font_path.stem

        with open(str(font_path).replace(f".{extension}", ".txt"), encoding="utf-8") as f:
            chars = f.read()
        if char_filter is not None:
            chars = set(chars).intersection(char_filter)

        if not chars:
            print(font_path.name, "is excluded! (no available characters)")
            continue
        else:
            font = read_font(font_path)
            print(font_path)
            key_font_dict[key] = font
            key_char_dict[key] = list(chars)

    return key_font_dict, key_char_dict

def load_ttf_data(data_dirs, char_filter=None, extension="ttf", n_font=None):
    data_dirs = [data_dirs] if not isinstance(data_dirs, list) else data_dirs
    # print(data_dirs)
    char_filter = set(char_filter) if char_filter is not None else None
    # print(char_filter)

    key_font_dict = {}
    key_char_dict = {}

    for pathidx, path in enumerate(data_dirs):
        _key_font_dict, _key_char_dict = load_ttf_data_from_single_dir(path, char_filter, extension, n_font)
        key_font_dict.update(_key_font_dict)
        key_char_dict.update(_key_char_dict)

    return key_font_dict, key_char_dict

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        default=None,
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="number of ddpm sampling steps",
    )
    parser.add_argument(
        "--path_genchar",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_refchar",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_ttf",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--source_path",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_config_rec",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_rec_model",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_config_trans",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--path_trans_model",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    opt = parser.parse_args()

    save_dir = opt.outdir

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    genimgs_save_dir = Path(os.path.join(save_dir, "genimgs"))
    genimgs_save_dir.mkdir(parents=True, exist_ok=True)
    gtimgs_save_dir = Path(os.path.join(save_dir, "gtimgs"))
    gtimgs_save_dir.mkdir(parents=True, exist_ok=True)

    path_genchar = opt.path_genchar

    path_refchar = opt.path_refchar

    path_ttf = opt.path_ttf

    source_path = opt.source_path

    gen_chars = json.load(open(path_genchar))
    ref_chars = json.load(open(path_refchar))
    extension = "ttf"

    key_font_dict, key_ref_dict = load_ttf_data(path_ttf, char_filter=ref_chars, extension=extension)

    key_gen_dict = {k: gen_chars for k in key_ref_dict}
    
    batch_size = 64

    source = read_font(source_path)

    path_config_rec = opt.path_config_rec
    path_rec_model = opt.path_rec_model
    config1 = OmegaConf.load(path_config_rec)
    rec_model = instantiate_from_config(config1.model)
    rec_model.load_state_dict(torch.load(path_rec_model)["state_dict"],
                          strict=False)
    path_config_trans = opt.path_config_trans
    path_trans_model = opt.path_trans_model
    config2 = OmegaConf.load(path_config_trans)
    trans_model = instantiate_from_config(config2.model)

    sd = torch.load(path_trans_model, map_location="cpu")
    if "state_dict" in list(sd.keys()):
        sd = sd["state_dict"]
    keys = list(sd.keys())
    ignore_keys = []
    for k in keys:
        if 'first_stage_model' in k:
            ignore_keys.append(k)
        if 'cond_stage_model' in k:
            ignore_keys.append(k)

    for k in keys:
        for ik in ignore_keys:
            if k.startswith(ik):
                print("Deleting key {} from state_dict.".format(k))
                del sd[k]
    trans_model.load_state_dict(sd, strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    rec_model = rec_model.to(device)
    trans_model = trans_model.to(device)


    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with rec_model.ema_scope():
            with trans_model.ema_scope():
                for key, gchars in key_gen_dict.items():
                    (genimgs_save_dir / key).mkdir(parents=True, exist_ok=True)
                    (gtimgs_save_dir / key).mkdir(parents=True, exist_ok=True)

                    ref_chars = key_ref_dict[key] # the ref_chars defined in the ref_chars.json
                    ref_imgs = torch.stack([render_Grey(key_font_dict[key], c) for c in ref_chars]).cuda() # for font img

                    ref_batches = torch.split(ref_imgs, batch_size)

                    iter = 0
                    for batch in ref_batches:
                        _, zs1_temp = rec_model.style_stage_model.encode(batch)
                        _, zs2_temp = trans_model.style_stage_model.encode(batch)

                        if iter == 0:
                            zs1 = zs1_temp
                            zs2 = zs2_temp
                        else:
                            zs1 = torch.cat((zs1, zs1_temp), dim=0)
                            zs2 = torch.cat((zs2, zs2_temp), dim=0)
                        iter = iter + 1
                    zs1 = torch.mean(zs1, dim=0, keepdim=True)
                    zs2 = torch.mean(zs2, dim=0, keepdim=True)
                    # print("zs.shape", zs.shape)

                    source_imgs = torch.stack([render_RGB(source, c) for c in gchars]).cuda() # for source img
                    source_batches = torch.split(source_imgs, batch_size)

                    source_Grey_imgs = torch.stack([render_Grey(source, c) for c in gchars]).cuda() # for source img
                    source_Grey_batches = torch.split(source_Grey_imgs, batch_size)

                    iter = 0
                    for batch_RGB,  batch_Grey in zip(source_batches, source_Grey_batches):
                        xf_encoder_posterior = rec_model.encode_first_stage(batch_RGB)
                        zf = rec_model.get_first_stage_encoding(xf_encoder_posterior).detach()

                        zcf1, _ = rec_model.style_stage_model.encode(batch_Grey)
                        zcf2, _ = trans_model.style_stage_model.encode(batch_Grey)

                        ##################### diffusion process #####################
                        nbs, nc, nh, nw = zf.shape
                        shape = [nbs, 4, 16, 16]
                        zs_bs1 = torch.repeat_interleave(zs1, nbs, dim=0)
                        c1 = dict(zf=[zf], zcs=[zs_bs1], zcf=[zcf1])
                        zs_bs2 = torch.repeat_interleave(zs2, nbs, dim=0)
                        c2 = dict(zf=[zf], zcs=[zs_bs2], zcf=[zcf2])

                        z0, progressives = trans_model.progressive_denoising(rec_model, c1, c2, shape=shape)
                        
                        x_gen = rec_model.decode_first_stage(z0)

                        for gen_img in x_gen:
                            char = gchars[iter]
                            # save_char_name = char
                            save_char_name = char.encode("utf-8").decode("latin1") # some system need transfer encoding to save chinese file
                            genpath = genimgs_save_dir / key / f"{save_char_name}.png"
                            img = custom_to_pil(gen_img)
                            img.save(genpath)

                            gt_img = render(key_font_dict[key], char)
                            gtpath = gtimgs_save_dir / key / f"{save_char_name}.png"
                            gt_img.save(gtpath)

                            iter = iter + 1
