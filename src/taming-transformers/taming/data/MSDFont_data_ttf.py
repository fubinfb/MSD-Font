# modified from font_ttf.py
import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex

from taming.data.base_font import read_font, load_ttf_data, load_img_data, sample, render
from pathlib import Path
from itertools import chain
import random
from PIL import Image
import cv2
import torch
import numpy as np
from torchvision import transforms
import json

class BaseDataset(Dataset):
    def __init__(self, size, data_dirs, train_chars, extension="ttf", n_font=None):
        
        chars = json.load(open(train_chars))

        if isinstance(data_dirs, str):
            self.data_dirs = [data_dirs]
        elif isinstance(data_dirs, list):
            self.data_dirs = data_dirs
        else:
            raise TypeError(f"The type of data_dirs is invalid: {type(data_dirs)}")

        self.use_ttf = (extension == "ttf")
        if self.use_ttf:
            self.load_ttf_data(chars, extension, n_font)
        else:
            self.load_img_data(chars, extension, n_font)

        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.n_fonts = len(self.keys)
        self.n_chars = len(self.chars)

        self.transform = transforms.Compose([
                            transforms.Resize((size, size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

    def load_ttf_data(self, chars, extension, n_font):
        self.key_font_dict, self.key_char_dict = load_ttf_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.get_img = self.render_from_ttf
        self.get_RGBimg = self.render_from_ttf
        self.get_Greyimg = self.render_from_ttf_Grey


    def load_img_data(self, chars, extension, n_font):
        self.key_dir_dict, self.key_char_dict = load_img_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.extension = extension
        self.get_img = self.load_img

    def render_from_ttf(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def render_from_ttf_Grey(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img

    def load_img(self, key, char):
        img_dir = self.key_dir_dict[key][char]
        img = Image.open(str(img_dir / f"{char}.{self.extension}"))
        img = self.transform(img)
        return img


class BaseTrainDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)

class FontTTFTrain(BaseTrainDataset):
    def __init__(self, source_path, keys=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_char_dict, self.char_key_dict = self.filter_chars()
        self.keys = sorted(self.key_char_dict)
        self.chars = sorted(set.union(*map(set, self.key_char_dict.values())))
        self.data_list = [(_key, _char) for _key, _chars in self.key_char_dict.items() for _char in _chars]
        self.n_in_s = 3
        self.n_in_c = 3
        self.n_chars = len(self.chars)  
        self.n_fonts = len(self.keys)   

        self.source = read_font(source_path)

    def render_from_source(self, char):
        img = render(self.source, char)
        img = img.convert("RGB") # current version only support H X W X 3 img, due to the encoder
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def render_from_source_Grey(self, char):
        img = render(self.source, char)
        img = self.transform(img)
        img = img.permute(1,2,0)
        return img
    
    def filter_chars(self):
        char_key_dict = {}
        for char, keys in self.char_key_dict.items():
            num_keys = len(keys)
            if num_keys > 1:
                char_key_dict[char] = keys
            else:
                pass

        filtered_chars = set(char_key_dict)
        key_char_dict = {}
        for key, chars in self.key_char_dict.items():
            key_char_dict[key] = list(set(chars).intersection(filtered_chars))

        return key_char_dict, char_key_dict
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        index = i
        key, char = self.data_list[index]

        trg_img = self.get_RGBimg(key, char) # 128, 128, 3

        style_chars = set(self.key_char_dict[key]).difference({char})
        style_chars = sample(list(style_chars), self.n_in_s)
        style_imgs = torch.stack([self.get_Greyimg(key, c) for c in style_chars]) # 3, 128, 128, 1

        char_img = self.render_from_source(char) # 128, 128, 3
        grey_char_img = self.render_from_source_Grey(char) # 128, 128, 1

        example = dict()
        example["image"] = trg_img # 128, 128, 3
        example["style_imgs"] = style_imgs # 128, 128, 1
        example["char_img"] = char_img # 128, 128, 3
        example["grey_char_img"] = grey_char_img # 128, 128, 1
        return example
    