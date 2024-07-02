"""
Development of compound deformation field (复合形变场) by Bin Fu
This code is based on Pytorch platform, and some functions are implemented by:
FFG-benchmarks NAVER Corp.
Copyright (c) 2022-present MMLAB-SIAT
MIT license
"""
from turtle import distance
from PIL import Image
import torch
from torch.utils.data import Dataset

from .ttf_utils import render
from .data_utils import load_ttf_data, load_img_data, sample
import numpy as np


# development:
import cv2
from torchvision import transforms
# import skimage
from skimage import morphology

# debug and development
import matplotlib.pyplot as plt


TRANSFORM1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

TRANSFORM2 = transforms.Compose([
    transforms.Normalize([0.5], [0.5])
])

class CDFBaseDataset(Dataset):
    def __init__(self, data_dirs, chars, transform=None, extension="png", n_font=None):
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

        self.transform1 = TRANSFORM1
        self.transform2 = TRANSFORM2

    def load_ttf_data(self, chars, extension, n_font):
        self.key_font_dict, self.key_char_dict = load_ttf_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.get_img = self.render_from_ttf
        self.get_glyph_with_distance = self.gen_glyph_radius
        # self.get_glyph = self.gen_glyph
        self.get_DistanceField = self.gen_DistanceField
        self.transimg = self.transform_img

    def load_img_data(self, chars, extension, n_font):
        self.key_dir_dict, self.key_char_dict = load_img_data(self.data_dirs, char_filter=chars, extension=extension, n_font=n_font)
        self.extension = extension
        self.get_img = self.load_img
        self.get_glyph_with_distance = self.gen_glyph_radius
        # self.get_glyph = self.gen_glyph
        self.get_DistanceField = self.gen_DistanceField
        self.transimg = self.transform_img

    def render_from_ttf(self, key, char):
        font = self.key_font_dict[key]
        img = render(font, char)
        img = self.transform1(img)
        return img

    def load_img(self, key, char):
        img_dir = self.key_dir_dict[key][char]
        img = Image.open(str(img_dir / f"{char}.{self.extension}"))
        img = self.transform1(img)
        return img
    
    def transform_img(self, img):
        img = self.transform2(img)
        return img

    # def render_from_ttf(self, key, char):
    #     font = self.key_font_dict[key]
    #     img = render(font, char)
    #     img = self.transform(img)
    #     return img

    # def load_img(self, key, char):
    #     img_dir = self.key_dir_dict[key][char]
    #     img = Image.open(str(img_dir / f"{char}.{self.extension}"))
    #     img = self.transform(img)
    #     return img

    

    # development for generate glyph and corresponding radius
    # in debug mode, double check the input binary image
    def gen_glyph_radius(self, binary_img):
        # imgpath = os.path.join(path, filename)
        # gray = cv2.imread(imgpath,0)
        # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # binary = 255 - binary

        binary_img = binary_img.squeeze(0).numpy()
        # print("the max value of the binary image")
        # print(np.max(binary_img))
        binary_img = 1.0 - binary_img
        thin, distance = morphology.medial_axis(binary_img, return_distance=True)
        # print("the max value of the distance image")
        # print(np.max(distance))
        distance = torch.tensor(distance).unsqueeze(0).float()
        # thin = binary_img
        # distance = binary_img

        # distance = 
        # dist_on_skel = distance * thin
        # return thin, dist_on_skel
        return thin, distance

    # def gen_glyph(self, binary_img):
    #     thin = morphology.medial_axis(binary_img)
    #     return thin

    def gen_DistanceField(self, binary_img):
        binary_img = binary_img.squeeze(0).numpy()
        binary_img = 1.0 - binary_img
        thin, distance = morphology.medial_axis(binary_img, return_distance=True)
        distance = torch.tensor(distance).unsqueeze(0).float()

        # distance = binary_img
        # thin, distance = morphology.medial_axis(binary_img, return_distance=True)
        return distance

class CDFBaseTrainDataset(CDFBaseDataset):
    def __init__(self, data_dir, chars, transform=None, extension="png"):
        super().__init__(data_dir, chars, transform, extension)

        self.char_key_dict = {}
        for key, charlist in self.key_char_dict.items():
            for char in charlist:
                self.char_key_dict.setdefault(char, []).append(key)