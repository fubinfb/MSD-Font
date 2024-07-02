"""
FFG-benchmarks
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
from calendar import c
from fileinput import filename
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont, ImageDraw
import numpy as np
from torch import save


def get_defined_chars(fontfile):
    ttf = TTFont(fontfile)
    chars = [chr(y) for y in ttf["cmap"].tables[0].cmap.keys()]
    return chars

def get_defined_chars_general(fontfile):
    ttf = TTFont(fontfile)
    m_dict = ttf.getBestCmap()
    unicode_list = []
    for key, _ in m_dict.items():
        unicode_list.append(key)

    char_list = [chr(ch_unicode) for ch_unicode in unicode_list]
    return char_list

# def get_defined_chars_for_FZfont(fontfile):
#     ttf = TTFont(fontfile)
#     Glyphorder_table = ttf.getGlyphOrder()
#     Nf = len(Glyphorder_table)
#     chars = []
#     for i in range(Nf):
#         glyname = Glyphorder_table[i]
#         # glyID = ttf['cmap'].tables[0].ttFont.getGlyphID(glyname)
#         if glyname[0:3] == 'uni':
#             hexnum = Glyphorder_table[3:].lower()


#         chars.append(chr(glyID))
#     return chars

def get_filtered_chars(fontpath):
    ttf = read_font(fontpath)
    defined_chars = get_defined_chars_general(fontpath)
    # defined_chars = get_defined_chars(fontpath)
    # defined_chars = get_defined_chars_for_FZfont(fontpath)
    avail_chars = []

    for char in defined_chars:
        img = np.array(render(ttf, char))
        if img.mean() == 255.:
            pass
        else:
            avail_chars.append(char.encode('utf-16', 'surrogatepass').decode('utf-16'))

    return avail_chars


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
