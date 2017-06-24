#!/use/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
import sys

def read():
    args = sys.argv
    pil_img = Image.open(args[1])
    img = array(pil_img)
    height, width = img.shape[0:2]
    img[:, 5, :] = 0
    pil_img2 = Image.fromarray(uint8(img))
    outname = args[1] + ".jpg"
    pil_img2.save(outname)

if __name__ == '__main__':
    read()
