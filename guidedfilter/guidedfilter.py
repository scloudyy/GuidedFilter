#!/use/bin/env python3
# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
from numpy.matlib import repmat
import sys

def read():
    args = sys.argv
    pil_img = Image.open(args[1])
    img = array(pil_img)
    (height, width) = img.shape[0:2]
    t = array([[1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1]])
    boxfilter(t, 3)
    guidedfilter(t,t,3,1)
    pil_img2 = Image.fromarray(uint8(img))
    outname = args[1] + ".jpg"
    pil_img2.save(outname)

def guidedfilter(I, p, r, eps):
    """
    GUIDEDFILTER: O(1) time implementation of guided filter.

    ----------
    :param I: guidance image (should be a gray-scale/single channel image)
    :param p: filtering input image (should be a gray-scale/single channel image)
    :param r: local window radius
    :param eps: regularization parameter
    :return:
    """
    (hei, wid) = I.shape[0:2]
    N = boxfilter(ones((hei,wid)), r)
    mean_I = divide(boxfilter(I, r), N)
    mean_p = divide(boxfilter(p, r), N)
    mean_Ip = divide(boxfilter(multiply(I, p), r), N)
    cov_Ip = mean_Ip - multiply(mean_I, mean_p)

    mean_II = divide(boxfilter(multiply(I, I), r), N)
    var_I = mean_II - multiply(mean_I, mean_I)

    a = divide(cov_Ip, (var_I + eps))
    b = mean_p - multiply(a, mean_I)

    mean_a = divide(boxfilter(a, r), N)
    mean_b = divide(boxfilter(b, r), N)

    dst = multiply(mean_a, I) + mean_b
    return dst

def boxfilter(src, r):
    """
    BOXFILTER: O(1) time box filtering using cumulative sum.

    ----------
    :param src: input (should be a gray-scale/single channel image)
    :param r: local window radius
    :return: dst(x, y)=sum(sum(src(x-r:x+r,y-r:y+r)))
    """
    (hei, wid) = src.shape[0:2]
    dst = zeros((hei, wid))
    
    cum = cumsum(src, axis=0)
    dst[0:r+1, :] = cum[r:2*r+1, :]
    # 末尾到第r+1个元素，因为以该元素为中心的局部区域正好都在有效范围内，即不需要使用累加值相减的方法得到累加值
    dst[r+1:hei-r, :] = cum[2*r+1:hei, :] - cum[0:hei-2*r-1, :]
    # 区域底部累加值减去顶部减一的累加值是区域内元素的累加值，
    # 区域中心从全局第一个需要累加值相减的元素(偏移量r+1)
    # 到最后一个可用此方法的元素(height-1-r)，此时直接是最大值减去半径r
    dst[hei-r:hei, :] = repmat(cum[hei-1:hei, :], r, 1) - cum[hei-2*r-1:hei-r-1, :]

    cum = cumsum(dst, axis=1)
    dst[:, 0:r+1] = cum[:, r:2*r+1]
    dst[:, r+1:wid-r] = cum[:, 2*r+1:wid] - cum[:, 0:wid-2*r-1]
    dst[:, wid-r:wid] = repmat(cum[:, wid-1:hei], 1, r) - cum[:, wid-2*r-1:wid-r-1]
    return dst

if __name__ == '__main__':
    read()
