import sys

from matplotlib import pyplot as plt

import tool as tl
import cv2
import numpy as np


# 参数分别为图片路径以及是否开启debug
# 开启debug之后函数将返回第二个参数：图像的云检测二值图
# 否则只返回一个参数：云占百分比
# 本算法参考论文http://html.rhhz.net/CHXB/html/2018-7-996.htm
# 由于算法本身缺陷，在特殊情况下会有比较大的虚警率
def get_precision(img, debug=False):
    # 影像降位到8bit
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 压缩计算量
    img=tl.resize_image(img)

    # 获取大图像的形状
    r, c = img.shape[:2]
    # 初始化结果图像
    res = 0
    if debug:
        result_image = np.zeros((r, c), dtype=img.dtype)

    result_image = tl.hand(img)
    res += np.count_nonzero(result_image == 255)
    res /= r * c

    if debug:
        return res, result_image
    else:
        return res
