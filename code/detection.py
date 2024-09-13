#coding:utf-8
import cv2
import matplotlib.pyplot as plt

import final as fl
import time

t1 = time.time()
path_in = input()
path_out = input()
img = cv2.imread(path_in)
fd = open(path_out,"w")
#debug为true时，将返回第二个额外变量，为一张二值图，代表检测的结果
# pre, timg = fl.get_precision(img, debug=True)

#debug为false时，只返回一个参数，代表云涵盖率
pre = fl.get_precision(img)
fd.write(str(pre))
fd.close()
# plt.imshow(timg, "gray")
# plt.show()
