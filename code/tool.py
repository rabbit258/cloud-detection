import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import local_binary_pattern


def hand(img):

    # 分离R,G,B通道
    b, g, r = cv2.split(img)[:3]

    # 对每个通道进行直方图均衡化处理
    b_eq = cv2.equalizeHist(b)
    g_eq = cv2.equalizeHist(g)
    r_eq = cv2.equalizeHist(r)

    # 合并处理后的通道
    img_eq = cv2.merge((b_eq, g_eq, r_eq))

    # 转换为灰度图
    gray_eq = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)

    # 计算阈值
    threshold = get_threshold(gray_eq)

    # print(threshold)

    # 设置平均超像素大小为500，跑SLIC进行超像素分割
    slic = cv2.ximgproc.createSuperpixelSLIC(img_eq, region_size=22)
    slic.iterate(50)

    # 获取超像素数目
    number_slic = slic.getNumberOfSuperpixels()

    # 下面内容按照论文算法而来，但是除掉了灰度共生矩阵计算的过程
    avg_b = np.zeros(number_slic)
    avg_g = np.zeros(number_slic)
    avg_r = np.zeros(number_slic)
    avg_gray = np.zeros(number_slic)
    cnt = np.zeros(number_slic)
    is_cloud = np.zeros(number_slic)
    belong = slic.getLabels()

    r, c = img.shape[:2]
    for i in range(r):
        for j in range(c):
            label = belong[i][j]
            cnt[label] += 1

            tb, tg, tr = img_eq[i, j]
            avg_b[label] += tb
            avg_g[label] += tg
            avg_r[label] += tr

    for i in range(number_slic):
        avg_gray[i] = avg_r[i] * 0.299 + avg_g[i] * 0.587 + avg_b[i] * 0.114
        if cnt[i] == 0:
            continue
        avg_gray[i] /= cnt[i]
        # print(avg_gray[i])
        if avg_gray[i] >= threshold:
            is_cloud[i] = 1

    radius = 1.0
    npoint = radius * 8

    lbp = local_binary_pattern(gray_eq, npoint, radius, method='ror')
    lbp = np.round(lbp).astype(np.uint8)

    # 设置参数
    distances = [1]
    angles = [0]
    levels = 256

    SLBP = np.zeros(number_slic)

    for i in range(r):
        for j in range(c):
            label = belong[i][j]
            SLBP[label] += lbp[i][j]

    for i in range(number_slic):
        if cnt[i] == 0:
            continue
        # 计算当前超像素的LBP值
        SLBP[i] /= cnt[i]

    aveLBP = np.mean(SLBP)
    queue = []

    # print(aveLBP)
    for i in range(number_slic):
        if is_cloud[i] == 1 and check_not_cloud(SLBP[i], aveLBP):
            is_cloud[i] = 0
        elif is_cloud[i] == 1:
            queue.append(i)

    adj = np.zeros((number_slic, number_slic), dtype=int)

    for i in range(r):
        for j in range(c):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < r and 0 <= nj < c:
                        adj[belong[i, j], belong[ni, nj]] = 1

    adj_list = [[] for _ in range(adj.shape[0])]

    for i in range(number_slic):
        for j in range(number_slic):
            if adj[i, j] == 1:
                adj_list[i].append(j)

    while len(queue) > 0:
        u = queue.pop(0)
        is_cloud[u] = 1
        for v in adj_list[u]:
            if avg_gray[v] >= threshold and is_cloud[v] == 0:
                is_cloud[v] = 1
                queue.append(v)

    return growup(draw(is_cloud, slic.getLabels()), 5)


def resize_image(image):
    # 获取图像的宽度和高度
    height, width = image.shape[:2]

    thr=512
    # 如果图像的像素值大于thrxthr
    if width * height > thr * thr:
        # 计算缩放比例
        scale = (thr * thr / (width * height)) ** 0.5

        # 计算新的宽度和高度
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 对图像进行等比例缩放
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        return resized_image
    else:
        return image


# 参考论文的方法计算阈值
def get_threshold(gray):
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    count = hist.ravel()

    # plt.plot(hist)
    # plt.show()

    # 找到极值点
    extrema = []
    for i in range(161, 253):
        if count[i - 1] < count[i] > count[i + 1] or count[i - 1] > count[i] < count[i + 1]:
            extrema.append(i)

    # 找到符合条件的最小x_i
    for i in range(len(extrema) - 1):
        x_i = extrema[i]
        x_j = extrema[i + 1]
        if abs(count[x_i] - count[x_j]) / max(count[x_i], count[x_j]) >= 0.425:
            return x_i - 2

    # 如果不存在这样的x_i，则返回251
    return 251


# 论文给出的去地物法，数值有微调
def check_not_cloud(SLBP, aveLBP):
    return SLBP < aveLBP * 1.5


# debug用
def draw(is_cloud, label):
    tar = np.zeros(label.shape, dtype=int)
    r, c = label.shape
    for i in range(r):
        for j in range(c):
            if is_cloud[label[i][j]] == 1:
                tar[i][j] = 255
    return tar


# debug用
def draw_mask(slic, img):
    mask_slic = slic.getLabelContourMask()
    mask_inv_slic = cv2.bitwise_not(mask_slic)
    img_slic = cv2.bitwise_and(img, img, mask=mask_inv_slic)  # 在原图上绘制超像素边界
    plt.imshow(img_slic,"gray")
    plt.show()


# 云区域生长，用于补全超像素专有的漏洞
def growup(g_img, loop):
    queue = []
    r, c = g_img.shape[:2]
    for i in range(r):
        for j in range(c):
            if g_img[i, j] == 255:
                queue.append((i, j))
    for _ in range(loop):
        nqueue = []
        for u in queue:
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = u[0] + di, u[1] + dj
                    if 0 <= ni < r and 0 <= nj < c and g_img[ni, nj] != 255:
                        g_img[ni, nj] = 255
                        nqueue.append((ni, nj))
        queue = nqueue

    # plt.imshow(g_img, "gray")
    # plt.show()
    return g_img
