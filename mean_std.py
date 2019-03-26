import numpy as np
import read_image

images_data = read_image.get_images_data()
# gaussian_data = read_image.get_gaussian_data()
# sp_data = read_image.get_sp_data()

#计算所有图片的平均值
def cal_imgs_mean(imgs):
    imgs_mean = []
    for img in imgs:
        img_mean = np.mean(img)
        imgs_mean.append(img_mean)
    return imgs_mean

#计算所有图片的标准差
def cal_imgs_std(imgs):
    imgs_std = []
    for img in imgs:
        img_std = np.std(img, ddof=1)
        imgs_std.append(img_std)
    return imgs_std

#最大值
def max_point():
    maxs = []
    for img in images_data:
        maxs.append(img.max())
    return np.array(maxs)
#最小值
def min_point():
    mins = []
    for img in images_data:
        mins.append(img.min())
    return np.array(mins)

#原图均值
def get_images_mean():
    means = cal_imgs_mean(images_data)
    return np.array(means)
#原图标准差
def get_images_std():
    stds = cal_imgs_std(images_data)
    return np.array(stds)
# #高斯噪声均值
# def get_gaussian_mean():
#     means = cal_imgs_mean(gaussian_data)
#     return np.array(means)
# #高斯噪声标准差
# def get_gaussian_std():
#     stds = cal_imgs_std(gaussian_data)
#     return np.array(stds)
# #椒盐噪声均值
# def get_sp_mean():
#     means = cal_imgs_mean(sp_data)
#     return np.array(means)
# #椒盐噪声标准差
# def get_sp_std():
#     stds = cal_imgs_std(sp_data)
#     return np.array(stds)