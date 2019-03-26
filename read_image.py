import cv2
import numpy as np
import os
import glob

#展示图片
def show_images(images_path, image_size):
    images = []
    path = os.path.join(images_path, '*g')
    files = glob.glob(path) #所有文件全路径列表
    for fl in files:
        image = cv2.imread(fl, 0)  # 通过路径把图读进来(0代表灰度图片)
        image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR) #图片重塑,shape是(256, 256)
        #展示图片
        winname = 'Image ' + str(files.index(fl)+1)
        cv2.imshow(winname, image) #用窗口展示图片
        cv2.waitKey(0) #等待一个字符
        cv2.destroyWindow(winname) #销毁用来展示图片的窗口
        image = np.array(image, dtype=np.float)
        images.append(image)
    return images #(256, 256)灰度图片

print('展示原图...')
path_clear_image = 'Image'
images = show_images(path_clear_image, 256)
# print('展示高斯噪声图...')
# path_gaussian_image = 'Image_Gaussian_Noise'
# gaussian_images = show_images(path_gaussian_image, 256)
# print('展示椒盐噪声图...')
# path_sp_image = 'Image_Salt_and_Pepper_Noise'
# sp_images = show_images(path_sp_image, 256)

#返回原图
def get_images_data():
    return np.array(images) #(9, 256, 256)
# #返回高斯噪声图
# def get_gaussian_data():
#     return np.array(gaussian_images)
# #返回椒盐噪声图
# def get_sp_data():
#     return np.array(sp_images)