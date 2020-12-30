import random
import os
import cv2
import pandas as pd


###
# 数据预处理：将数据裁剪，并区分训练集、验证集、测试集
###

# 函数：找到图像对应的信息，包括人脸bbox位置、人脸标签等
# 输入：图像名image_name, 图像信息矩阵info_list
# 输出： Dataframe类型数据，包含图像的标签、bbox等信息
def find_name(image_name, info_list):
    row = info_list[info_list['name'] == image_name]
    return row

def out_info(list, out_name):
    with open(out_name,'w') as f:
        for item in list:
            for subitem in item:
                f.write(str(subitem) + ' ')
            f.write('\n')



# 创建文件夹
if not os.path.exists('../train'):
    os.mkdir('../train')

if not os.path.exists('../val'):
    os.mkdir('../val')

if not os.path.exists('../test'):
    os.mkdir('../test')

# 读取图像标签、bbox等信息
info = pd.read_table('./label.lst', header=None, sep=' ')
info.columns = ['name', 'face_id_in_image', 'face_box_top', 'face_box_left', 'face_box_right',
                'face_box_bottom', 'face_box_cofidence', 'expression_label']

# 读取图像数据
data_folder = '../all/origin'
all_image = os.listdir(data_folder)

train_info = []
val_info = []
test_info = []
for i,image in enumerate(all_image):
    # 截取图片人脸
    image_path = os.path.join(data_folder, image)
    img = cv2.imread(image_path, 1)
    img_info = find_name(image, info)
    try: # 排除找不到图像信息的情况
        img_info['name']
        crop_image = img[int(img_info['face_box_top']):int(img_info['face_box_bottom']),
                     int(img_info['face_box_left']):int(img_info['face_box_right']), :]

        # 将图片随机分为训练集、验证集、测试集；分配比例为7：1：2
        tag = random.randint(1, 10)
        if tag == 9 or tag == 10:
            cv2.imwrite(os.path.join('../test', image), crop_image)
            test_info.append([list(img_info['name'])[0], list(img_info['expression_label'])[0]])
        elif tag == 8:
            cv2.imwrite(os.path.join('../val', image), crop_image)
            val_info.append([list(img_info['name'])[0], list(img_info['expression_label'])[0]])
        else:
            cv2.imwrite(os.path.join('../train', image), crop_image)
            train_info.append([list(img_info['name'])[0], list(img_info['expression_label'])[0]])
    except:
        continue

    if i % 1000 == 0:
        print(i)

# 存储训练集标签
out_info(train_info, 'train_info.txt')
out_info(val_info, 'val_info.txt')
out_info(test_info, 'test_info.txt')

import numpy as np

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
