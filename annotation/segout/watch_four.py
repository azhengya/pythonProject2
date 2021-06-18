import os
import json
import tensorflow as tf
import numpy as np
# import cv2
# import operator
from PIL import Image, ImageDraw, ImageFont, ImageOps
import keras
import matplotlib.pyplot as plt
import matplotlib
import time


class AnalysisWatchImg:

    def __init__(self, path, class_name_json, img_size ,model_file):
        self.path = path
        self.class_name_json = class_name_json
        self.img_size = img_size
        self.model = tf.keras.models.load_model(model_file)
        self.confidence100_map = 8
        self.sub_img_num_column = 7
        self.sub_img_num_line = 6
        self.output_dir = "./check"
        self.model_name = 'ddtn'
        self.sort = False
        self.zhfont1 = matplotlib.font_manager.FontProperties(fname='ukai.ttc')
        self.gan = (1, 2, 8, 9)
        self.shi = (3, 4, 5, 6, 7, 10, 11)
        self.page = 59

    def get_files(self):
        file_path_list = []
        class_name_list = []
        for _, dirs, _ in os.walk(self.path):
            break
        print(dirs)
        for dirName in dirs:
            correct = 0
            print("coming " + dirName)
            for _, _, files in os.walk(self.path + '/' + dirName):
                break
            n = len(files)
            print('files num is ' + str(n))

            for i, file in enumerate(files):
                if file == '.DS_Store':
                    continue
                url = self.path + '/' + dirName + '/' + file
                file_path_list.append(url)
                class_name_list.append(dirName)

                if i >= self.page:
                    break
        print(class_name_list)
        print(len(class_name_list))
        return file_path_list, class_name_list

    def read_json(self):
        with open(self.class_name_json, 'r') as fp:
            class_name = json.load(fp)
            if isinstance(class_name, dict):
                class_name_temp = []
                for k, v in class_name.items():
                    class_name_temp.append(v)
            print(len(class_name))
            print(class_name)
            return class_name_temp

    def imgs_predic(self,url_list):
        lengh = len(url_list)
        img_shape = (lengh,) + (self.img_size, self.img_size) + (3,)
        # print(img_shape)
        img_arrays = np.zeros(img_shape, dtype=np.float32)
        # print(img_arrays)

        for i, url in enumerate(url_list):
            try:
                img = tf.keras.preprocessing.image.load_img(url, target_size=(self.img_size, self.img_size))
            except Exception as e:
                print(url, e)
                continue

            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_arrays[i] = img_array

        ret = self.model.predict(img_arrays)
        return ret, img_arrays

    # def forecast_values(self, d):
    #     poor = []
    #     lt_r = []
    #     lt = d['confidence'][:6]
    #
    #     value = (sum(i[1] for i in lt) / 6) - 0.038
    #     value_1 = (tf.clip_by_value(value,0.02,0.25)).numpy()
    #
    #     for i in lt:
    #         if i[1] >= value_1:
    #             lt_r.append(i)
    #             poor.append(i[0])
    #         elif i[1] == 0.0 and lt.index(i) == 0:
    #             lt_r.append(i)
    #             poor.append(i[0])
    #         else:
    #             lt_r.append((0, 0.0))
    #
    #     if lt_r[0][1] == 0.0:
    #         kong_value = 1.0
    #         return 'empty_unknow'
    #     elif lt_r[0][1] != 0.0 and operator.ge(set(self.gan), set(poor)) == True:
    #         gan_value = sum(i[1] for i in lt_r) / len(lt_r)
    #         return 'dry'
    #     elif lt_r[0][1] != 0.0 and operator.ge(set(self.shi), set(poor)) == True:
    #         shi_value = sum(i[1] for i in lt_r) / len(lt_r)
    #         return 'wet'
    #     else:
    #         s = 0
    #         j = 0
    #         for i in lt_r:
    #
    #             if operator.ge(set(self.gan), set((i[0],))):
    #                 s += i[0]
    #             elif operator.ge(set(self.shi), set((i[0],))):
    #                 j += i[0]
    #             else:
    #                 continue
    #         if s < j:
    #             su = (s / (s + j)) * 2
    #         elif s > j:
    #             su = (j / (s + j)) * 2
    #         else:
    #             su_a = (j / (s + j)) * 2
    #             su = (tf.clip_by_value(su_a,0,1)).numpy()
    #         return 'mix'



    def get_autocontrast_confidence(self, inputs, class_name_list):
        '''
        autocontrast_info = {
            "img" : img,      #autocontrast img
            "confidence":{}
            "result":干垃圾
        }
        '''

        autocontrast_info_list = []
        main_out = inputs[0]  # [b, 505]
        seg_out = inputs[1]  # [b, 96,96,505]
        b, w, h, c = seg_out.shape
        bath_size = b
        a = np.ones(shape=(h, w))
        b = np.zeros(shape=(h, w))

        seg_out = tf.where(tf.less(seg_out, 2 / c), b[..., None], seg_out)
        grad = tf.argmax(seg_out, axis=-1)
        top_class = tf.argsort(main_out, direction='DESCENDING')
        count = 0
        for i in range(bath_size):
            # print('img' + str(i))
            autocontrast_info = {}
            confidence_dic = {}
            for target_index in range(c):
                confidence_tensor = tf.where(tf.cast(grad[i], dtype=tf.int32) == target_index,
                                             seg_out[i, :, :, target_index], b)
                conf = tf.reduce_sum(confidence_tensor) * self.confidence100_map / (w * h)
                # print(tf.reduce_sum(confidence_tensor).numpy())

                conf = tf.clip_by_value(conf, 0.0, 1.0)
                # print(conf)
                conf = round(conf.numpy(), 3)
                confidence_dic[target_index] = conf

            autocontrast_info['confidence'] = sorted(confidence_dic.items(), key=lambda x: x[1], reverse=True)

            # value = self.forecast_values(autocontrast_info)
            #
            #
            # if value == class_name_list[range(bath_size).index(i)]:
            #     count += 1
            #
            #
            #
            # autocontrast_info['result'] = value
            # print(value)
            mask = np.argmax(seg_out[i], axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            img = ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
            autocontrast_info['img'] = np.array(img)

            autocontrast_info_list.append(autocontrast_info)
        print('正确的有：' + str(count))
        # print(autocontrast_info_list)
        return autocontrast_info_list

    def plt_img(self, imgs_class_name, imgs, intput_array, ds_class_name, info):
        fig = None
        img_name = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        img_num = 0

        for i, array in enumerate(intput_array):

            array = [
                (ds_class_name[k] + str(v), array[:, :, k]) for k, v in info[i].get('confidence')
            ]
            # print(array)

            array.insert(0, ("seg_out", info[i].get('img') / 255.0))
            array.insert(0, (info[i].get('result'), imgs[i]))

            if i % self.sub_img_num_line == 0:
                if img_name and fig:
                    plt.savefig(os.path.join(self.output_dir, img_name))
                    plt.close()
                # 新的一张图片
                fig = plt.figure(figsize=(self.sub_img_num_line * 3, self.sub_img_num_column * 3), dpi=150., clear=True)
                img_name = str(time.time()) + ".png"
                plt.title(self.model_name)

            for j in range(self.sub_img_num_column):
                ax = fig.add_subplot(self.sub_img_num_line, self.sub_img_num_column,
                                     img_num % (self.sub_img_num_line * self.sub_img_num_column) + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.get_xaxis().set_visible(True)
                ax.set_title(array[j][0], fontproperties=self.zhfont1)
                ax.imshow(array[j][1] / 255.)
                img_num += 1
        plt.savefig(os.path.join(self.output_dir, img_name))
        plt.close()


    def main(self, *args, **options):
        file_list, class_name_list = self.get_files()
        class_name = self.read_json()
        ret, imgs = self.imgs_predic(url_list=file_list)
        info = self.get_autocontrast_confidence(ret, class_name_list)
        self.plt_img(imgs_class_name=class_name_list, imgs=imgs, intput_array=ret[1], ds_class_name=class_name, info=info)
        exit(0)



class_name_json = '/Users/dd-26/dd/lihenglong/EfficientNet/ddtn_thirty-six_class_name.json'
DATA_SET_PATH = '/Users/dd-26/dd/lihenglong/EfficientNet/yanzheng/'
model_file = '/Users/dd-26/dd/lihenglong/EfficientNet/ddtn_thirty-six_2021-05-31_model_111_b5.h5'
img_size = 456
AnalysisWatchImg(DATA_SET_PATH, class_name_json, img_size, model_file).main()
