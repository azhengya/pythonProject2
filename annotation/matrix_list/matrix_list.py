import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# model_file = '/Users/dd-26/dd/lihenglong/EfficientNet/ddtn_thirty-six_2021-05-31_model_111_b5.h5'
model_file = '/home/daidai/heyude/daidai531/b4/ddtn_thirty-six_2021-05-31_model_111_b5.h5'
model = tf.keras.models.load_model(model_file)
DATA_SET_PATH = '/home/daidai/heyude/data_set/dataset/ddtn_thirty-six/valid'
# DATA_SET_PATH = '/Users/dd-26/dd/lihenglong/EfficientNet/yanzheng/valid'
json_file = '/Users/dd-26/dd/lihenglong/EfficientNet/ddtn_thirty-six_class_name.json'
json_file = '/home/daidai/heyude/daidai531/b4/ddtn_thirty-six_class_name.json'


# DATA_SET_PATH = '/Users/dd-26/dd/lihenglong/EfficientNet/yanzheng/valid'


def imgs_predic(url_list):
    img_size = 456
    lengh = len(url_list)
    img_shape = (lengh,) + (img_size, img_size) + (3,)
    # print(img_shape)
    img_arrays = np.zeros(img_shape, dtype=np.float32)
    # print(img_arrays)

    for i, url in enumerate(url_list):
        try:
            img = tf.keras.preprocessing.image.load_img(url, target_size=(img_size, img_size))
        except Exception as e:
            print(url, e)
            continue

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_arrays[i] = img_array

    ret = model.predict(img_arrays)
    return ret

# def img_predic(url):
#     img_size = 456
#     try:
#         img = tf.keras.preprocessing.image.load_img(url, target_size=(img_size, img_size))
#     except Exception as e:
#         print(url, e)
#
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = img_array[None, ...]
#
#     ret = model.predict(img_array)
#     return ret


if __name__ == '__main__':

    model = tf.keras.models.load_model(model_file)
    with open(json_file, 'r') as f:
        data = f.read()
        labels_name = [k for i, k in json.loads(data).items()]

    class_mean_conf = []
    labels = []

    for dirName in labels_name:
        print("coming " + dirName)
        for _, _, files in os.walk(DATA_SET_PATH + '/' + dirName):
            n = len(files)
            labels.append(dirName + str(n))
        print('files num is ' + str(n))
        num = 200
        confidence = []
        url_list = []
        for i, file in enumerate(files):

            if file == '.DS_Store':
                continue
            url = DATA_SET_PATH + '/' + dirName + '/' + file
            url_list.append(url)
        # print(url_list)
        for j in range(0, len(files), num):
            _file_list = url_list[j:j + num]
            # print(_file_list)

            main_out, _ = imgs_predic(_file_list)
            print(main_out.shape)
            # print(main_out)
    #         #     exit(0)
            confidence.append(main_out)

        if len(confidence) == 0:
            aa = tf.zeros((38, 38))
            confidence.append(aa)


        confusion_matrix_row = tf.concat(confidence, axis=0)
    #
        print(confusion_matrix_row.shape)
    #
        confusion_matrix_row = tf.reduce_mean(confusion_matrix_row, axis=0)
        print(tf.argmax(confusion_matrix_row))
        # confusion_matrix_row = confusion_matrix_row.round()
        confusion_matrix_row = tf.cast(confusion_matrix_row * 1000, dtype=tf.uint32)
        print(confusion_matrix_row.shape)

        class_mean_conf.append(confusion_matrix_row)
        print(confusion_matrix_row)
        print()
    #     # exit(0)
    #
    confusion_matrix = tf.stack(class_mean_conf, axis=0)
    confusion_matrix = confusion_matrix.numpy()
    print(confusion_matrix)
    print(confusion_matrix.shape)
    print(labels)
    #
    _, ax = plt.subplots(figsize=(30, 30))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")
    plt.savefig('hunxiao_train.jpg')
