# _*_coding:utf-8_*_
# __author: guo
import json
import os

import numpy as np
from django.core.management.base import BaseCommand

import tensorflow as tf
from tensorflow import keras

from image.models import Category, Image
from sentry_sdk import capture_exception

from utils.img_gen import load_img, img_to_array

np.set_printoptions(suppress=True)


class Command(BaseCommand):
    model_file = "model.h5"
    conf_file = "conf.json"
    class_file = "class_name.json"
    img_suffix = {"jpg", "jpeg", "JPEG", "png", "JPG", "PNG", "gif", "bmp"}
    one_num = 100
    top_num = 100000

    model_path = None
    # assert model_path
    using = None

    def get_float(self, f, num, is_just=False):
        s = f"%.{num}f"
        f = s % float(f)
        su = f.strip('0')
        f = f"0{su}"
        if is_just:
            f = f.ljust(num, "0")
        return f[:num]

    def get_file_data(self, json_file):
        with open(json_file, "r") as fr:
            data = json.loads(fr.read())
        return data

    def gen_keras_data(self, img_list, size, is_resize=False):
        if is_resize:
            batch_x = np.zeros((len(img_list), *size, 3))
            for i, _img_path in enumerate(img_list):
                img = load_img(_img_path, target_size=size)
                x = img_to_array(img, size[0])
                batch_x[i] = x
            return batch_x

        img = []
        for _img_path in img_list:
            try:
                _x = tf.keras.preprocessing.image.load_img(_img_path, target_size=size)
            except Exception as e:
                capture_exception(e)
                print(_img_path, e)
                continue

            _x = tf.keras.preprocessing.image.img_to_array(_x)
            _x = np.expand_dims(_x, axis=0)
            img.append(_x)

        x = np.concatenate([x for x in img])
        return x

    def get_model(self):
        model = keras.models.load_model(os.path.join(self.model_path, self.model_file))
        size = self.get_file_data(os.path.join(self.model_path, self.conf_file))["img_size"]
        class_names = list(self.get_file_data(os.path.join(self.model_path, self.class_file)).values())

        return model, size, class_names

    def gen_category_obj(self, class_names):
        class_name__id_dict = dict(Category.objects.using(self.using).filter(is_active=1).values_list("value", "id"))

        for class_name in set(class_names) - set(class_name__id_dict):
            obj = Category.objects.using(self.using).create(value=class_name)
            class_name__id_dict[class_name] = obj.id

        return class_name__id_dict

    def create_bulk_data(self, objs, model):
        print()
        num = 500
        for i in range(0, len(objs), num):
            print(f"\r _create_bulk_data {model}: {i + num}", end="")
            _objs = objs[i: i + num]
            try:
                model.objects.using(self.using).bulk_create(_objs)
            except Exception as e:
                for obj in _objs:
                    try:
                        obj.save(using=self.using)
                    except Exception as e:
                        pass
                        # capture_exception(e)

        print()

    def handle(self, *args, **options):
        GPUS = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)

        model, size, class_names = self.get_model()

        class_name__id_dict = self.gen_category_obj(class_names)
        # self.gen_img_obj(path_dict, source)
        # self.gen_classify_obj(model, size, class_name__id_dict, class_names)
