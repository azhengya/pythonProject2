# _*_coding:utf-8_*_
# __author: guo
import os

import numpy as np
from django.core.management.base import BaseCommand

import tensorflow as tf
from tensorflow.python.keras.engine import data_adapter

from annotation.models import Annotation
from image.models import Category, Image, Settings
from annotation.management.commands import classify_img_to_sql
from utils.img_gen import MyImgGen

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from annotation.management.commands import base_classify_img_to_sql


class Command(base_classify_img_to_sql.Command):
    """
    获取超限分类识别结果
    """
    model_path = Settings.objects.filter(key="precision_model").values_list("value", flat=True).first() or ""
    img_dir = Settings.objects.filter(key="precision_img_dir").values_list("value", flat=True).first() or ""
    csv_dir = Settings.objects.filter(key="precision_csv_dir").values_list("value", flat=True).first() or ""
    batch_size = 64

    def gen_img_list(self, bs):
        class_names = []
        img_paths = []
        for root, _, files in os.walk(self.img_dir):
            if files:
                for file in files:
                    if file.rsplit(".", 1)[-1] in self.img_suffix:
                        class_names.append(root.rsplit("/", 1)[-1])
                        img_paths.append(os.path.join(root, file))

        for i in range(0, len(class_names), bs):
            yield class_names[i: i + bs], img_paths[i: i + bs]

    def handle(self, *args, **options):
        model, size, class_names = self.get_model()
        img_gen = MyImgGen()
        val_gen = img_gen.flow_from_directory(
            self.img_dir,
            target_size=size,  # all images will be resized to 150x150
            batch_size=self.batch_size,
            # save_to_dir=data_set_path+'/train_gen', save_prefix='good', save_format='jpg',
            shuffle=False,
            class_mode='categorical'
        )
        try:
            val_logs = model.evaluate(
                x=val_gen,
                # y=val_y,
                # sample_weight=val_sample_weight,
                batch_size=32,
                steps=10,
                max_queue_size=64,
                workers=16,
                return_dict=True)
            print(val_logs)
        except:
            pass

        data = {}
        class_id__class_name_dict = {v: k for k, v in val_gen.class_indices.items()}

        for num, (x, y) in enumerate(val_gen):
            if num == len(val_gen):
                break

            predictions = model(x, training=False)
            main_out = predictions[0]
            # main_out = predictions
            arg_main = tf.argmax(main_out, axis=1)
            y = tf.argmax(y, axis=1)
            y = y.numpy()
            for i, one_class_id in enumerate(arg_main):
                _sum = num * self.batch_size + i + 1
                _success = sum([data[i]['success'] for i in data])
                print(f"\r {num}, {i}, {_sum}, {_success}: {_success / _sum}", end="")
                class_name = class_id__class_name_dict[y[i]]
                _class_name = class_names[one_class_id]
                data.setdefault(class_name, {"sum": 0, "success": 0})
                data[class_name]["sum"] += 1

                if class_name == _class_name:
                    data[class_name]["success"] += 1
                # else:
                #     print(class_name, _class_name)

        with open(os.path.join(self.csv_dir, f'{self.model_path.rsplit("/", 1)[-1]}.csv'), "w") as fw:
            fw.write(",".join(["类别", "总数", "成功"]))
            for class_name in data:
                _sum = data[class_name]["sum"]
                success = data[class_name]["success"]
                s = f"\n{class_name},{_sum},{success}"
                fw.write(s)

        print("\nend...")
        return
        data = {}

        for num, (class_name_list, img_path_list) in enumerate(self.gen_img_list(self.batch_size)):
            x = self.gen_keras_data(img_path_list, size, is_resize=False)
            predictions = model.predict(x)
            main_out = predictions[0]
            # main_out = predictions
            arg_main = tf.argmax(main_out, axis=1)

            for i, one_class_id in enumerate(arg_main):
                print(f"\r {num}, {i}, {num * self.batch_size + i + 1}", end="")
                class_name = class_name_list[i]
                _class_name = class_names[one_class_id]
                data.setdefault(class_name, {"sum": 0, "success": 0})
                data[class_name]["sum"] += 1

                if class_name == _class_name:
                    data[class_name]["success"] += 1

        with open(os.path.join(self.csv_dir, f'{self.model_path.rsplit("/", 1)[-1]}.csv'), "w") as fw:
            fw.write(",".join(["类别", "总数", "成功"]))
            for class_name in data:
                _sum = data[class_name]["sum"]
                success = data[class_name]["success"]
                s = f"\n{class_name},{_sum},{success}"
                fw.write(s)

        print("\nend...")
