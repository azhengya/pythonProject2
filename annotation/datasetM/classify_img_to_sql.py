# _*_coding:utf-8_*_
# __author: guo
import json
import os

import numpy as np
from django.core.management.base import BaseCommand

from django.db import transaction
import tensorflow as tf
from tensorflow import keras

from annotation.models import Annotation
from image.models import Category, Image, Settings
from sentry_sdk import capture_exception

from utils.oss_download import main as oss_download_main
from annotation.management.commands import base_classify_img_to_sql


np.set_printoptions(suppress=True)


class Command(base_classify_img_to_sql.Command):
    model_file = "model.h5"
    conf_file = "conf.json"
    class_file = "class_name.json"
    img_suffix = {"jpg", "jpeg", "JPEG", "png", "JPG", "PNG", "gif", "bmp"}
    one_num = 64
    top_num = 100000
    model_path = Settings.objects.filter(key="model_path").values_list("value", flat=True).first()
    # assert model_path
    using = Settings.objects.filter(key="using_classify_img_to_sql").values_list("value", flat=True).first() or "default"

    def _get_img_dict(self):
        result = {}
        paths = Settings.objects.filter(key="img_paths").values_list("value", flat=1).first()
        paths = paths.split(",")
        for path in paths:
            for root, _, files in os.walk(path):
                for file in files:
                    if "." in file and file.rsplit(".", 1)[1] in self.img_suffix:
                        oss_path = root.replace("/home/dd/Share/dataset_pool/imgs/dd-eco-ai-picture/", "")
                        oss_path = f"http://dd-eco-ai-picture.oss-cn-hangzhou.aliyuncs.com/{oss_path}/{file}"
                        result[os.path.join(root, file)] = oss_path
        return result

    def gen_classify_obj(self, model, size, class_name__id_dict, class_names):

        # 查询哪些图片没有分类，则去分类
        img_ids = Annotation.objects.using(self.using).filter(is_active=1).values_list("img_id", flat=1)
        img_list = list(Image.objects.using(self.using).filter(is_active=1, source__lte=1000).exclude(id__in=img_ids))

        objs = []

        for i in range(0, len(img_list), self.one_num):
            _img_obj_list = img_list[i: i + self.one_num]
            img_path = [obj.local_path for obj in _img_obj_list]

            img_data = self.gen_keras_data(img_path, size)
            predictions = model(img_data, training=False)[0]

            # 获取前n的识别结果
            for index, one_pred in enumerate(predictions):
                print(f"\r 识别分类: {i + index + 1}", end="")

                try:
                    index_sort_list = np.argsort(one_pred)[::-1]

                    other_classify = []
                    other_pred = []
                    # print(self.top_num)
                    # print(one_pred)
                    for j in range(1, min(self.top_num, len(one_pred))):
                        classify_index = index_sort_list[j]
                        # if one_pred[classify_index] < 0.001:
                        #     break
                        p = self.get_float(one_pred[classify_index], 10, is_just=True)
                        other_pred.append(p)
                        other_classify.append(str(class_name__id_dict[class_names[classify_index]]))

                    other_pred = ",".join(other_pred)
                    other_classify = ",".join(other_classify)
                    classify = class_name__id_dict[class_names[index_sort_list[0]]]
                    pred = f"{self.get_float(one_pred[index_sort_list[0]], 10, is_just=True)}_{class_names[index_sort_list[0]]}"
                    objs.append(
                        Annotation(
                            img=_img_obj_list[index], classify_id=classify, pred=pred,
                            other_classify=other_classify, other_pred=other_pred
                        )
                    )
                except Exception as e:
                    print(e)
                    capture_exception(e)

        self.create_bulk_data(objs, Annotation)

    def gen_img_obj(self, path_dict, source):
        img_obj_list = []
        for index, (local_path, oss_path) in enumerate(path_dict.items()):
            img_obj_list.append(Image(local_path=local_path, oss_path=oss_path, source=source))
        self.create_bulk_data(img_obj_list, Image)

    def handle(self, *args, **options):
        GPUS = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)

        source = int(options["source"])
        if source == 1000:
            path_dict = self._get_img_dict()
        elif source == 0:
            path_dict = oss_download_main()
            print("download: ", len(path_dict))
        else:
            return

        if isinstance(path_dict, list):
            path_dict = {i: "" for i in path_dict}

        model, size, class_names = self.get_model()

        class_name__id_dict = self.gen_category_obj(class_names)
        self.gen_img_obj(path_dict, source)
        self.gen_classify_obj(model, size, class_name__id_dict, class_names)

    def add_arguments(self, parser):
        parser.add_argument("source", type=int)
