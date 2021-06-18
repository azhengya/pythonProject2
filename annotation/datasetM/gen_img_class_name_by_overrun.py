# _*_coding:utf-8_*_
# __author: guo
import os

import numpy as np
from django.core.management.base import BaseCommand

import tensorflow as tf

from annotation.models import Annotation
from image.models import Category, Image, Settings
from annotation.management.commands import classify_img_to_sql

# os.environ["CUDA_VISIBLE_DEVICES"] = ''


class Command(BaseCommand):
    """
    获取超限分类识别结果
    """
    db = Settings.objects.filter(key="db_name").values_list("value", flat=True).first() or ""
    overrun_class_name_ids = Settings.objects.filter(key="overrun").values_list("value", flat=True).first() or ""
    overrun_class_name_ids = overrun_class_name_ids.split(",")
    c = classify_img_to_sql.Command()

    def gen_category_obj(self, class_names):
        class_name__id_dict = dict(
            Category.objects.using(self.db).filter(is_active=1).values_list("value", "id"))

        return class_name__id_dict

    def handle(self, *args, **options):
        GPUS = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in GPUS:
            tf.config.experimental.set_memory_growth(gpu, True)

        model, size, class_names = self.c.get_model()

        class_name__id_dict = self.gen_category_obj(class_names)
        img_list = list(
            Image.objects.using(self.db).filter(is_active=1, ann__status=1, ann__classify_id__in=self.overrun_class_name_ids))
        for i in range(0, len(img_list), self.c.one_num):
            _img_obj_list = img_list[i: i + self.c.one_num]
            img_path = [obj.local_path for obj in _img_obj_list]

            img_data = self.c.gen_keras_data(img_path, size)
            predictions = model.predict(img_data)[0]

            # 获取前n的识别结果
            for index, one_pred in enumerate(predictions):
                print(f"\r 识别分类: {i + index + 1}", end="")

                index_sort_list = np.argsort(one_pred)[::-1]

                other_classify = []
                other_pred = []
                # print(self.top_num)
                # print(one_pred)
                for j in range(1, min(self.c.top_num, len(one_pred))):
                    classify_index = index_sort_list[j]
                    # if one_pred[classify_index] < 0.001:
                    #     break
                    if class_names[classify_index] in class_name__id_dict:
                        other_pred.append(str(round(one_pred[classify_index], 5)))
                        other_classify.append(str(class_name__id_dict[class_names[classify_index]]))

                other_pred = ",".join(other_pred)
                other_classify = ",".join(other_classify)
                # classify = class_name__id_dict[class_names[index_sort_list[0]]]
                class_name = class_names[index_sort_list[0]]
                pred = str(round(one_pred[index_sort_list[0]], 5))
                Annotation.objects.using(self.db).filter(img=_img_obj_list[index]) \
                    .update(pred=f"{pred}_{class_name}", other_pred=other_pred, other_classify=other_classify)
