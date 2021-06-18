# _*_coding:utf-8_*_
# __author: guo
import datetime
import os

import numpy as np

import tensorflow as tf

from annotation.models import Annotation
from image.models import Settings
from annotation.management.commands import base_classify_img_to_sql

np.set_printoptions(precision=20)
"""
每个模型只跑自己需要跑的类
    跑模型的时候，调用记录当前模型跑的都是哪些类
 """
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


get_session(10 / 1)


class Command(base_classify_img_to_sql.Command):
    not_update_class_name_ids = Settings.objects.filter(key="not_update_class_name_ids") \
                                    .values_list("value", flat=1).first() or ""
    not_update_class_name_ids = [int(i) for i in not_update_class_name_ids.split(",") if i]

    model_path = Settings.objects.filter(key="model_path_classify_img_to_sql_by_two").values_list("value", flat=True) \
        .first()
    using = Settings.objects.filter(key="using_classify_img_to_sql_by_two").values_list("value",
                                                                                        flat=True).first() or "default"
    one_num = 32

    def gen_classify_obj(self, model, size, class_name__id_dict, class_names, status, mv_classify,
                         is_save_cur_class_name):
        # 查询哪些图片没有被人工分类，则去使用新模型重新分类
        # classify_id = [1344,1677,1678,1683,1772,1744,1687,1702,1685,1686,1768,1689,1766,1771,1690,1732,1743,1779,1779,1737,1729,1782,1722,1731,1767,1693,1721,1688,1773,1736,1730,1699,1696,1785,1774,1712,1769,1691,1776,1738,1715,1765,1777,1755,1781,1697,1780,1698,1735,1784,1703,1749,1761,1786,1733,1714,1708,1787,1750,1701,1728,1710,1694,1806,1796,]
        classify_id=[1344,1610,1721,1779,1744,1843,1743,1677,1789,1691,1780,1749,1750,1784,1699,1785,1787,1688,1685
            ,1686,1683,1690,1687,1689,1781,1678,1694,1731,1732,1765,1698,1733,1708,1779,1711,1693,1730,1696,1744
            ,1833,1712,1738,1761,1844,1843,1722,1715,1729,1786,1714,1776,1736,1777,1743,1203,1702,1782,1773,1766
            ,1772,1771,1768,1767,1769,1703,1710,1755,1789,1538,1533,1484,1225,1267,1318,1125 ,1292,1627,1572,1108
            ,1073,1488,1279 ,1349,1449,1602,1486]
        # classify_id = [1699,1740,1344,1677,1742,1683,1678,1744,1743,1745,1685,1689,1686,1610,1766,1687]
        img_list = Annotation.objects.using(self.using) \
            .filter(is_active=1, status=status, classify_id__in=classify_id)\
            .values("id","img__local_path","classify_id").order_by("-classify_id")
        img_list = list(img_list)
        print(mv_classify == 1)
        re_class_names = {v: i for i, v in enumerate(class_names)}
        id__class_name_dict = {v: k for k, v in class_name__id_dict.items()}

        for i in range(0, len(img_list), self.one_num):
            now = datetime.datetime.now()
            _img_list = img_list[i: i + self.one_num]
            img_path = [item["img__local_path"] for item in _img_list]

            img_data = self.gen_keras_data(img_path, size)
            predictions = model(img_data, training=False)
            main_out = predictions[0]
            queryset = Annotation.objects.using(self.using).filter(id__in=[item["id"] for item in _img_list]) \
                .values("id", "classify_id", "classify__value")
            ann_items = {i["id"]: i for i in queryset}

            for index, one_pred in enumerate(main_out):
                print(f"\r 识别分类({_img_list[0]['classify_id']}): {i + index + 1}", end="")
                other_classify = []
                other_pred = []
                # print(float(one_pred[index_sort_list[0]]))

                flag = False
                index_sort_list = list(np.argsort(one_pred)[::-1])

                if is_save_cur_class_name:
                    ann_item = ann_items[_img_list[index]["id"]]
                    clf_id = ann_item["classify_id"]
                    class_name = id__class_name_dict[clf_id]

                    if class_name in re_class_names:
                        class_name_id = re_class_names[class_name]
                        pred = self.get_float(one_pred[class_name_id], 7, is_just=True)
                        pred = f"{pred}_{class_names[class_name_id]}"
                        flag = True
                        # print(class_name_id, class_name, class_names[class_name_id])
                        del_value = class_name_id
                        top_index = index_sort_list[0]

                if not flag:
                    # if class_names[index_sort_list[0]] == "厨余垃圾_室内垃圾桶":
                    #     _index = 1
                    # else:
                    #     _index = 0
                    _index = 0
                    pred = self.get_float(one_pred[index_sort_list[_index]], 7, is_just=True)
                    pred = f"{pred}_{class_names[index_sort_list[_index]]}"
                    # 0.36945_可回收物_电风扇
                    del_value = index_sort_list[_index]
                    top_index = del_value
                    # index_sort_list = index_sort_list[1:]

                index_sort_list.remove(del_value)

                for j in range(min(self.top_num, len(index_sort_list))):
                    classify_index = index_sort_list[j]
                    # print(one_pred[classify_index].numpy())
                    # other_pred.append(str(round(one_pred[classify_index].numpy(), 20)))
                    other_pred.append(self.get_float(one_pred[classify_index], 20))
                    other_classify.append(str(class_name__id_dict[class_names[classify_index]]))

                other_pred = ",".join(other_pred)
                other_classify = ",".join(other_classify)

                if mv_classify == 1:
                    # if float(pred.split('_')[0]) > 0.9:
                    classify = class_name__id_dict[class_names[top_index]]
                    Annotation.objects.using(self.using).filter(id=_img_list[index]["id"]) \
                        .update(
                        pred=pred, classify=classify, other_classify=other_classify, other_pred=other_pred,
                        u_time=now
                    )
                else:
                    Annotation.objects.using(self.using).filter(id=_img_list[index]["id"]) \
                        .update(
                        pred=pred, other_classify=other_classify, other_pred=other_pred,
                        u_time=now
                    )

    def handle(self, *args, **options):
        # GPUS = tf.config.experimental.list_physical_devices(device_type='GPU')
        # for gpu in GPUS:
        #     tf.config.experimental.set_memory_growth(gpu, True)

        status = int(options["status"])
        mv_classify = int(options["mv_classify"])
        is_save_cur_class_name = int(options["is_save_cur_class_name"])

        model, size, class_names = self.get_model()

        class_name__id_dict = self.gen_category_obj(class_names)
        self.gen_classify_obj(model, size, class_name__id_dict, class_names, status, mv_classify,
                              is_save_cur_class_name)

    def add_arguments(self, parser):
        # 选择筛选过的图片还是未筛选过的数据, 0 未筛选过，1 筛选过
        parser.add_argument("status", type=int)
        # 是否移动分类，0否1是
        parser.add_argument("mv_classify", type=int)
        # 是否把当前类别保存为第一置信度
        parser.add_argument("is_save_cur_class_name", type=int)
