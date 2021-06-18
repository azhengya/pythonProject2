# _*_coding:utf-8_*_
# __author: guo
import bisect
import os

import numpy as np
from django.core.management import BaseCommand
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from annotation.models import Annotation
from datasetManage.settings import BASE_DIR
from image.models import Settings

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
ignore_list = [
]

ignore_list = [f"{i}.csv" if not i.endswith(".csv") else i for i in ignore_list]


class Command(BaseCommand):
    # todo 待修改，从数据库拿数据
    using = Settings.objects.filter(key="confusion_matrix_using").values_list("value", flat=1) or "default"

    def get_qs_num(self, path, x):
        # 如果第
        # 获取类别平均置信度
        # 判断平均置信度和排第二的比例，如果比例接近则去做混淆矩阵
        class_li = []
        queryset = Annotation.objects.using(self.using).filter(is_active=1) \
            .values("classify_id", "pred", "other_pred", "other_classify").order_by("classify_id")

        data = {}
        class_num = len(queryset[0]["other_pred"].split(",")) + 1
        class_name_ids_order = [i for i in range(class_num)]
        for item in queryset:
            data.setdefault(item["classify_id"], []).append(item)

        classify_id__mean_preds_dict = {}

        for classify_id, li in data.items():
            arr = np.zeros((len(li), class_num))
            for index, item in enumerate(li):
                pred = float(item["pred"].split("_", 1)[0])
                preds = [float(i) for i in item["other_pred"].split(",") if i]
                bisect.insort(preds, pred)
                index = bisect.bisect(preds, pred)
                class_name_ids = [int(i) for i in item["other_classify"].split(",") if i]
                class_name_ids.insert(index, item["classify_id"])
                class_id__pred_dict = {i: preds[i] for i in class_name_ids}

                arr[index] = np.array([class_id__pred_dict[i] for i in class_name_ids_order])

            qs = np.mean(arr, axis=0)
            arg_arr = np.argsort(qs)[::-1][:2]
            pred1 = qs[arg_arr[0]]
            pred2 = qs[arg_arr[1]]
            if pred2 * x > pred1:
                class_li.extend([classify_id, arg_arr[0], arg_arr[1]])
                classify_id__mean_preds_dict[classify_id] = qs

        class_li = list(set(class_li))
        confusion_matrix = np.zeros([len(class_li), class_num])

        _files = []
        all_files = set()
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    with open(os.path.join(root, file), "r") as f:
                        data = [i.split(",") for i in f.read().split("\n")][1:]
                    data.sort(key=lambda l: float(l[1]), reverse=True)
                    if not all_files:
                        all_files = set([z[0] for z in data])
                    one = data[0]
                    two = data[1]
                    # other = data[2:]
                    if one[0] != file.replace(".csv", ""):
                        print("one != file", one, two)

                    # other_sum = sum([float(i[1]) for i in other]) / len(other) * x
                    # print(two[1], other_sum)
                    if float(two[1]) * x > float(one[1]):
                        for i in {file, f"{one[0]}.csv", f"{two[0]}.csv"}:
                            if i not in _files:
                                _files.append(i)
                        print(True, file)
        print(len(_files), _files)
        xxx = [i.replace(".csv", "") for i in _files]
        all_files -= set(xxx)
        return _files
        # print(len(_files), _files)

    def get_qs_num(self, path, x):
        # 如果第
        _files = []
        all_files = set()
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    with open(os.path.join(root, file), "r") as f:
                        data = [i.split(",") for i in f.read().split("\n")][1:]
                    data.sort(key=lambda l: float(l[1]), reverse=True)
                    if not all_files:
                        all_files = set([z[0] for z in data])
                    one = data[0]
                    two = data[1]
                    # other = data[2:]
                    if one[0] != file.replace(".csv", ""):
                        print("one != file", one, two)

                    # other_sum = sum([float(i[1]) for i in other]) / len(other) * x
                    # print(two[1], other_sum)
                    if float(two[1]) * x > float(one[1]):
                        for i in {file, f"{one[0]}.csv", f"{two[0]}.csv"}:
                            if i not in _files:
                                _files.append(i)
                        print(True, file)
        print(len(_files), _files)
        xxx = [i.replace(".csv", "") for i in _files]
        all_files -= set(xxx)
        return _files
        # print(len(_files), _files)

    def gen_np(self, path, _files):
        num_classes = len(_files)
        confusion_matrix = np.zeros([0, num_classes])

        for file in _files:
            with open(os.path.join(path, file), "r") as f:
                data = f.read()
                data = {i.split(",")[0]: i.split(",")[1] for i in data.split("\n") if
                        f'{i.split(",")[0]}.csv' in _files}
            data = [int(float(data[i.replace(".csv", "")]) * 100) for i in _files]
            confusion_matrix = np.append(confusion_matrix, np.array([data]), axis=0)

        return confusion_matrix

    def show(self, _files, confusion_matrix, x, qs):
        # Display a confusion matrix.
        labels = [
            i.replace(".csv", "")
            for i in _files
        ]
        # plt.figure(figsize=(20, 20))
        # x = plt.figure(dpi=800)
        _, ax = plt.subplots(figsize=(50, 50))
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
        disp.plot(include_values=True, cmap="viridis", ax=ax, xticks_rotation="vertical")

        # plt.show()
        plt.savefig(os.path.join(BASE_DIR, f"data/cm_img/qs_{qs.rsplit('/', 1)[1]}_{x}_2.png"))

    def handle(self, *args, **options):
        qs = os.path.join(BASE_DIR, f"data/qs")
        x = 1.5
        result = self.get_qs_num(qs, x)
        confusion_matrix = self.gen_np(qs, result)
        self.show(result, confusion_matrix, x, qs)
