# _*_coding:utf-8_*_
# __author: guo
from django.core.management import BaseCommand
import shutil
import ast
import os
import copy

from datasetManage.settings import img_suffix, BASE_DIR
from annotation.management.commands.classify_img_to_sql import Command as clasify_img_to_sql_command

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import numpy as np


class Command(BaseCommand):
    c_command = clasify_img_to_sql_command()

    def write_outlier_csv(self, outlier_map, threshold, output_dir, dir_num):
        li = list(outlier_map.values())
        li = sorted(li, key=lambda x: sum([i["loss"] < threshold for i in x]) / len(x), reverse=True)
        for index, data in enumerate(li):
            dir_path = os.path.join(output_dir, "outlier", f"{index % dir_num}", f"{index % dir_num}")
            os.makedirs(dir_path, exist_ok=True)
            n = sum([i["loss"] < threshold for i in data]) / len(data)
            # _class_name = data[0]['class_name']
            with open(os.path.join(dir_path, f"{str(index).zfill(5)}_{data[0]['class_name']}_{n}.csv"), "w") as f:
                f.write("\t".join(["class", "img", "loss", "pred_class_name", "pred"]))
                f.write("\n")
                for item in data:
                    f.write("\t".join(
                        [
                            item["class_name"],
                            "/".join(item["img_path"].rsplit("/", 2)[1:]),
                            str(item["loss"]),
                            item["pred_class_name"],
                            f"{item['pred']}"
                        ]
                    )
                    )
                    f.write("\n")

    def gen_class_name_and_img_files(self, img_dirs, class_names, pre_data):
        # li = ['Baidu_0224.jpeg', 'Baidu_0225.jpeg', 'Baidu_0226.jpeg', 'Baidu_0228.jpeg', 'Baidu_0230.jpeg']
        # yield "厨余垃圾_白菜", [f"/Users/guo/Downloads/{i}" for i in li]
        labels_li = ["厨余垃圾_室内垃圾桶"]
        labels = os.listdir(img_dirs[0])
        labels.sort()
        for label in labels:
            # if label not in labels_li:
            #     continue

            if label not in class_names or label in pre_data:
                continue
            yield label, [
                os.path.join(img_dir, label, file)
                for img_dir in img_dirs
                for file in os.listdir(os.path.join(img_dir, label))
                if len(file.rsplit(".", 1)) == 2 and file.rsplit(".", 1)[1] in img_suffix
            ]

    def get_loss(self, img_dirs, output_dir=".", threshold=.0075, dir_num=4):
        """
        class_names: ['other', '垃圾包', '垃圾桶', '塑料袋', '散垃圾']
        """
        # model, *_ = cls.get_model_and_class_name("model")

        pre_data = {}
        if os.path.exists("bak.json"):
            with open("bak.json", "r") as f:
                pre_data = ast.literal_eval(f.read())

        if os.path.exists(os.path.join(output_dir, "outlier")):
            shutil.rmtree(os.path.join(output_dir, "outlier"))

        model, size, class_names = self.c_command.get_model()

        qs_path = os.path.join(output_dir, "qs")
        os.path.exists(qs_path) and shutil.rmtree(qs_path)

        success_num = 0
        for_num = 100
        outlier_map = pre_data  # {class_name: [{"img_path": "", "loss": 0}]}
        class_name = ""
        sum_num = 0
        try:
            for num, (class_name, img_paths) in enumerate(
                    self.gen_class_name_and_img_files(img_dirs, class_names, pre_data), 1):
                class_index = class_names.index(class_name)

                _label = copy.copy([0] * len(class_names))
                _label[class_names.index(class_name)] = 1
                _label = np.array(_label)
                loss_arrays = np.empty([0, len(_label)])
                for i in range(0, len(img_paths), for_num):
                    print(f"\rclass_num: {num}, class_name: {class_name}, img_num: {i} ~ {i + for_num}", end="")
                    _img_paths = img_paths[i: i + for_num]
                    sum_num += len(_img_paths)
                    x = self.c_command.gen_keras_data(_img_paths, size)
                    predictions = model.predict(x)

                    for index, p in enumerate(predictions):
                        if class_name == class_names[p.argmax()]: success_num += 1
                        loss_arrays = np.append(loss_arrays, np.array([p]), axis=0)
                        outlier_map.setdefault(class_name, []).append(
                            {
                                "img_path": _img_paths[index],
                                "loss": p[class_index],
                                "class_name": class_name,
                                "pred_class_name": class_names[p.argmax()],
                                "pred": p.max()
                            }
                        )

                print()
                qs = np.mean(loss_arrays, axis=0)
                os.makedirs(qs_path, exist_ok=True)
                with open(os.path.join(qs_path, f"{class_name}.csv"), "w") as qs_fw:
                    qs_fw.write(",".join(["class", "avg"]))
                    for index in range(0, len(qs)):
                        avg = qs[index]
                        line = ",".join([class_names[index], f"{avg}"])
                        qs_fw.write("\n")
                        qs_fw.write(line)
            self.write_outlier_csv(outlier_map, threshold, output_dir, dir_num)
            print("accuracy: ", success_num, sum_num, success_num / sum_num)
        finally:
            outlier_map.pop(class_name, "")
            with open("bak.json", "w") as f:
                print(outlier_map, file=f)

    def handle(self, *args, **options):
        img_dirs = [
            # "/home/dd/Share/dataset_pool/dd531/train",
            "/home/dd/Share/dataset_pool/dd531/valid"
        ]
        self.get_loss(
            img_dirs,
            threshold=0.5,
            output_dir=os.path.join(BASE_DIR, "data")
        )
