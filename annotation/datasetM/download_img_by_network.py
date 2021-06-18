# _*_coding:utf-8_*_
# __author: guo
import os
import sys
import re
from annotation.management.commands import classify_img_to_sql
from django.core.management.base import BaseCommand
from django.db.models import Count
from datasetManage.settings_local import download_local_path
from annotation.models import Annotation
from image.models import Image, Settings
from django.db import transaction
import shutil


class Command(BaseCommand):
    """
    读取分类下图片如果小于50张，则去寻找一些图片补充进来
    1. 如果当前分类在531分类里，则从531分类补充过来，同时图片名字要改一下名字，添加一个标示
    2. 如果当前分类不在531分类里，则需要从网上下载数据，也要有一个标示
    已有下载图片的脚本
    """
    gen_command = classify_img_to_sql.Command()

    def get_classify_lt_50(self):
        # 获取小于50张图片的分类
        queryset = Annotation.objects.filter(is_active=1, status=1, classify_id__gt=1000).values_list("classify_id") \
            .annotate(classify_count=Count("classify_id")).filter(classify_count__lte=50) \
            .values_list("classify__value", "classify_id")
        return dict(queryset)

    def download(self, class_names):
        """
        :param class_names:
        :yield: class_name, cur_dir
        """
        # 531的路径
        train = '/Users/daidai/train'
        # yield '厨余垃圾-土豆丝', "/Users/guo/PycharmProjects/daidaiProject/datasetManage/tests"

        # 1. 如果当前分类在531分类里，则从531分类复制过来，同时图片名字要改一下名字，添加一个标示
        # 2. 如果当前分类不在531分类里，则需要从网上下载数据，也要有一个标示
        for class_name in class_names:
            if class_name in os.listdir(train):
                for picture in os.listdir(train + '/' + class_name):
                    new_obj_name = '531' + picture
                    source = train + '/' + class_name + '/' + picture
                    target = '' + new_obj_name
                    shutil.copy(source, target)
                    print("添加531完成")

            else:
                keyword = re.split(r'-_', str(class_name))[-1]
                arv = 'python image_downloader.py {} -e baidu -n 300 -o {}'.format(keyword, download_local_path + '/' + class_name)
                # todo main(arv)

                cur_dir = ""
        yield class_name, cur_dir

    def gen_obj(self, cur_dir, classify_id):

        try:
            local_path__img_dict = {}
            local_path_list = [
                os.path.join(cur_dir, local_path)
                for local_path in os.listdir(cur_dir)
                if "." in local_path and local_path.rsplit(".", 1)[1] in self.gen_command.img_suffix
            ]
            with transaction.atomic():
                for local_path in local_path_list:
                    local_path__img_dict[local_path] = Image.objects.create(
                        local_path=local_path, oss_path="", source=1001
                    )

                ann_objs = []
                for local_path in local_path_list:
                    img = local_path__img_dict[local_path]
                    ann_objs.append(
                        Annotation(img=img, classify_id=classify_id, pred="0")
                    )
                self.gen_command.create_bulk_data(ann_objs, Annotation)
        except Exception as e:
            shutil.rmtree(cur_dir)
            raise e

    def handle(self, *args, **options):
        download_is_run = int(Settings.objects.filter(key="download_is_run").values_list("value", flat=1).first())
        if download_is_run:
            return

        try:
            Settings.objects.filter(key="download_is_run").update(value="1")
            class_names_dict = self.get_classify_lt_50()
            class_names = list(class_names_dict)
            # 使用这些class_name去下载网络数据
            for class_name, cur_dir in self.download(class_names):
                self.gen_obj(cur_dir, class_names_dict[class_name])
        except Exception as e:
            Settings.objects.filter(key="download_is_run").update(value="0")
            raise e
