# _*_coding:utf-8_*_
# __author: guo
import ast
import datetime
import hashlib
import multiprocessing
import os

from django.core.management import BaseCommand
from sentry_sdk import capture_exception

from annotation.management.commands.constant import gen_merge_data
from annotation.models import Annotation
from datasetManage.settings import DATABASES
from image.models import Project, Category, Merge, Settings, ProjectDataBase, Dataset


def cp_img(src, dist):
    sh = f"cp '{src}' '{dist}'"
    os.system(sh)


class Command(BaseCommand):
    is_cover = Settings.objects.filter(key="is_cover").values_list("value", flat=1).first() == "1"  # 训练集是否补位
    to_path = Settings.objects.filter(key="to_path").values_list("value", flat=1).first()
    to_111_path = Settings.objects.filter(key="to_111_path").values_list("value", flat=1).first()
    databases_dict = {
        v["NAME"]: k
        for k, v in DATABASES.items()
    }
    ratio = {"train": 0.7, "valid": 0.3}
    no_valid_class_names = {
        "其他垃圾_墙板地面",
        "桌面"
    }
    true_train_num = 60  # 真实图片大于多少张，才会替换网络图片
    train_num = 3  # 类别大于多少张才会训练

    def _rm(self, files, cur_dir):
        files = list(files)
        for i in range(0, len(files), 50):
            # print(f"\r {num}: {class_name}: {i}", end="")
            _cp_files = [f"'{os.path.join(cur_dir, f)}'" for f in files[i:i + 50]]
            sh = f"rm -rf  {' '.join(_cp_files)}"
            os.system(sh)

    def filter_data(self, data, num):
        result = {}
        for class_name in data:
            if len(data[class_name]) > num:
                result[class_name] = data[class_name]

        return result

    def cp_or_mv_img(self, cur_dir, cur_img_path_list, pool):
        local_img_name_set = set(os.listdir(cur_dir))
        img_name__local_path_dict = {}
        for img_path in cur_img_path_list:
            key = f"{hashlib.md5(img_path.encode()).hexdigest()}.{img_path.rsplit('.', 1)[-1]}"
            img_name__local_path_dict[key] = img_path

        # 删除图片
        self._rm(list(local_img_name_set - set(img_name__local_path_dict.keys())), cur_dir)
        # 添加图片
        for img_name in set(img_name__local_path_dict.keys()) - local_img_name_set:
            local_path = img_name__local_path_dict[img_name]
            pool.apply_async(cp_img, (local_path, os.path.join(cur_dir, img_name)))
            # sh = f"cp '{local_path}' '{os.path.join(cur_dir, img_name)}'"
            # os.system(sh)

    def get_now(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_data(self, project, name="dataset_manage"):
        # 获取数据，从网络数据库还是真实数据库
        """
        {
            类名: []  # 类下所有图片
        }
        :return:
        """
        result = {}
        for pb in ProjectDataBase.objects.filter(is_active=1, project=project, name=name):
            category_object = Category.objects.using(self.databases_dict[pb.name])
            annotation_object = Annotation.objects.using(self.databases_dict[pb.name])
            class_name_id__value_dict = dict(category_object.filter(is_active=1).values_list("id", "value"))
            for db in Dataset.objects.filter(is_active=1, pb=pb):
                # for db in Dataset.objects.filter(is_active=1, pb_id=2):
                _result = {}
                required_class_name_ids = {int(i) for i in db.required.split(",")}

                # 获取所需要的图片
                queryset = annotation_object.filter(is_active=1, status=1, classify_id__in=required_class_name_ids) \
                    .values_list("classify_id", "img__local_path")
                # print(len(queryset))
                for class_name_id, local_path in queryset:
                    class_name = class_name_id__value_dict[class_name_id]
                    result.setdefault(class_name, []).append(local_path)
                    # _result.setdefault(class_name, []).append(local_path)

                # print(pb.name, list(_result.keys()))
        return result

    def gen_project_data(self, _project):
        """
        获取项目所需要数据
        :return:
        """
        if _project == "all":
            query = {}
        else:
            query = {"title": _project}
        # 获取项目
        for project in Project.objects.filter(is_active=1, **query):
            true_data = self.get_data(project)
            web_data = self.get_data(project, name="dataset_manage_531")

            for class_name in list(true_data.keys()):
                if len(true_data[class_name]) >= self.true_train_num:
                    web_data[class_name] = true_data.pop(class_name, [])
                else:
                    web_data.setdefault(class_name, []).extend(true_data.pop(class_name, []))

            data = gen_merge_data(web_data, _project)
            # 过滤图片，大于多少才是有效的
            data = self.filter_data(data, self.train_num)
            # 切分图片
            result = {}
            key__ratio_dict = self.ratio
            for class_name, local_img_path_list in data.items():
                local_img_path_list.sort()
                if class_name in self.no_valid_class_names:
                    result.setdefault("train", {})
                    result["train"].setdefault(class_name, set(local_img_path_list))
                    continue

                start, end = 0, 0
                for num, (key, ratio) in enumerate(key__ratio_dict.items()):
                    start = end
                    if num == len(key__ratio_dict) - 1:
                        end = len(local_img_path_list)
                    else:
                        end = int(end + len(local_img_path_list) // sum(key__ratio_dict.values()) * ratio)
                    # if class_name == "其他垃圾_墙板地面":
                    #     print(start, end, len(local_img_path_list), len(set(local_img_path_list[start:end])))
                    #     print(key, ratio, key__ratio_dict, )

                    result.setdefault(key, {})
                    result[key].setdefault(class_name, set())
                    result[key][class_name] |= set(local_img_path_list[start:end])

            train = set(result["train"])
            print("train", train, len(train))
            for key in result:
                if key != "train":
                    for del_key in set(result[key]) - train:
                        result[key].pop(del_key)

            yield project, result

    def del_data(self, cur_dir, li):
        for i in li:
            sh = f"rm -rf '{os.path.join(cur_dir, i)}'"
            os.system(sh)

    def handle(self, *args, **options):
        # 获取当前项目需要的类别列表
        # 根据类别列表获取图片
        # 如果图片存在则不做修改，如果文件夹里包含多出来的图片，则移除
        pool = multiprocessing.Pool(processes=os.cpu_count() * 2)
        manage_class_names_data = {}
        _project = "dddp"

        # data = {train: {class_name: [img_list], ...}, valid: {class_name: [img_list], ...}, ....}
        for project, data in self.gen_project_data(_project):
            project_dir = os.path.join(self.to_path, project.title)
            os.makedirs(project_dir, exist_ok=True)
            manage_class_names_data[project_dir] = {"main": "train"}
            # 删除多余的数据集
            local_dataset = set(os.listdir(project_dir))
            self.del_data(project_dir, local_dataset - set(data))

            for dataset, class_name__img_list_dict in data.items():
                if dataset != "train":
                    manage_class_names_data[project_dir].setdefault("other", []).append(dataset)
                if self.is_cover:
                    class_name__img_list_dict["占位符"] = []

                sum_num = sum([len(v) for _, v in class_name__img_list_dict.items()])
                print(f"cp img {dataset}...", sum_num, self.get_now())
                dataset_dir = os.path.join(project_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)

                # 删除多余的类
                local_class_name = set(os.listdir(dataset_dir))
                self.del_data(dataset_dir, local_class_name - set(class_name__img_list_dict))

                # 添加图片
                for class_name, img_list in class_name__img_list_dict.items():
                    cur_dir = os.path.join(dataset_dir, class_name)
                    os.makedirs(cur_dir, exist_ok=True)
                    self.cp_or_mv_img(cur_dir, img_list, pool)

        pool.close()
        pool.join()

        # 处理训练集和验证集类对应
        for _dir in manage_class_names_data:
            main = manage_class_names_data[_dir]["main"]
            other = manage_class_names_data[_dir].get("other", [])
            main_dir = os.path.join(_dir, main)

            main_class_names = set(os.listdir(main_dir))
            for o in other:
                other_dir = os.path.join(_dir, o)
                other_class_names = set(os.listdir(other_dir))

                # 删除多余的
                self._rm(other_class_names - main_class_names, other_dir)
                # 添加没有的
                for file in main_class_names - other_class_names:
                    path = os.path.join(other_dir, file)
                    os.makedirs(path, exist_ok=True)

        print("rsync...", self.get_now())
        for project in Project.objects.filter(is_active=1, title=_project):
            path78 = os.path.join(self.to_path, project.title)
            path111 = os.path.join(self.to_111_path, project.title)
            sh = f"rsync -rtu --delete {path78}/ 111:{path111}/"
            print("rsync: ", sh, self.get_now())
            os.system(sh)
            print(f"rsync {project.title} end...", self.get_now())
        print("end...", self.get_now())
