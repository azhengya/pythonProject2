# _*_coding:utf-8_*_
# __author: guo
from django.core.management.base import BaseCommand

from datasetManage.settings import DATABASES
from image.models import Image, Settings, Category, Project, Merge, ProjectDataBase, Dataset
from annotation.models import Annotation


class Command(BaseCommand):
    """
    真实图片要当验证集
    """
    data = {
        "dddp": {
            "dataset_manage_531": {
                "rename": {
                    "厨余垃圾_鲜枣": "厨余垃圾_鲜红枣",
                    "厨余垃圾_红枣": "厨余垃圾-红枣核",
                    "其他垃圾_塑料袋装牛奶": "其他垃圾_酸奶袋",
                    "厨余垃圾_生煎包子": "厨余垃圾_生煎",
                    "厨余垃圾_水饺": "厨余垃圾_速冻水饺",
                    # "可回收物_上衣": "可回收物_衣服",
                    # "可回收物_裤子": "可回收物_牛仔裤",
                },
                # "delete": {
                #     "厨余垃圾_葱青", "有害垃圾_咳嗽糖浆玻璃瓶",a
                # },

                "merge": {
                    "可回收物_衣服": [
                        "可回收物_衣服", "可回收物_牛仔裤",
                    ],
                    "可回收物_打汁机": [
                        "可回收物_豆浆机", "可回收物_榨汁机"
                    ],
                    "其他垃圾_纸巾包装袋": [
                        "其他垃圾_纸巾包装袋", "其他垃圾_抽纸塑料袋"
                    ],
                    "可回收物_化妆品瓶": [
                        "可回收物_化妆品瓶", "其他垃圾_化妆品瓶"
                    ],
                    "常见肉类": [
                        "厨余垃圾_鸡胸肉", "厨余垃圾_烤肉", "厨余垃圾-熟里脊肉", "厨余垃圾_牛肉牛排"
                    ],
                    "可回收物_床单被套": [
                        "可回收物_毛毯", "可回收物_床单被套"
                    ],
                    "可回收物_书纸杂志": [
                        "可回收物_书纸", "可回收物_杂志", "可回收物_报纸", "可回收物_测试卷子", "可回收物_纸"
                    ],
                    "厨余垃圾_火腿肠": [
                        "厨余垃圾_烤肠", "其他垃圾_火腿肠"
                    ],
                    "笔": [
                        "其他垃圾_白板笔", "有害垃圾_油漆笔", "其他垃圾_蜡笔", "其他垃圾_眼线笔", "可回收物_木头铅笔",
                        "其他垃圾_荧光笔", "可回收物_笔"
                    ],
                    # heyude
                    "厨余垃圾_生菜包菜": [
                        "厨余垃圾_生菜", "厨余垃圾_包菜"
                    ],
                    "厨余垃圾_豆腐": [
                        "厨余垃圾_豆腐", "厨余垃圾_冻豆腐"
                    ],
                    "可回收物_电风扇": [
                        "可回收物_手持电风扇", "可回收物_电风扇"
                    ],
                    "鞋": [
                        "其他垃圾_一次性拖鞋", "可回收物_鞋", "可回收物_皮鞋"
                    ],
                    "可回收物_滑冰鞋": [
                        "可回收物_双排旱冰鞋", "可回收物_单排旱冰鞋"
                    ],
                    "厨余垃圾_室内垃圾桶": [
                        "厨余垃圾_室内垃圾桶", "空垃圾桶或侧", "闭合垃圾桶"
                    ],
                    "厨余垃圾_鸡蛋": [
                        "厨余垃圾_茶叶蛋", "厨余垃圾_蛋壳", "厨余垃圾_鸡蛋"
                    ],
                },
                "remove": [
                    "可回收物_笔盒", "厨余垃圾_水果果皮", "厨余垃圾_剩菜剩饭",
                    "可回收物_垫子", "厨余垃圾-饺子皮", "厨余垃圾-鸡蛋皮", "可回收物_包装用纸", "del", "unknown",
                    "可回收物_吉他鼓", "可回收物_钢琴", "可回收物_足球", "可回收物_橱柜", "可回收物_门", '可回收物_洗碗机',
                    '可回收物_长笛', '厨余垃圾_奶茶中的珍珠', '可回收物_手风琴', "可回收物_灯罩"
                ]
            }
        }
    }
    data["dddp"]["dataset_manage"] = {
        "merge": data["dddp"]["dataset_manage_531"]["merge"],
        "rename": data["dddp"]["dataset_manage_531"]["rename"],
    }
    databases_dict = {
        v["NAME"]: k
        for k, v in DATABASES.items()
    }

    def handle(self, *args, **options):
        for title, database_items in self.data.items():
            for database, item in database_items.items():
                using = self.databases_dict[database]

                project_object = Project.objects
                merge_object = Merge.objects
                project_database_object = ProjectDataBase.objects

                category_object = Category.objects.using(using)
                ann_object = Annotation.objects.using(using)

                project = project_object.filter(title=title).first()
                pb = project_database_object.filter(project=project, name=database).first()
                if "delete" in item:
                    category_object.filter(value__in=item["delete"]).update(is_active=0)

                if "rename" in item:
                    for new_name, old_name in item["rename"].items():
                        obj = category_object.filter(value=new_name).first()
                        if obj:
                            ann_object.filter(classify__value=old_name).update(classify_id=obj.id)
                            category_object.filter(value=old_name).delete()
                        category_object.filter(value=old_name).update(value=new_name)

                merge_object.filter(pb=pb).delete()

                for class_name, merge_class_name in item["merge"].items():
                    category, _ = category_object.get_or_create(value=class_name, is_active=1)
                    merge_class_name_ids = category_object.filter(value__in=merge_class_name).values_list("id", flat=1)
                    merge_class_name_ids = list(merge_class_name_ids)
                    merge_class_name_ids.append(category.id)
                    merge_class_name_ids = ",".join(map(str, merge_class_name_ids))

                    merge_object.update_or_create(pb=pb, class_name_id=category.id,
                                                  defaults={"merge_class_name": merge_class_name_ids})

                # if "remove" in item:
                #     key = "not_required"
                #     value_list = item["remove"]
                # else:
                #     key = "required"
                #     value_list = item["merge"].keys()
                #
                if "remove" in item:
                    li = category_object.filter(value__in=item["remove"]).values_list("id", flat=1)
                    if not li:
                        return
                    # s = ",".join(map(str, li))
                    # Dataset.objects.update_or_create(pb=pb, defaults={key: s})
                    for db in Dataset.objects.filter(pb=pb, is_active=1):
                        if "train" in db.ratio:
                            if db.required:
                                c = set(db.required.split(",")) - {str(i) for i in li}
                                db.required = ",".join(c)
                            else:
                                c = set(db.not_required.split(",")) | {str(i) for i in li}
                                db.not_required = ",".join(c)
                            db.save()
