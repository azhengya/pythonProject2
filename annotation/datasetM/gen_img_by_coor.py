# _*_coding:utf-8_*_
# __author: guo
import os

import numpy as np
from PIL import ImageFont, ImageDraw
from cv2 import cv2
from django.core.management.base import BaseCommand

from annotation.models import Annotation
from datasetManage.settings import DATABASES, img_suffix, BASE_DIR
from image.models import Image, Category, Settings
import PIL as pil
from annotation.management.commands import classify_img_to_sql
import tensorflow as tf


class Command(BaseCommand):
    to_path = "/home/dd/Share/dataset_pool/pil_img"
    # to_path = "/Users/guo/PycharmProjects/daidaiProject/datasetManage/data"
    class_name_ids = Settings.objects.filter(key="test_gen_coor_img").values_list("value", flat=1).first() or ""
    class_name_ids = [int(i) for i in class_name_ids.split(",") if i]
    using = Settings.objects.filter(key="test_gen_coor_img_using").values_list("value", flat=1).first() or "default"
    ttf_path = os.path.join(BASE_DIR, "conf", "ShangShouYuSongTi-2.ttf")
    fnt = ImageFont.truetype(ttf_path, 15)
    c = classify_img_to_sql.Command()
    model, size, class_names = c.get_model()
    size = size[0]
    sh = f"rm -rf '{to_path}/'"
    os.system(sh)
    print(model)

    def gen_img(self):
        queryset = Annotation.objects.using(self.using).filter(is_active=1, status=1,
                                                               classify_id__in=self.class_name_ids)
        result = {}
        for obj in queryset:
            class_name = obj.classify.value
            img_path = obj.img.local_path
            result.setdefault(class_name, []).append(img_path)

        return result

    def call(self, inputs, class_name):
        main_out = inputs[0]  # [b, 505]
        seg_out = inputs[1]  # [b, 96,96,505]

        # print(main_out.shape)  # [b,96,96]
        # print(seg_out.shape)

        b, w, h, c = seg_out.shape
        th_area = w * h // 100
        bath_size = b

        grad = tf.argmax(seg_out, axis=-1)
        top_class = tf.argsort(main_out, direction='DESCENDING')

        batch_result = []

        b = tf.zeros(shape=grad[0].shape)
        # print(top_class)

        for i in range(bath_size):
            result = []
            # print()
            # print("top_class[i]", top_class[i])
            # print("main_out[i]", main_out[i])
            for j, target_index in enumerate(top_class[i]):
                # print(main_out[i,target_index])
                # if j >= 3:
                #     break

                gloabal_ratio = main_out[i,target_index]
                if gloabal_ratio < 0.1:
                    break

                coor = tf.where(tf.cast(grad[i], dtype=tf.int32) == target_index)
                # print(coor) # shape=(1818, 2), dtype=int6 # x,y 坐标

                # if len(coor) < th_area:
                #     break
                pred = {}
                pred["name"] = class_name[target_index]
                confidence = tf.where(tf.cast(grad[i], dtype=tf.int32) == target_index, seg_out[i, :, :, target_index],b)
                confidence = tf.reduce_sum(confidence)/len(coor)
                # print("confidence", main_out[i][target_index])
                # confidence = tf.reduce_mean(confidence)
                # pred["confidence"] = confidence.numpy()
                pred["confidence"] = confidence
                print(confidence)
                # print(pred["confidence"])
                if pred["confidence"] < 0.3:
                    continue

                # print(confidence)
                y1 = tf.reduce_min(coor[:, 0]) * self.size / w
                x1 = tf.reduce_min(coor[:, 1]) * self.size / h
                y2 = tf.reduce_max(coor[:, 0]) * self.size / w
                x2 = tf.reduce_max(coor[:, 1]) * self.size / h
                x1 = x1.numpy()
                y1 = y1.numpy()
                x2 = x2.numpy()
                y2 = y2.numpy()

                pred["bbox"] = (x1, y1, x2, y2)
                xc = (x1 + x2) / 2.0
                yc = (y1 + y2) / 2.0
                pred["centre"] = (xc, yc)

                result.append(pred)
            # break

            batch_result.append(result)
        # print(batch_result)

        return batch_result

    def get_contours(self, thresh=None):
        if cv2.__version__[0] >= str(2):
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL RETR_LIST
        else:
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # get轮廓
    def get_accept_contours(self, contours):

        contoursAreas = []
        accept_contours = []
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            contoursAreas.append(cv2.contourArea(hull))
            # contoursAreas.append(cv2.contourArea(cnt))
        max_contour = max(contoursAreas)
        for i, c_a in enumerate(contoursAreas):
            if c_a * 2 > max_contour:
                accept_contours.append(contours[i])
        return accept_contours

    def get_counters_call(self, inputs, class_name):
        '''

        :param inputs:
        :param class_name:
        :param img_size:
        :return:

        result：
        [
            {
            name = “手机”
            confidence = “0.6”
            coor = [
                {
                bbox = (x1,y1,x2,y2)
                center = (x,y)
                }
                {
                bbox = (x1,y1,x2,y2)
                center = (x,y)
                }
                 ]
            }

            {
            name = “鼠标”
            confidence = “0.8”
            coor = [
                {
                bbox = (x1,y1,x2,y2)
                center = (x,y)
                }
                 ]
            }
        ]
        '''

        main_out = inputs[0]  # [b, 505]
        seg_out = inputs[1]  # [b, 96,96,505]

        b,w,h,c = seg_out.shape
        bath_size = b

        grad = tf.argmax(seg_out, axis=-1)
        top_class = tf.argsort(main_out, direction='DESCENDING')

        batch_result = []
        result = []

        b = tf.zeros(shape=grad[0].shape)

        for i in range(bath_size):
            if i==3:
                break
            print()
            print('img '+str(i))
            for target_index in top_class[i]:
                if class_name[target_index] == "无效图片":
                    continue

                pred = {}
                print('gloabal_ratio')
                print(main_out[i,target_index])
                gloabal_ratio = main_out[i,target_index]
                if gloabal_ratio < 0.1:
                    break

                coor = tf.where(tf.cast(grad[i], dtype=tf.int32) == target_index)
                #print(coor) # shape=(1818, 2), dtype=int6 # x,y 坐标
                confidence_tensor = tf.where(tf.cast(grad[i], dtype=tf.int32)==target_index, seg_out[i, :, :, target_index], b)
                confidence = tf.reduce_sum(confidence_tensor)/len(coor)
                confidence = confidence.numpy()
                print('confidence')
                print(confidence)
                print(class_name[target_index])
                if confidence < 0.3:
                    break
                pred["confidence"] = confidence
                pred["name"] = class_name[target_index]

                edges = np.uint8(confidence_tensor.numpy() * 255)
                ret, thresh = cv2.threshold(src=edges, thresh=1, maxval=255, type=cv2.THRESH_BINARY)
                # x = thresh
                # print(type(x), x.shape, x.dtype, np.min(x), np.max(x))
                # print(x[26:36, 20:30])

                contours = self.get_contours(thresh)
                if len(contours) == 0:
                    # warnings.warn('predict top1 have no contours,dirty or background img !')
                    print(class_name[target_index])
                    print()
                    continue
                contours = self.get_accept_contours(contours)

                coors = []
                for cn in contours:
                    coor = {}
                    x, y, w, h = cv2.boundingRect(cn)
                    coor["bbox"] = (x, y, x+w, y+h)
                    xc = x + w / 2
                    yc = y + h / 2
                    coor["center"] = (xc, yc)
                    coors.append(coor)
                pred["coor"] = coors

                result.append(pred)

            batch_result.append(result)

        return batch_result

    def get_model_pred(self, img_path_list):
        x = self.c.gen_keras_data(img_path_list, (self.size, self.size))
        predictions = self.model.predict(x)
        predictions = self.get_counters_call(predictions, self.class_names)
        # predictions = self.call(predictions, self.class_names)
        return predictions

    def draw_img(self, img_path, pred):
        """
        [
        {
        'confidence': 0.835955, 'name': '鞋', 'coor': [
        {'bbox': (55, 55, 60, 60), 'center': (57.5, 57.5)},
        {'bbox': (0, 55, 5, 60), 'center': (2.5, 57.5)},
        {'bbox': (23, 45, 28, 52), 'center': (25.5, 48.5)},
        {'bbox': (24, 36, 28, 40), 'center': (26.0, 38.0)},
        {'bbox': (36, 32, 44, 36), 'center': (40.0, 34.0)},
        {'bbox': (36, 24, 44, 31), 'center': (40.0, 27.5)},
        {'bbox': (55, 0, 60, 6), 'center': (57.5, 3.0)},
         {'bbox': (43, 0, 46, 3), 'center': (44.5, 1.5)},
         {'bbox': (35, 0, 40, 2), 'center': (37.5, 1.0)},
         {'bbox': (0, 0, 5, 5), 'center': (2.5, 2.5)}]},
          {'confidence': 0.8343279, 'name': '鞋', 'coor': [{'bbox': (0, 58, 5, 60), 'center': (2.5, 59.0)}, {'bbox': (55, 55, 60, 60), 'center': (57.5, 57.5)}, {'bbox': (23, 43, 29, 49), 'center': (26.0, 46.0)}, {'bbox': (31, 40, 35, 44), 'center': (33.0, 42.0)}, {'bbox': (40, 30, 43, 33), 'center': (41.5, 31.5)}, {'bbox': (55, 0, 60, 5), 'center': (57.5, 2.5)}, {'bbox': (3, 0, 5, 3), 'center': (4.0, 1.5)}]}, {'confidence': 0.80052024, 'name': '可回收物_袜子', 'coor': [{'bbox': (40, 41, 44, 45), 'center': (42.0, 43.0)}, {'bbox': (23, 40, 28, 45), 'center': (25.5, 42.5)}, {'bbox': (33, 35, 39, 42), 'center': (36.0, 38.5)}, {'bbox': (15, 29, 19, 34), 'center': (17.0, 31.5)}, {'bbox': (39, 19, 44, 25), 'center': (41.5, 22.0)}]}, {'confidence': 0.79037756, 'name': '可回收物_玩具玩偶', 'coor': [{'bbox': (22, 23, 26, 28), 'center': (24.0, 25.5)}]}, {'confidence': 0.9098015, 'name': '鞋', 'coor': [{'bbox': (27, 19, 51, 45), 'center': (39.0, 32.0)}]}]

        :param img_path:
        :param pred:
        :return:
        """
        # print(pred)
        im = pil.Image.open(img_path)
        ratio = float(self.size) / np.max(im.size)
        # w = int(ratio * im.size[0])
        # h = int(ratio * im.size[1])
        w_ratio = im.size[0] / 60
        h_ratio = im.size[1] / 60
        # im = im.resize((int(ratio * im.size[0]), int(ratio * im.size[1])))
        # im = im.resize((self.size, self.size))
        draw = ImageDraw.Draw(im)
        for p in pred:
            # print(p)
            name = p["name"]
            confidence = p["confidence"]
            for coor in p["coor"]:
                center = list(coor["center"])
                bbox = list(coor["bbox"])
                # bbox = [i / ratio for i in bbox]
                # center = [i / ratio for i in center]
                print("old", center, bbox)
                bbox[0] = bbox[0] * w_ratio
                bbox[2] = bbox[2] * h_ratio
                bbox[1] = bbox[1] * w_ratio
                bbox[3] = bbox[3] * h_ratio
                center[0] = center[0] * w_ratio
                center[1] = center[1] * h_ratio
                print("new", center, bbox)
                # print(name, confidence, bbox, centre)
                draw.rectangle(bbox, outline ='red', width=5)
                # draw.line(bbox, width=5, fill=128)  # 线的起点和终点，线宽
                draw.text(center, f"· {name}-{confidence}", font=self.fnt, fill=(212, 72, 19, 128))
                # draw.text((bbox[0] + 10, bbox[1] - 10), f"{name}-{confidence}", font=self.fnt, fill=(212, 72, 19, 128))
        im.save(img_path)

    def handle(self, *args, **options):
        num = 100
        print("start...")
        data = self.gen_img()

        for class_name, img_path_list in data.items():
            to_path = os.path.join(self.to_path, class_name)
            os.makedirs(to_path, exist_ok=True)
            for i in range(0, len(img_path_list), num):
                _img_path_list = img_path_list[i: i + num]
                preds = self.get_model_pred(_img_path_list)
                for index, pred in enumerate(preds):
                    print(f"\r {num + index}", end="")
                    #
                    img_path = _img_path_list[index]
                    img_name = img_path.rsplit("/", 1)[-1]
                    sh = f"cp -r '{img_path}' '{to_path}/'"
                    os.system(sh)

                    cur_img_path = os.path.join(to_path, img_name)
                    self.draw_img(cur_img_path, pred)
