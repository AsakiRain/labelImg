#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

from libs.constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

class YOLOWriter:

    def __init__(self, folder_name, filename, img_size, database_src='Unknown', local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_bnd_box(self, x_min, y_min, x_max, y_max, name, difficult):
        bnd_box = {'xmin': x_min, 'ymin': y_min, 'xmax': x_max, 'ymax': y_max}
        bnd_box['name'] = name
        bnd_box['difficult'] = difficult
        self.box_list.append(bnd_box)

    def bnd_box_to_yolo_line(self, box, class_list=None):
        if class_list is None:
            class_list = []

        x_min = box['xmin']
        x_max = box['xmax']
        y_min = box['ymin']
        y_max = box['ymax']

        x_center = float((x_min + x_max)) / 2 / self.img_size[1]
        y_center = float((y_min + y_max)) / 2 / self.img_size[0]

        w = float((x_max - x_min)) / self.img_size[1]
        h = float((y_max - y_min)) / self.img_size[0]

        # PR387
        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)

        class_index = class_list.index(box_name)

        return class_index, x_center, y_center, w, h

    def save(self, class_list=None, target_file=None, class_list_path=None):

        if class_list is None:
            class_list = []
        else:
            # Do not mutate MainWindow.label_hist or any caller-owned list.
            class_list = list(class_list)

        out_file = None  # Update yolo .txt
        out_class_file = None   # Update class list .txt

        if target_file is None:
            target_file = self.filename + TXT_EXT

        target_dir = os.path.dirname(os.path.abspath(target_file))
        os.makedirs(target_dir, exist_ok=True)
        if class_list_path:
            classes_file = os.path.abspath(class_list_path)
            os.makedirs(os.path.dirname(classes_file), exist_ok=True)
        else:
            classes_file = os.path.join(target_dir, "classes.txt")

        # Preserve the class order already stored in the configured class file.
        # Its line numbers define the class indices written into YOLO rows.
        if os.path.isfile(classes_file):
            with codecs.open(classes_file, 'r', encoding=ENCODE_METHOD) as f:
                existing_class_list = [line.strip() for line in f if line.strip()]
            if existing_class_list:
                class_list = existing_class_list

        out_file = codecs.open(target_file, 'w', encoding=ENCODE_METHOD)
        out_class_file = codecs.open(classes_file, 'w', encoding=ENCODE_METHOD)

        for box in self.box_list:
            class_index, x_center, y_center, w, h = self.bnd_box_to_yolo_line(box, class_list)
            # print (classIndex, x_center, y_center, w, h)
            out_file.write("%d %.6f %.6f %.6f %.6f\n" % (class_index, x_center, y_center, w, h))

        # print (classList)
        # print (out_class_file)
        for c in class_list:
            out_class_file.write(c+'\n')

        out_class_file.close()
        out_file.close()



class YoloReader:

    def __init__(self, file_path, image, class_list_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)], color, color, difficult]
        self.shapes = []
        self.file_path = file_path

        dir_path = os.path.dirname(os.path.realpath(self.file_path))
        annotation_classes_path = os.path.join(dir_path, "classes.txt")
        # An explicitly configured class file is authoritative. Only use a
        # classes.txt alongside the annotation when no class file was supplied.
        if class_list_path is not None:
            self.class_list_path = class_list_path
        else:
            self.class_list_path = annotation_classes_path

        # print (file_path, self.class_list_path)

        classes_file = open(self.class_list_path, 'r')
        self.classes = classes_file.read().strip('\n').split('\n')

        # print (self.classes)

        img_size = [image.height(), image.width(),
                    1 if image.isGrayscale() else 3]

        self.img_size = img_size

        self.verified = False
        # try:
        self.parse_yolo_format()
        # except:
        #     pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, x_min, y_min, x_max, y_max, difficult):

        points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        self.shapes.append((label, points, None, None, difficult))

    def yolo_line_to_shape(self, class_index, x_center, y_center, w, h):
        label = self.classes[int(class_index)]

        x_min = max(float(x_center) - float(w) / 2, 0)
        x_max = min(float(x_center) + float(w) / 2, 1)
        y_min = max(float(y_center) - float(h) / 2, 0)
        y_max = min(float(y_center) + float(h) / 2, 1)

        x_min = round(self.img_size[1] * x_min)
        x_max = round(self.img_size[1] * x_max)
        y_min = round(self.img_size[0] * y_min)
        y_max = round(self.img_size[0] * y_max)

        return label, x_min, y_min, x_max, y_max

    def parse_yolo_format(self):
        bnd_box_file = open(self.file_path, 'r')
        for bndBox in bnd_box_file:
            class_index, x_center, y_center, w, h = bndBox.strip().split(' ')
            label, x_min, y_min, x_max, y_max = self.yolo_line_to_shape(class_index, x_center, y_center, w, h)

            # Caveat: difficult flag is discarded when saved as yolo format.
            self.add_shape(label, x_min, y_min, x_max, y_max, False)
