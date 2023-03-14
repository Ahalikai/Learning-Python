
# link: https://blog.csdn.net/ing100/article/details/125155065

import xml.etree.ElementTree as ET
import os
import random

VOC_CLASSES = ( # Yours classes
    'cscn', 'lkbn', 'tkbe'
)

# parameter
train_set = open('train.txt', 'w')
test_set = open('test.txt', 'w')
train_ratio = 0.7

def parse_rec(filename):
    tree = ET.parse(filename)
    objects = []

    for obj in tree.findall('object'):
        obj_struct = {}

        difficult = int(obj.find('difficult').text)
        if difficult == 1:
            continue

        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [
            int(float(bbox.find('xmin').text)),
            int(float(bbox.find('ymin').text)),
            int(float(bbox.find('xmax').text)),
            int(float(bbox.find('ymax').text))
        ]

        objects.append(obj_struct)

    return objects


def write_sets(lists, set):
    count = 0
    for train_list in lists:
        count += 1
        image_name = train_list.split('.')[0] + 'jpg'
        results = parse_rec(Annotations + train_list)
        if len(results) == 0:
            print(train_list)
            continue

        set.write(image_name)

        for result in results:
            class_name = result['name']
            bbox = result['bbox']
            class_name = VOC_CLASSES.index(class_name)

            set.write(
                ' ' + str(bbox[0]) +
                ' ' + str(bbox[1]) +
                ' ' + str(bbox[2]) +
                ' ' + str(bbox[3]) +
                ' ' + str(class_name)
            )

        set.write('\n')
    set.close()

def write_txt(Annotations):
    xml_files = os.listdir(Annotations)
    random.shuffle(xml_files)  # random xml files
    train_num = int(len(xml_files) * train_ratio)
    train_lists = xml_files[:train_num]
    test_lists = xml_files[train_num:]

    write_sets(train_lists, train_set)
    write_sets(test_lists, test_set)


if __name__ == "__main__":
    Annotations = 'D:\pycharm\yolov5-7.0\cscn/2021/200819cscn-3/Annotations/'
    write_txt(Annotations)
