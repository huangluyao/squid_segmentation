import json
import os
import xml.etree.ElementTree as ET

# 解析json
def parse_json(config_path):
    if os.path.isfile(config_path) and config_path.endswith('json'):
        data = json.load(open(config_path))
        data = data['data']
        return data


def parse_annotation(xml_path, category_id_and_name):
    in_file = open(xml_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    boxes = []
    category_ids = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text

        if category_id_and_name.get(cls) != None:
            cls_id = category_id_and_name[cls]
            xmlbox = obj.find('bndbox')
            boxes.append([int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text)])
            category_ids.append(cls_id)

    return boxes, category_ids
