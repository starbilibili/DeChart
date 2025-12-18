# 饼图数据增强，需对图片和对应的json标注文件处理
import os
import json
import numpy as np
import copy
from PIL import Image

# 随机缩放,缩放比例在0.8-1.2之间
def random_scale(img, json_file):
    """
    "polygon": {
        "x0": 238,
        "x1": 325,
        "x2": 325,
        "x3": 238,
        "y0": 24,
        "y1": 24,
        "y2": 59,
        "y3": 59
    },
    """
    new_file = copy.deepcopy(json_file)
    scale = np.random.uniform(low=0.8, high=1.2)
    img = img.resize((int(img.width * scale), int(img.height * scale)))
    text_blocks = new_file["task2"]["output"]["text_blocks"]
    for block in text_blocks:
        polygon = block["polygon"]
        for key in polygon:
            polygon[key] = int(polygon[key] * scale)
    json_file["task2"]["output"]["text_blocks"] = text_blocks
    json_file["task3"]["input"]["task2_output"]["text_blocks"] = text_blocks

    return img, json_file

# 颜色变换
def random_color_change(img):
    shift = 0.8
    img = np.array(img)
    bgr = img[:, :, :3].astype(np.int16)  # 转为int16避免溢出
    bgr = np.clip(bgr*shift, 0, 255).astype(np.uint8)  # 限制范围并转回uint8
    img[:, :, :3] = bgr
    img = Image.fromarray(img)
    return img
    

if __name__ == '__main__':
    image_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/images/ICPR2022/pie"
    json_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/json/ICPR2022/pie"
    out_image_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/images/ICPR2022/pie_aug"
    out_json_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/json/ICPR2022/pie_aug"
    file_names = []
    for item in os.listdir(image_dir):
        file_name = item[:-4]
        file_names.append(file_name)
    
    for file_name in file_names:
        print("processing {}".format(file_name))
        img = Image.open(os.path.join(image_dir, file_name + '.jpg')).convert('RGB')
        with open(os.path.join(json_dir, file_name + '.json'), 'r', encoding='utf-8') as f:
            json_file = json.load(f)
        
        img_scale, json_scale = random_scale(img, json_file)
        img_color_change = random_color_change(img)
        # 保存修改后的图片和json文件
        img_scale.save(os.path.join(out_image_dir, file_name + '_scale.jpg'))
        with open(os.path.join(out_json_dir, file_name + '_scale.json'), 'w', encoding='utf-8') as f:
            json.dump(json_scale, f, ensure_ascii=False, indent=4)

        img_color_change.save(os.path.join(out_image_dir, file_name + '_color.jpg'))
        with open(os.path.join(out_json_dir, file_name + '_color.json'), 'w', encoding='utf-8') as f:
            json.dump(json_file, f, ensure_ascii=False, indent=4)
