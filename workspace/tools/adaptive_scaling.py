import re
import os
import cv2
import numpy as np
from tqdm import tqdm

# 从prompt中提取出文本框坐标
def get_boxes(prompt):
    content = prompt[0]['content']
    boxes = re.findall('are as follows:(.*?)Please analyze the chart image', content)
    boxes = boxes[0].split('<SEP>')
    boxes = [item.split(":")[0].strip('(').strip(')') for item in boxes]
    new_boxes = []
    for box in boxes:
        box = box.split(',')
        box = [int(item) for item in box]
        new_boxes.append(box)
    return new_boxes

# 从ocr结果中提取出文本框坐标
def parse_ocr_result(ocr_file):
    boxes = []
    texts = []
    with open(ocr_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            box = line.strip().split("),")[0].strip('(').split(",")
            text = line.strip().split("),")[1:]
            box = [int(float(item)) for item in box]
            text = "),".join(text)
            boxes.append(box)
            texts.append(text)
    return boxes, texts

# 基于文本框的高度计算缩放因子,默认阈值为20
def get_scale_based_on_height(boxes, threshold=25):
    # 文本框坐标形式：[(x1,y1,x2,y2)] 
    # 统计所有文本框的平均高度
    total_h = 0
    valid_count = 0
    for box in boxes:
        if len(box) != 4:
            continue  # 跳过无效格式
            
        x1, y1, x2, y2 = box
        h = y2 - y1
        w = x2 - x1
        
        # 如果高度与宽度之比超过3，则不纳入统计
        if w == 0 or h / w > 3:
            continue
            
        total_h += h
        valid_count += 1
        
    if valid_count == 0:
        return 1.0  # 默认缩放因子
        
    avg_h = total_h / valid_count
    S1 = threshold / avg_h
    # print(f"Average height: {avg_h}, Scale factor: {S1}")
    return S1

def get_adaptive_scale(prompt):
    try:
        boxes = get_boxes(prompt)
        scale = get_scale_based_on_height(boxes)
    except:
        print(prompt)
        scale = 1.0
    return scale

# 
if __name__ == "__main__":
    # pass
    images_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/train/png"
    ocr_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/train/ocr_results"
    images_output = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test_scale_t30/png"
    ocr_output = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test_scale_t30/ocr_results"
    # 创建输出目录
    for dir_path in [images_output, ocr_output]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    # boxes = get_boxes(prompt)
    count_less_than_one, count_between_one_and_two, count_between_two_and_three, count_greater_than_three = 0, 0, 0, 0
    total_count = 0
    avg_w, avg_h = 0, 0
    for file in tqdm(os.listdir(ocr_root)[:]):
        boxes, texts = parse_ocr_result(os.path.join(ocr_root, file))
        scale = get_scale_based_on_height(boxes)
        # 对文本框坐标进行缩放
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            boxes[i][0] = int(x1 * scale)
            boxes[i][1] = int(y1 * scale)
            boxes[i][2] = int(x2 * scale)
            boxes[i][3] = int(y2 * scale)
        # 将缩放后的文本框坐标以及文本保存到文件中
        # with open(os.path.join(ocr_output, file), 'w') as f:
        #     for box, text in zip(boxes, texts):
        #         f.write(str(tuple(box)))
        #         f.write(",")
        #         f.write(text)
        #         f.write("\n")
        # 读取图像并进行缩放
        img = cv2.imread(os.path.join(images_root, file[:-4] + ".png"))
        # img_scale = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # print(f"Scale factor: {scale:.2f}, Image shape: {img_scale.shape}, Original shape: {img.shape}")
        # 统计放缩之后图像的平均尺寸
        avg_w += img.shape[1]
        avg_h += img.shape[0]
        # 统计scale的分布，<1, 1<=scale<=1.2, 1.2<scale<1.5, scale>=1.5
        total_count += 1
        if scale < 1:
            count_less_than_one += 1
        elif 1 <= scale <= 1.5:
            count_between_one_and_two += 1
        elif 1.5 < scale < 2:
            count_between_two_and_three += 1
        else:
            count_greater_than_three += 1
    print(f"Scale factor: {total_count}, Less than 1: {count_less_than_one}, Between 1 and 1.5: {count_between_one_and_two}, Between 1.5 and 2: {count_between_two_and_three}, Greater than 2: {count_greater_than_three}, avg_w:{avg_w//total_count}, avg_h:{avg_h//total_count}")
        # cv2.imwrite(os.path.join(images_output, file[:-4] + ".png"), img_scale)
