# 使用paddleocr完成文本检测和识别
import os
import numpy as np
import torch
import json
import cv2
import sys
import re
sys.path.append('/data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/tools')
from paddleocr import PaddleOCR, draw_ocr
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
from adaptive_scaling import get_scale_based_on_height
from scipy.optimize import linear_sum_assignment

def cpmpute_iou(boxA, boxB):
    # 计算两个矩形框的交集面积
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # 计算两个矩形框的总面积
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(areaA + areaB - interArea)
    return iou

def compute_edit_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    distance = np.zeros((m+1, n+1))
    for i in range(m+1):
        distance[i][0] = i
    for j in range(n+1):
        distance[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if str1[i-1] == str2[j-1] else 1
            distance[i][j] = min(distance[i-1][j] + 1, distance[i][j-1] + 1, distance[i-1][j-1] + cost)
    return distance[-1][-1]/m

def evaluate(gt_data, pred_data, iou_threshold=0.5):
    """
    :param gt_data: 列表，每个元素是(bbox, text)的元组
    :param pred_data: 列表，每个元素是(bbox, text)的元组
    :param iou_threshold: IoU阈值
    :return: (detection_score, recognition_score, final_score)
    """
    num_gt = len(gt_data)
    num_pred = len(pred_data)

    if num_gt == 0 and num_pred == 0:
        return 1.0, 1.0, 1.0
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, 0.0
    
    # 构建IOU矩阵
    iou_matrix = np.zeros((num_gt, num_pred))
    for i, (gt_box, _) in enumerate(gt_data):
        for j, (pred_box, _) in enumerate(pred_data):
            iou_matrix[i][j] = cpmpute_iou(gt_box, pred_box)
    
    # 使用匈牙利算法进行匹配
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)

    ###### 匹配文本 #####
    detection_scores = []
    recognition_scores = []

    matched_gt = set()
    matched_pred = set()

    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        if iou_matrix[gt_idx, pred_idx] >= iou_threshold:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            
            gt_box, gt_text = gt_data[gt_idx]
            pred_box, pred_text = pred_data[pred_idx]
            detection_scores.append(iou_matrix[gt_idx, pred_idx])
            rec_score = max(1 - compute_edit_distance(gt_text, pred_text), 0)
            recognition_scores.append(rec_score)
    
    # 计算检测分数
    detection_score = sum(detection_scores)/max(num_gt, num_pred)

    # 计算识别分数
    if len(recognition_scores) > 0:
        recognition_score = sum(recognition_scores)/num_gt
    else:
        recognition_score = 0

    # 计算最终得分
    if detection_score + recognition_score > 0:
        final_score = 2 * detection_score * recognition_score / (detection_score + recognition_score)
    else:
        final_score = 0.0
        
    return detection_score, recognition_score, final_score


def get_all_images(images_root):
    images = []
    # 收集所有PNG图片的相对路径
    image_paths = []
    for root, dirs, files in os.walk(images_root):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith('.png') or file.lower().endswith('.jpg') and not file.startswith('.'):
                # 获取相对于根目录的路径
                rel_path = os.path.relpath(os.path.join(root, file), images_root)
                image_paths.append(rel_path)
    
    # 分层抽样（保持目录结构分布）
    path_groups = defaultdict(list)
    for path in image_paths:
        dir_part = os.path.dirname(path)
        path_groups[dir_part].append(path)
    
    # 对每个目录单独抽样
    for dir_path, files in path_groups.items():
        images.extend(files)
    
    return images

# 自动缩紧边界框
def tighten_boxes(image, boxes, margin=2):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    tightened_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, parse_box(box))
        w, h = gray.shape

        # 裁剪出边界框区域
        cropped_region = gray[y1:y2, x1:x2]
        if cropped_region.size == 0:
            tightened_boxes.append([[x1,y1], [x2, y1], [x2, y2], [x1, y2]])
            continue

        # 二值化处理
        _, binary = cv2.threshold(cropped_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 判断字体颜色
        white_pixels = (binary==255).sum()
        black_pixels = (binary==0).sum()
        if white_pixels > black_pixels:
            binary = 255 - binary

        # 查找文本实际边界
        ys, xs = np.where(binary > 0)
        if len(ys) == 0 or len(xs) == 0:
            tightened_boxes.append([[x1,y1], [x2, y1], [x2, y2], [x1, y2]])
            continue

        # 计算紧缩后的坐标
        min_x = xs.min() - margin
        min_y = ys.min() - margin
        max_x = xs.max() + margin
        max_y = ys.max() + margin

        new_x1 = x1 + min_x
        new_y1 = y1 + min_y
        new_x2 = x1 + max_x
        new_y2 = y1 + max_y
        tightened_boxes.append([(new_x1, new_y1),(new_x2, new_y1),(new_x2, new_y2),(new_x1, new_y2)])  
    return tightened_boxes

# 检测文本内容，如果文本内容全部是无用字符，则删除该文本框
def remove_invalid_boxes(boxes, txts):
    # 预编译正则规则
    noise_patterns = [
        re.compile(r'^[\W_]+$'),  # 纯符号
        re.compile(r'^[x\-]{3,}$'),  # 连续无意义字符
    ]
    valid_boxes = []
    valid_txts = []
    for box, txt in zip(boxes, txts):
        txt = txt.strip()
        # 空文本直接过滤
        if not txt:
            continue
        # 噪声模式匹配
        if any(pattern.match(txt) for pattern in noise_patterns):
            continue
        valid_boxes.append(box)
        valid_txts.append(txt)
    return valid_boxes, valid_txts
    
def parse_box(box: List[List[float]]) -> tuple:
    # 将{'x0': 197, 'x1': 217, 'x2': 217, 'x3': 197, 'y0': 263, 'y1': 263, 'y2': 277, 'y3': 277}形式转化成(x1, y1, x2, y2)格式
    if len(box) != 4:
        p0 = (box['x0'], box['y0'])
        p1 = (box['x1'], box['y1'])
        p2 = (box['x2'], box['y2'])
        p3 = (box['x3'], box['y3'])
        box = [p0, p1, p2, p3]
    """将四边形框转换为(x1, y1, x2, y2)格式"""
    x_coords = [p[0] for p in box]
    y_coords = [p[1] for p in box]
    return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

def is_number_sequence(text: str) -> bool:
    """判断是否为数字序列(如0.0, 0.1等)"""
    try:
        float(text)
        return True
    except ValueError:
        return False

# # 合并文本块
# def merge_text_blocks(boxes, txts):
#     # 垂直合并
#     processed = [{
#         'text': text,
#         'bbox': parse_box(bbox),
#         'is_number': is_number_sequence(text),
#     } for bbox, text in zip(boxes, txts)]

#     # 按先垂直(top->bottom)后水平(left->right)排序
#     sorted_items = sorted(processed, key=lambda x: (x['bbox'][0], x['bbox'][1]))

#     blocks = []
#     current_block = []
    
#     for item in sorted_items:
#         if not current_block:
#             current_block.append(item)
#             continue
            
#         last = current_block[-1]
        
#         # 垂直间距计算
#         vertical_gap = abs(item['bbox'][3] - last['bbox'][1])
        
#         # 水平重叠计算
#         x_overlap = max(0, min(last['bbox'][2], item['bbox'][2]) - max(last['bbox'][0], item['bbox'][0]))
#         horizontal_overlap_ratio = x_overlap / min(last['bbox'][2]-last['bbox'][0], item['bbox'][2]-item['bbox'][0])
        
#         # 常规文本合并条件
#         if vertical_gap < (item['bbox'][3] - item['bbox'][1])/3 and horizontal_overlap_ratio > 0.6:
#             current_block.append(item)
#         else:
#             blocks.append(current_block)
#             current_block = [item]
    
#     if current_block:
#         blocks.append(current_block)
    
#     # 生成合并后的结果
#     boxes_merged = []
#     texts_merged = []
#     for block in blocks:
#         texts = [item['text'] for item in block]
#         x1 = min(item['bbox'][0] for item in block)
#         y1 = min(item['bbox'][1] for item in block)
#         x2 = max(item['bbox'][2] for item in block)
#         y2 = max(item['bbox'][3] for item in block)
#         boxes_merged.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
#         texts_merged.append(' '.join(texts))
    
#     return boxes_merged, texts_merged
def merge_text_blocks(boxes, txts, 
                     horizontal_overlap_threshold=0.6, 
                     vertical_gap_threshold=0.3,
                     gap_variation_threshold=0.2):
    """
    合并OCR检测到的多行文本框为逻辑文本块
    
    参数:
        boxes: 文本框坐标列表 [[[x1,y1], [x2,y1], [x2,y2], [x1,y2]], ...]
        txts: 对应文本内容列表
        horizontal_overlap_threshold: 水平重叠比例阈值 (默认0.6)
        vertical_gap_threshold: 垂直间距阈值 (默认0.5)
        gap_variation_threshold: 垂直间距变化阈值 (默认0.2)
    返回:
        boxes_merged: 合并后的文本框坐标
        texts_merged: 合并后的文本内容
    """
    # 解析文本框为统一格式 (x_min, y_min, x_max, y_max)
    parsed_boxes = []
    for box in boxes:
        points = np.array(box)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        parsed_boxes.append((x_min, y_min, x_max, y_max, box))
    
    # 水平分组：将水平重叠的文本框分为一组
    groups = defaultdict(list)
    group_index = 0
    
    # 按垂直位置排序 (从上到下)
    # sorted_boxes = sorted(parsed_boxes, key=lambda b: b[1])
    sorted_boxes = sorted(parsed_boxes, key=lambda x: (x[1], x[0]))
    
    for i, (x1_i, y1_i, x2_i, y2_i, orig_box_i) in enumerate(sorted_boxes):
        matched_group = None
        
        # 检查是否与现有组匹配
        for gid, group_boxes in groups.items():
            # 检查与组内所有框的水平重叠
            for x1_j, y1_j, x2_j, y2_j, _, _ in group_boxes:
                # 计算水平重叠
                overlap_x = max(0, min(x2_i, x2_j) - max(x1_i, x1_j))
                min_width = min(x2_i - x1_i, x2_j - x1_j)
                
                # 分组条件：水平重叠大于阈值，并且垂直间距小于文本框高度,并且文本框高度相近
                if overlap_x / min_width > horizontal_overlap_threshold and abs(y1_i - y2_j) <= (y2_i - y1_i)*vertical_gap_threshold and 0.75 < abs(y2_j - y1_j)/abs(y2_i - y1_i) < 1.25:
                    matched_group = gid
                    break
            if matched_group is not None:
                break
        
        # 如果没有匹配的组，创建新组
        if matched_group is None:
            group_index += 1
            groups[group_index].append((x1_i, y1_i, x2_i, y2_i, orig_box_i, txts[i]))
        else:
            groups[matched_group].append((x1_i, y1_i, x2_i, y2_i, orig_box_i, txts[i]))

    # print(groups.items())
    # 垂直合并：对每组内的文本框进行垂直合并
    merged_boxes = []
    merged_texts = []
    
    for group_id, boxes_in_group in groups.items():
        # 按垂直位置排序（从上到下）
        sorted_group = sorted(boxes_in_group, key=lambda b: b[1])
        
        # 计算所有相邻文本框的垂直间距
        gaps = []
        for j in range(1, len(sorted_group)):
            _, y1_prev, _, y2_prev, _, _ = sorted_group[j-1]
            _, y1_curr, _, _, _, _ = sorted_group[j]
            gaps.append(y1_curr - y2_prev)
        
        # 如果只有一个文本框，直接合并
        if not gaps:
            merge_block([sorted_group[0]], merged_boxes, merged_texts)
            continue
            
        # 计算统计特征
        avg_gap = sum(gaps) / len(gaps)
        std_gap = np.std(gaps)
        # 动态阈值：使用3σ原则检测异常间距（可根据需求调整系数）
        gap_threshold = avg_gap * (1 + gap_variation_threshold)
        
        # 找出断点位置（显著大于平均值的间距）
        break_points = []
        for i, gap in enumerate(gaps):
            if gap > gap_threshold:
                break_points.append(i)
        
        # 添加起始和结束标记
        break_points = [-1] + break_points + [len(gaps)-1]
        
        # 根据断点分割文本框并合并
        for k in range(1, len(break_points)):
            start_idx = break_points[k-1] + 1
            end_idx = break_points[k] + 1  # end_idx是当前断点位置+1
            
            # 提取子组
            sub_group = sorted_group[start_idx:end_idx+1]
            
            # 如果子组长度为0，跳过
            if not sub_group:
                continue
                
            # 合并子组
            merge_block(sub_group, merged_boxes, merged_texts)
        # # 初始化合并块
        # current_block = [sorted_group[0]]
        
        # for j in range(1, len(sorted_group)):
        #     _, y1_curr, _, y2_curr, _, _ = current_block[-1]
        #     x1_next, y1_next, x2_next, y2_next, _, _ = sorted_group[j]
            
        #     # 计算垂直间距
        #     vertical_gap = y1_next - y2_curr
            
        #     # 计算垂直重叠 (如果有)
        #     vertical_overlap = max(0, min(y2_curr, y2_next) - max(y1_curr, y1_next))
        #     min_height = min(y2_curr - y1_curr, y2_next - y1_next)
            
        #     # 检查合并条件,
        #     if (vertical_gap > 0 and 
        #         vertical_gap < vertical_gap_threshold * min(y2_curr - y1_curr, y2_next - y1_next)) or \
        #        (vertical_overlap > 0 and 
        #         vertical_overlap / min_height > vertical_overlap_threshold):
        #         current_block.append(sorted_group[j])
        #     else:
        #         # 完成当前块的合并
        #         merge_block(current_block, merged_boxes, merged_texts)
        #         current_block = [sorted_group[j]]
        
        # # 处理最后一个块
        # merge_block(current_block, merged_boxes, merged_texts)
    
    return merged_boxes, merged_texts

def merge_block(block, merged_boxes, merged_texts):
    """合并一个块内的文本框"""
    # 提取原始框和文本
    orig_boxes = [item[4] for item in block]
    texts = [item[5] for item in block]
    
    # 计算合并后的边界框
    all_points = np.vstack(orig_boxes)
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # 创建合并后的四边形
    merged_box = [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ]
    
    # 合并文本 (按垂直顺序)
    merged_text = ' '.join(texts)
    
    merged_boxes.append(merged_box)
    merged_texts.append(merged_text)


def run_ocr(image):
    image = np.array(image)
    result = ocr.ocr(image)
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    return boxes, txts

def save_result(save_path, pred_data):
    # 如果文件夹不存在，则创建
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as f:
        print(f"Saving results to {save_path}")
        for data in pred_data:
            f.write(str(data[0]) + "," + data[1] + "\n")
            # f.write(data + '\n')

def run_ICDAR2019(images_root, label_root, output_root):
    images = get_all_images(images_root)
    print("total images:", len(images))
    IOU_scores, OCR_scores, F1_scores = [], [], []
    for i, path in tqdm(enumerate(images[::])):
        image_path = os.path.join(images_root, path)
        json_path = os.path.join(label_root, path.replace('.png', '.json').replace('.jpg', '.json'))
        output_path = os.path.join(output_root, path.replace(".png", "_ocr.txt").replace(".jpg", "_ocr.txt"))
        image = Image.open(image_path).convert('RGB')
        with open(json_path, 'r') as f:
            label = json.load(f)
            label = label['task2']['output']["text_blocks"]
            gt_data = [(parse_box(item['polygon']), item['text'].replace('\n', ' ')) for item in label]
            # print(gt_data)
        # image转成numpy
        try:
            # 如果图像高度或宽度小于256，则进行缩放，保证高度和宽度均大于256
            scale = 1
            if image.size[0] <= 256 or image.size[1] <= 256:
                scale = max(256/image.size[0], 256/image.size[1])
                image = image.resize((int(scale*image.size[0]), int(scale*image.size[1])))
            boxes, txts = run_ocr(image)
            # 将检测结果缩放回原尺寸
            for idx, box in enumerate(boxes):
                boxes[idx] = [(int(box[0][0] * scale), int(box[0][1] * scale)),
                            (int(box[1][0] * scale), int(box[1][1] * scale)),
                            (int(box[2][0] * scale), int(box[2][1] * scale)),
                            (int(box[3][0] * scale), int(box[3][1] * scale))]
            boxes = tighten_boxes(image, boxes)
            print(i)
            # S1 = get_scale_based_on_height(boxes) # 根据高度进行自适应缩放
            # image_scaled = image.resize((int(S1*image.size[0]), int(S1*image.size[1])))
            # boxes_scaled, txts_scaled = run_ocr(image_scaled)
            boxes, txts = merge_text_blocks(boxes, txts)

            pred_data = [(parse_box(box), text) for box, text in zip(boxes, txts)]
            # print(pred_data)
            IOU_score, OCR_score, F1_score = evaluate(gt_data, pred_data)
            print(IOU_score, OCR_score, F1_score)
            IOU_scores.append(IOU_score)
            OCR_scores.append(OCR_score)
            F1_scores.append(F1_score)
            save_result(output_path, pred_data)

            im_show = draw_ocr(image, boxes, txts, font_path='/data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/tools/simfang.ttf')
            im_show = Image.fromarray(im_show)
            im_show.save('/data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/analyse/ocr_results'+str(i) + '.jpg')
        except:
            print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(image_path)
    print(f"IOU score: {np.mean(IOU_scores)}, OCR score: {np.mean(OCR_scores)}, F1 score: {np.mean(F1_scores)}")

def run_ChartQA(images_root, output_root):
    images = os.listdir(images_root)
    print("total images:", len(images))
    for i, path in tqdm(enumerate(images[:1:])):
        # image_path = os.path.join(images_root, path)
        image_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/png/894.png"
        output_path = os.path.join(output_root, path.replace(".png", ".txt").replace(".jpg", ".txt"))
        image = Image.open(image_path).convert('RGB')
        try:
            scale = 1
            if image.size[0] <= 512 or image.size[1] <= 512:
                scale = max(512/image.size[0], 512/image.size[1])
                image = image.resize((int(scale*image.size[0]), int(scale*image.size[1])))
            boxes, txts = run_ocr(image)
            # 将检测结果缩放回原尺寸
            for idx, box in enumerate(boxes):
                boxes[idx] = [(int(box[0][0] * scale), int(box[0][1] * scale)),
                            (int(box[1][0] * scale), int(box[1][1] * scale)),
                            (int(box[2][0] * scale), int(box[2][1] * scale)),
                            (int(box[3][0] * scale), int(box[3][1] * scale))]
            # 文本框缩紧
            boxes = tighten_boxes(image, boxes)
            # 文本块合并
            boxes, txts = merge_text_blocks(boxes, txts)
            pred_data = [(parse_box(box), text) for box, text in zip(boxes, txts)]
            save_result(output_path, pred_data)
            im_show = draw_ocr(image, boxes, txts, font_path='/data5/home/xiechenyu2023/project/ChartQA/Qwen2/workspace/tools/simfang.ttf')
            im_show = Image.fromarray(im_show)
            im_show.save(os.path.join('/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/train_1K/ocr_images', path))
        except:
            print(image_path)

if __name__ == '__main__':
    ocr = PaddleOCR(ocr_version='PP-OCRv4' ,use_angle_cls=True, lang="en", det_db_box_thresh=0.2, use_dilation=True, enable_mkldnn=True)
    # images_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/images"
    images_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test/png"
    label_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/json"
    # output_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/ocr_results"
    output_root = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/train_1K/ocr_results"
    run_ChartQA(images_root, output_root)
    # run_ICDAR2019(images_root, label_root, output_root)
    # images = get_all_images(images_root)

    # with open("/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/list.txt", 'w') as f:
    #     for image in images:
    #         f.write(image + "\n")