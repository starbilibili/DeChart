import os, sys
import json
import base64
from socket import IOCTL_VM_SOCKETS_GET_LOCAL_CID
import requests
from tqdm import tqdm

import time
import timeit
import datetime
from PIL import Image
import numpy as np
from io import BytesIO
import random
import multiprocessing as mp
import math
import cv2
import copy
import heapq
from collections import defaultdict
from run_paddleocr import evaluate, tighten_boxes, merge_text_blocks, parse_box

# 创建多级目录
def MakePath(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return


# 矩形区域转多边形区域
def BboxToPolygon(bbox):
    l, r, t, b = bbox

    poly = np.zeros((4,2), np.int32)
    poly[0, :] = [l, t]
    poly[1, :] = [r, t]
    poly[2, :] = [r, b]
    poly[3, :] = [l, b]
    
    return poly


# 多边形外接框
def PolygonToBBox(poly):
    x, y, w, h = cv2.boundingRect(poly)
    l, r, t, b = x, x+w, y, y+h

    return [l, r, t, b]


# 读取多页TIF图像
def tifread(path):
    if path.endswith("tif"):
        img = Image.open(path)
        imgs = []
        for i in range(img.n_frames):
            img.seek(i)
            imgs.append(np.array(img))
            if imgs[-1].dtype == bool:
                imgs[-1] = imgs[-1].astype(np.uint8) * 255
            else:
                imgs[-1] = imgs[-1].astype(np.uint8)
    else:
        img = Image.open(path)
        imgs = [np.array(img)]

    return imgs


# numpy 转 base64
def numpy_to_base64(image_np): 
    data = cv2.imencode('.jpg', image_np)[1]
    image_bytes = data.tobytes()
    image_base4 = base64.b64encode(image_bytes).decode('utf8')

    return image_base4


# base64转数组
def base64_to_numpy(image_base64):    
    image_bytes = base64.b64decode(image_base64)
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    return image_np2


# 字符占据的面积
def char_area_ratio(img, anno):
    hei, wid = img.shape[0], img.shape[1]
    mask = np.zeros((hei, wid), np.uint8)

    for line in anno["lines"]:
        for char in line["chars"]:
            polygon = np.array(char["coords"], np.int32).reshape((-1, 2))
            mask = cv2.fillPoly(mask, [polygon], color=1)

    return mask.sum() / (hei * wid)


# 标注旋转
def anno_rotate(img, anno):
    hei, wid = img.shape[0], img.shape[1]

    for i in range(len(anno["lines"])):
        coords = [0,0,0,0,0,0,0,0]
        coords[0] = anno["lines"][i]["coords"][1]
        coords[1] = hei - anno["lines"][i]["coords"][0]
        coords[2] = anno["lines"][i]["coords"][3]
        coords[3] = hei - anno["lines"][i]["coords"][2]
        coords[4] = anno["lines"][i]["coords"][5]
        coords[5] = hei - anno["lines"][i]["coords"][4]
        coords[6] = anno["lines"][i]["coords"][7]
        coords[7] = hei - anno["lines"][i]["coords"][6]
        anno["lines"][i]["coords"] = coords

        for j in range(len(anno["lines"][i]["chars"])):
            coords = [0,0,0,0,0,0,0,0]
            coords[0] = anno["lines"][i]["chars"][j]["coords"][1]
            coords[1] = hei - anno["lines"][i]["chars"][j]["coords"][0]
            coords[2] = anno["lines"][i]["chars"][j]["coords"][3]
            coords[3] = hei - anno["lines"][i]["chars"][j]["coords"][2]
            coords[4] = anno["lines"][i]["chars"][j]["coords"][5]
            coords[5] = hei - anno["lines"][i]["chars"][j]["coords"][4]
            coords[6] = anno["lines"][i]["chars"][j]["coords"][7]
            coords[7] = hei - anno["lines"][i]["chars"][j]["coords"][6]
            anno["lines"][i]["chars"][j]["coords"] = coords

    return anno


# OCR结果可视化
def visulization(img, anno):
    for line in anno["lines"]:
        polygon = np.array(line["coords"], np.int32).reshape((-1, 2))
        img = cv2.polylines(img, [polygon], isClosed=True, color=[255, 0, 0], thickness=2)
        for char in line["chars"]:
            polygon = np.array(char["coords"], np.int32).reshape((-1, 2))
            img = cv2.polylines(img, [polygon], isClosed=True, color=[0, 0, 255], thickness=1)

    return img


# 转换成字符串，节省存储空间
def json_to_str(anno):
    res = ""
    res += str(anno["hei"]) + ", " + str(anno["wid"]) + "\n"
    for line in anno["lines"]:
        res += str(len(line["chars"])) + ", "   # 字符个数
        # res += str(line["coords"][0]) + " " + str(line["coords"][1]) + " " + str(line["coords"][2]) + " " + str(line["coords"][3]) + \
        #    " " + str(line["coords"][4]) + " " + str(line["coords"][5]) + " " + str(line["coords"][6]) + " " + str(line["coords"][7]) + ", "
        for c in range(len(line["coords"]) - 1):
            res += str(line["coords"][c]) + " "
        res += str(line["coords"][-1]) + ", "
        
        for char in line["chars"]:
            res += str(char["coords"][0]) + " " + str(char["coords"][1]) + " " + str(char["coords"][4]) + " " + str(char["coords"][5]) + ", "
        res += line["text"] + "\n"

    return res


# 查找未处理的文件
def FindUnProcessedData(thres_id, data_path, save_path, imNames, expanded_name=".txt"):
    res = []
    start_time = time.time()  # 记录开始时间
    last_time = time.time()  # 记录上一次输出时间
    for index, imName in enumerate(imNames):
        if index % 1000 == 0:
            # 每处理1000个文件，输出一次进度信息
            time_cost = str(datetime.timedelta(seconds=int(time.time() - start_time)))  # 计算已用时间
            eta_seconds = ((time.time() - last_time) / 1000) * (len(imNames) - index)  # 计算剩余时间
            time_need = str(datetime.timedelta(seconds=int(eta_seconds)))  # 格式化剩余时间
            print(str(thres_id) + ": " + str(index) + "/" + str(len(imNames)) + ": " + time_cost + "<--" + time_need + ": " + imName)  # 输出进度信息
            last_time = time.time()  # 更新上一次输出时间

        if os.path.isfile(os.path.join(data_path, imName)):
            # 如果文件存在
            #saveName = os.path.splitext(imName)[0] + "_" + str(0) + expanded_name
            # 保存文件名，去掉文件扩展名并添加特定扩展名
            saveName = os.path.splitext(imName)[0] + expanded_name
            saveName = saveName.replace("/image/", "/zkysocr/")  # 替换文件路径中的部分字符串
            if not os.path.isfile(os.path.join(save_path, saveName)):
                # 如果目标文件不存在于保存路径
                res.append(imName)  # 将文件名添加到结果列表中

    return res


# 查找未处理的文件，多线程
def FindUnProcessedData_Parallel(data_path, save_path, imNames, expanded_name=".txt", thresNum=None):
    #thresNum = 4
    thresNum = min(int(mp.cpu_count()) * 2, len(imNames)) if thresNum is None else thresNum
    eachThresImNum = math.ceil(len(imNames) / thresNum)

    results = []
    pool = mp.Pool(thresNum)
    for i in range(thresNum):
        start = i * eachThresImNum
        end = min(start + eachThresImNum, len(imNames))
        subImNames = imNames[start:end]
        results.append(pool.apply_async(FindUnProcessedData, args=(i, data_path, save_path, subImNames, expanded_name)))

    results = [p.get() for p in results]
    res = []
    for item in results:
        res += item

    return res


# 中科阅深OCR
def SplitTiffIntoJpgs(thres_id, data_path, save_path, imNames):
    start_time = time.time()  
    last_time = time.time() 
    im_num = 0
    for index, imName in enumerate(imNames):
        if index % 100 == 0:
            time_cost = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            eta_seconds = ((time.time() - last_time) / 100) * (len(imNames) - index)
            time_need = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(str(thres_id) + ": " + str(index) + "/" + str(len(imNames)) + ": " + time_cost + "<--" + time_need + ": " + imName)
            last_time = time.time()

        try:
            img = Image.open(os.path.join(data_path, imName))
            for i in range(img.n_frames):
                img.seek(i)

                jpgName = os.path.splitext(imName)[0] + "_" + str(i) + '.png'
                MakePath(os.path.join(save_path, jpgName))
                img.save(os.path.join(save_path, jpgName))

        except:
            print(str(index) + "/" + str(len(imNames)) + ": " + imName + " exists error, skipped.")

    print("im_num:", im_num)
    return 


# 生成训练样本，多线程
def SplitTiffIntoJpgs_Parallel(data_path, save_path, data_list, thresNum=None):
    start = timeit.default_timer()

    imNames = [item.strip().split(" ")[0] for item in open(data_list).readlines()]

    end = timeit.default_timer()
    print('read data list time:', end-start,'seconds')

    #imNames = imNames[0:200]
    print("Total num:", len(imNames))
    imNames = FindUnProcessedData_Parallel(data_path, save_path, imNames, expanded_name=".png", thresNum=64)
    print("Unprocessed num:", len(imNames))

    ## 先处理一张图片，生成所有存储路径，避免多线程创建路径冲突
    #SplitTiffIntoJpgs(0, data_path, save_path, imNames)
    #return 

    #thresNum = 4
    thresNum = min(int(mp.cpu_count()) * 2, len(imNames)) if thresNum is None else thresNum
    eachThresImNum = math.ceil(len(imNames) / thresNum)

    results = []
    pool = mp.Pool(thresNum)
    for i in range(thresNum):
        start = i * eachThresImNum
        end = min(start + eachThresImNum, len(imNames))
        subImNames = imNames[start:end]
        results.append(pool.apply_async(SplitTiffIntoJpgs, args=(i, data_path, save_path, subImNames,)))

    results = [p.get() for p in results]


def Paddle_OCR(thres_id, data_path, save_path, imNames):
    from paddleocr import PaddleOCR, draw_ocr
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    # det_box_type="poly", "quad"
    # need to run only once to download and load model into memory
    ocr_ch = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True, gpu_id=thres_id % 8, show_log=False, \
                    det_algorithm="DB", det_model_dir="./models/ch_PP-OCRv4_det_server_infer", det_box_type="poly", det_limit_type="max", det_limit_side_len=2000, max_batch_size=10, \
                    rec_algorithm="SVTR_LCNet", rec_model_dir="./models/ch_PP-OCRv4_rec_server_infer", rec_image_shape=(3,48,480), max_text_length=250, rec_batch_num=16, \
                    drop_score=0.0,) 

    ocr_en = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True, gpu_id=thres_id % 8, show_log=False, \
                    det_algorithm="DB", det_model_dir="./models/en_PP-OCRv3_det_infer", det_box_type="poly", det_limit_type="max", det_limit_side_len=2000, max_batch_size=10, \
                    rec_algorithm="SVTR_LCNet", rec_model_dir="./models/en_PP-OCRv4_rec_infer", rec_image_shape=(3,48,480), max_text_length=250, rec_batch_num=16, \
                    drop_score=0.0,) 

    start_time = time.time()  
    last_time = time.time() 
    
    im_num = 0
    for index, imName in enumerate(imNames):
        if index % 10 == 0:
            time_cost = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            eta_seconds = ((time.time() - last_time) / 10) * (len(imNames) - index)
            time_need = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(str(thres_id) + ": " + str(index) + "/" + str(len(imNames)) + ": " + time_cost + "<--" + time_need + ": " + imName)
            last_time = time.time()

        try:
            imgs = tifread(os.path.join(data_path, imName))
            for i, img in enumerate(imgs):
                im_num += 1
                
                #scale = 1000.0 / max(img.shape[0], img.shape[1])
                scale_1 = 1024.0 / min(img.shape[0], img.shape[1])
                scale_2 = 2048.0 / max(img.shape[0], img.shape[1])
                scale = min(scale_1, scale_2)
                img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

                result_ch = ocr_ch.ocr(img, cls=True, det=True, rec=True)[0]
                result_en = ocr_en.ocr(img, cls=True, det=True, rec=True)[0]
                if len(result_ch) > len(result_en):
                    result = result_ch
                else:
                    result = result_en
                
                results = {"lines":[]}
                for line in result:
                    text = line[1][0]
                    score = line[1][1]
                    coords = []
                    for p in line[0]:
                        if len(coords) == 0:
                            coords.append(p[0])
                            coords.append(p[1])
                        else:
                            if p[0] != coords[-2] and p[1] != coords[-1]:
                                coords.append(p[0])
                                coords.append(p[1])
                    line = {}
                    line["text"] = text
                    line["coords"] = coords
                    line["chars"] = []
                    results["lines"].append(line)
                
                results["hei"] = img.shape[0]
                results["wid"] = img.shape[1]
                if len(results) != 0:
                    txtName = os.path.splitext(imName)[0] + "_" + str(i) + '.txt'
                    MakePath(os.path.join(save_path, txtName))
                    with open(os.path.join(save_path, txtName), 'w') as f:
                        f.write(json_to_str(results))

                    if len(img.shape) == 2:
                        res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        res = img
                    res = visulization(res, results)
                    jpgName = os.path.splitext(imName)[0] + "_" + str(i) + '.jpg'
                    MakePath(os.path.join(save_path, jpgName))
                    cv2.imwrite(os.path.join(save_path, jpgName), res)

        except:
            print(str(index) + "/" + str(len(imNames)) + ": " + imName + " exists error, skipped.")

    print("im_num:", im_num)
    return 


# 中科阅深OCR
def YueShen_OCR(thres_id, data_path, save_path, imNames):
    start_time = time.time()  
    last_time = time.time() 
    
    im_num = 0
    for index, imName in enumerate(imNames):
        if index % 10 == 0:
            time_cost = str(datetime.timedelta(seconds=int(time.time() - start_time)))
            eta_seconds = ((time.time() - last_time) / 10) * (len(imNames) - index)
            time_need = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(str(thres_id) + ": " + str(index) + "/" + str(len(imNames)) + ": " + time_cost + "<--" + time_need + ": " + imName)
            last_time = time.time()

        try:
            img = cv2.imread(os.path.join(data_path, imName), cv2.IMREAD_COLOR)
            #img = jpeg.JPEG(os.path.join(data_path, imName)).decode()

            #scale = 1000.0 / max(img.shape[0], img.shape[1])
            scale_1 = 1024.0 / min(img.shape[0], img.shape[1])
            scale_2 = 2048.0 / max(img.shape[0], img.shape[1])
            scale = min(scale_1, scale_2)
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            url = 'http://172.18.16.11:8910/ocr/ft-scene'
            header = {
                "Content-Type": "application/x-www-form-urlencoded"
            }
            body = {
                'base64img': numpy_to_base64(img),
            }

            r = requests.post(url, headers=header, data=body, timeout=100)
            results = json.loads(r.content.decode())["data"]
            
            def scale_coordinates(results, scale_factor):
                results_lines = results["lines"]
                results_coords = results["coords"]

                results_coords = [int(c/scale_factor) for c in results_coords]

                """将OCR结果的坐标从缩放图像映射回原始图像"""
                for i in range(len(results_lines)):
                    box = results_lines[i].get("coords", [])
                    box = [int(b/scale_factor) for b in box]
                    results_lines[i]['coords'] = box
                
                results["coords"] = results_coords
                results["lines"] = results_lines
                return results

            ####
            char_ratio = char_area_ratio(img, results)
            if char_ratio < 0.1:
                body_rot = {
                    'base64img': numpy_to_base64(np.rot90(img, -1)),
                }

                r_rot = requests.post(url, headers=header, data=body_rot, timeout=100)
                results_rot = json.loads(r_rot.content.decode())["data"]
                results_rot = anno_rotate(img, results_rot)
                char_ratio_rot = char_area_ratio(img, results_rot)

                if char_ratio < char_ratio_rot:
                    results = results_rot
            ####
            results = scale_coordinates(results, scale)

            results["hei"] = img.shape[0]
            results["wid"] = img.shape[1]
            if len(results) != 0:
                txtName = os.path.splitext(imName)[0] + '.txt'
                txtName = txtName.replace("/image", "/zkysocr")
                MakePath(os.path.join(save_path, txtName))
                with open(os.path.join(save_path, txtName), 'w') as f:
                    f.write(json_to_str(results))

                if im_num < 10:
                    res = visulization(copy.deepcopy(img), results)
                    jpgName = os.path.splitext(imName)[0] + '.jpg'
                    jpgName = jpgName.replace("/image", "/zkysocr")
                    MakePath(os.path.join(save_path, jpgName))
                    cv2.imwrite(os.path.join(save_path, jpgName), res)
            im_num += 1
        except:
            # 将报错信息打印
            print(sys.exc_info())
            print(str(index) + "/" + str(len(imNames)) + ": " + imName + " exists error, skipped.")

    print("im_num:", im_num)
    return 


# 生成训练样本，多线程
def YueShen_OCR_Parallel(data_path, save_path, data_list, thresNum=None):
    start = timeit.default_timer()

    imNames = [item.strip().split(" ")[0] for item in open(data_list, errors='ignore').readlines()]
    
    end = timeit.default_timer()
    print('read data list time:', end-start,'seconds')

    # imNames = imNames[0:100]
    print("Total num:", len(imNames))
    # imNames = FindUnProcessedData_Parallel(data_path, save_path, imNames, expanded_name=".txt", thresNum=64)
    print("Unprocessed num:", len(imNames))

    # ## 先处理一张图片，生成所有存储路径，避免多线程创建路径冲突
    # YueShen_OCR(0, data_path, save_path, imNames)
    # return 

    #thresNum = 4
    thresNum = min(int(mp.cpu_count()) * 2, len(imNames)) if thresNum is None else thresNum
    eachThresImNum = math.ceil(len(imNames) / thresNum)

    results = []
    pool = mp.Pool(thresNum)
    for i in range(thresNum):
        start = i * eachThresImNum
        end = min(start + eachThresImNum, len(imNames))
        subImNames = imNames[start:end]
        results.append(pool.apply_async(YueShen_OCR, args=(i, data_path, save_path, subImNames,)))

    results = [p.get() for p in results]


def get_all_txts(txts_root):
    txts = []
    # 收集所有PNG图片的相对路径
    image_paths = []
    for root, dirs, files in os.walk(txts_root):
        # 跳过隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.lower().endswith('.txt') and not file.startswith('.'):
                # 获取相对于根目录的路径
                rel_path = os.path.relpath(os.path.join(root, file), txts_root)
                image_paths.append(rel_path)
    
    # 分层抽样（保持目录结构分布）
    path_groups = defaultdict(list)
    for path in image_paths:
        dir_part = os.path.dirname(path)
        path_groups[dir_part].append(path)
    
    # 对每个目录单独抽样
    for dir_path, files in path_groups.items():
        txts.extend(files)
    
    return txts

# 判断字符串是否仅包含特殊字符,规定除数字、字母以外都是特殊字符
def only_special_characters(string):
    return not any(c.isalnum() for c in string)

# 评估
def evaluate_YueShen(data_path, save_path, label_path):
    # 读取检测结果，从中获取boxes和texts
    txt_list = get_all_txts(save_path)
    IOU_scores, OCR_scores, F1_scores = [], [], []
    for txt in txt_list[:]:
        boxes, texts = [], []
        image_path = os.path.join(data_path, txt.replace(".txt", ".png"))
        if not os.path.exists(image_path):
            image_path = os.path.join(data_path, txt.replace(".txt", ".jpg"))
        image = Image.open(image_path).convert('RGB')
        with open(os.path.join(save_path, txt), 'r') as f:
            data = f.readlines()
        for line in data[1:]:
            items = line.strip().split(",")
            box = items[1].strip().split(" ")
            text = items[-1].strip()
            if only_special_characters(text):
                continue
            x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(x)) for x in box]
            box = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            boxes.append(box)
            texts.append(text)
        # 对boxes进行缩紧和合并操作，得到最终的boxes和texts
        boxes, texts = merge_text_blocks(tighten_boxes(image, boxes), texts)
        # 绘制boxes,并保存到save_path, box格式为[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        img_with_boxes = np.array(image.copy())
        for box in boxes:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_with_boxes, [pts], isClosed=True, color=(0,255,0), thickness=2)
        cv2.imwrite(os.path.join(save_path, txt.replace(".txt", ".png")), img_with_boxes)
        print(f"Save to {os.path.join(save_path, txt.replace('.txt', '.png'))}")
        pred_data = [(parse_box(box), text) for box, text in zip(boxes, texts)]

        # 读取ground truth，从中获取boxes和texts
        with open(os.path.join(label_path, txt.replace(".txt", ".json")), 'r') as f:
            label = json.load(f)
            label = label['task2']['output']["text_blocks"]

            gt_data = [(parse_box(item['polygon']), item['text'].replace('\n', ' ')) for item in label]

        IOU_score, OCR_score, F1_score = evaluate(gt_data, pred_data)
        print(f"IOU_score: {IOU_score:.4f}, OCR_score: {OCR_score:.4f}, F1_score: {F1_score:.4f}")
        IOU_scores.append(IOU_score)
        OCR_scores.append(OCR_score)
        F1_scores.append(F1_score)
    print(f"IOU score: {np.mean(IOU_scores)}, OCR score: {np.mean(OCR_scores)}, F1 score: {np.mean(F1_scores)}")


if __name__ == "__main__":

    start = timeit.default_timer()
    
    data_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/images"
    save_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/YueShen_results/chartqa"
    data_list = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/list.txt"
    label_path = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ICDAR2019/json"
    # data_path = sys.argv[1]
    # save_path = sys.argv[2]
    # data_list = sys.argv[3]
    print(data_path)
    print(save_path)
    print(data_list)
    # YueShen_OCR_Parallel(data_path, save_path, data_list, thresNum=10)
    
    evaluate_YueShen(data_path, save_path, label_path)

    end = timeit.default_timer()
    print('total time:', end-start,'seconds')
    
    