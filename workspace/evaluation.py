import sys
sys.path.append('/data5/home/xiechenyu2023/project/ChartQA/chart-to-table')
from tools.metrics import table_datapoints_precision_recall_per_point
import re
import os
import json
import math
import csv
from tqdm import tqdm

def check_table_format(text):
    elements = re.findall(r'\[.*?\]', text[1:-1])
    for element in elements:
        element = list(element)
        print(element)
        print(type(element))

def get_all_files(directory):
    files = []
    for root, dirs, files_in_dir in os.walk(directory):
        for file in files_in_dir:
            files.append(os.path.relpath(os.path.join(root, file), directory))
    return files


def evaluate_in_chartqa(pred_dir, gt_dir, version='v1'):
    predictions = []
    gts = []
    
    pred_files = get_all_files(pred_dir)
    # pred_files = os.listdir(pred_dir)
    gt_files = get_all_files(gt_dir)
    # pred_files = [item for item in pred_files if item in gt_files]
    # pie_files = os.listdir("/lustre/home/xiechenyu2023/ChartQA_Dataset/DeChart/test_split/pie")
    # pie_files = [item.replace(".png", ".csv") for item in pie_files]
    # line_files = os.listdir("/lustre/home/xiechenyu2023/ChartQA_Dataset/DeChart/test_split/line")
    # line_files = [item.replace(".png", ".csv") for item in line_files]
    # pred_files = [item for item in pred_files if item in pie_files]
    # pred_files = [item for item in pred_files if item not in line_files and item not in pie_files]
    # pred_files = [item for item in pred_files if item in gt_files]
    for file in tqdm(pred_files):
        if not file.endswith('.csv'):
            continue
        with open(os.path.join(pred_dir, file), "r") as pf:
            # data = pf.readlines()
            # pred = [line.strip().split('<sep>') for line in data]
            data = csv.reader(pf)
            pred = [line for line in data]
            # pred = pred[1:]
        # if not os.path.exists(os.path.join(gt_dir, file)):
        #     continue
        with open(os.path.join(gt_dir, file), "r") as gtf:
            data = csv.reader(gtf)
            gt = [line for line in data]
            # gt = [line.strip().split(",") for line in data]
        
        predictions.append(pred)
        gts.append(gt)
    
    result = table_datapoints_precision_recall_per_point(gts, predictions, version=version)
    for img, p, r, f1 in zip(pred_files, result["precision"], result["recall"], result["f1"]):
        if p<0.5 or r<0.5 or f1<0.5:
            print(f"{img} : {p:.2f}, {r:.2f}, {f1:.2f}")
    precision = sum(result["precision"])/len(result["precision"])
    recall = sum(result["recall"])/len(result["recall"])
    f1_ = sum(result["f1"])/len(result["f1"])
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1_: ", f1_)

if __name__ == "__main__":
    pred_dir = "/data5/home/xiechenyu2023/project/ChartQA/CPAgent/output/Qwen3-VL-8B/chartqa_pie"
    gt_dir = "/data5/home/xiechenyu2023/project/ChartQA/mydatasets/ChartQA/test_pie/tables"
    evaluate_in_chartqa(pred_dir, gt_dir, version='v2')
