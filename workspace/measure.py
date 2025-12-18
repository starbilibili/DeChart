import Levenshtein
from collections import defaultdict

def measure_task1(output):
    total_num = len(output)
    cnt = 0
    for item in output:
        if item["prediction"] == item["gt"]:
            cnt += 1
        else:
            print(f"{item['image_path']}:{item['prediction']}: {item['gt']}")
    acc = cnt / total_num * 100
    print("Accuracy: ", acc)

def measure_task2(outputs):
    total_samples = len(outputs)
    detection_results = defaultdict(list)
    classification_results = defaultdict(list)
    for item in outputs:
        prediction = item["prediction"]
        gt = item["gt"]
        pred_parts = [p.split(":") for p in prediction.split("<SEP>")]
        gt_parts = [g.split(":") for g in gt.split("<SEP>")]
        for (p_text, p_type), (g_text, g_type) in zip(pred_parts, gt_parts):
            # 计算归一化编辑距离
            edit_dist = Levenshtein.distance(p_text, g_text)
            max_len = max(len(p_text), len(g_text))
            similarity = 1 - (edit_dist / max_len) if max_len > 0 else 1.0
            
            detection_results[p_type].append(similarity)

        pred_types = [p[1] for p in pred_parts if len(p) > 1]
        gt_types = [g[1] for g in gt_parts if len(g) > 1]
        
        for p_type, g_type in zip(pred_types, gt_types):
            classification_results[g_type].append(int(p_type == g_type))

    # 计算整体指标
    def calc_avg(scores):
        return sum(scores)/len(scores) if scores else 0.0
    
    detection_metrics = {
        k: {"accuracy": calc_avg(v), "samples": len(v)} 
        for k, v in detection_results.items()
    }
    detection_metrics["overall"] = calc_avg(
        [v for scores in detection_results.values() for v in scores]
    )
    
    classification_metrics = {
        k: {"accuracy": calc_avg(v), "samples": len(v)} 
        for k, v in classification_results.items()
    }
    classification_metrics["overall"] = calc_avg(
        [v for scores in classification_results.values() for v in scores]
    )
    
    return {
        "text_detection": detection_metrics,
        "text_classification": classification_metrics,
        "total_samples": total_samples
    }