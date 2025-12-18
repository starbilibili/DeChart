# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metrics functions for Chart and Table related tasks."""

from collections.abc import Mapping, Sequence
from collections import defaultdict
import dataclasses
import itertools
from typing import Optional

import numpy as np
import sys
import pix2struct_metrics
# from pix2struct import metrics as pix2struct_metrics
from scipy import optimize


# 百分数转换成浮点数
def _to_float(text):
  try:
    if text.endswith("%"):
      # Convert percentages to floats.
      return float(text.rstrip("%"))
    else:
      return float(text)
  except ValueError:
    return None

def _is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
# 计算目标值和预测值之间相对误差
def _get_relative_distance(
    target, prediction, theta = 1.0
):
  """Returns min(1, |target-prediction|/|target|)."""
  if not target:
    return int(not prediction)
  distance = min(abs((target - prediction) / target), 1)
  return distance if distance < theta else 1

# 计算两个表格中数字的匹配相似度，使用相对距离和线性匹配。
def _table_numbers_match(target, prediction):
  """Calculates matching similarity between two tables following ChartQA."""
  target_numbers = _get_table_numbers(target)
  prediction_numbers = _get_table_numbers(prediction)
  if not target_numbers and not prediction_numbers:
    return 1
  if not target_numbers or not prediction_numbers:
    return 0
  max_len = max(len(target_numbers), len(prediction_numbers))
  distance = []
  for t in target_numbers:
    distance.append([_get_relative_distance(t, p) for p in prediction_numbers])
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  return 1 - cost_matrix[row_ind, col_ind].sum() / max_len

# 从给定的文本中提取数字，这些数字可能在表格的单元格中。
def _get_table_numbers(text):
  numbers = []
  for line in text.splitlines():
    for part in line.split(" | "):
      if part.strip():
        try:
          numbers.append(float(part))
        except ValueError:
          pass
  return numbers

# 计算每个目标表格与预测表格之间的数字匹配相似度，并返回一个包含这些相似度的列表。
def table_number_accuracy_per_point(
    targets,
    predictions,
):
  """Calculates matching similarity between two tables following ChartQA.

  Keeps only numbers and performas a linear matching using the relative error.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    A list of float numbers.
  """
  all_points_scores = []
  for p, targets in zip(predictions, targets):
    all_points_scores.append(max(_table_numbers_match(t, p) for t in targets))
  return all_points_scores

# 同时计算多个表格的table_number_accuracy_per_point，返回一个包含整体匹配相似度的字典。
def table_number_accuracy(
    targets,
    predictions,
):
  """Aggregated version of table_number_accuracy_per_point().

  Same as table_number_accuracy_per_point() but returning an aggregated score.

  Args:
    targets: ground truth text.
    predictions: predicted text.

  Returns:
    dictionary with metric names as keys and metric value as values.
  """
  scores = table_number_accuracy_per_point(targets, predictions)
  return {"numbers_match": (100.0 * sum(scores)) / len(targets)}

# 根据给定的索引重新排列值，并返回一个元组
def _permute(values, indexes):
  return tuple(values[i] if i < len(values) else "" for i in indexes)


@dataclasses.dataclass(frozen=True)
class Table:
  """Helper class for the content of a markdown table."""

  title: Optional[str] = None
  headers: tuple[str, Ellipsis] = dataclasses.field(default_factory=tuple)
  rows: tuple[tuple[str, Ellipsis], Ellipsis] = dataclasses.field(default_factory=tuple)

  def permuted(self, indexes):
    """Builds a version of the table changing the column order."""
    return Table(
        title=self.title,
        headers=_permute(self.headers, indexes),
        rows=tuple(_permute(row, indexes) for row in self.rows),
    )
  
  def aligned(
      self, headers, text_theta = 0.5
  ):
    """Builds a column permutation with headers in the most correct order."""
    if len(headers) != len(self.headers):
      raise ValueError(f"Header length {headers} must match {self.headers}.")
    distance = []
    for h2 in self.headers:
      distance.append(
          [
              1 - pix2struct_metrics.anls_metric(h1, h2, text_theta)
              for h1 in headers
          ]
      )
    cost_matrix = np.array(distance)
    row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
    permutation = [idx for _, idx in sorted(zip(col_ind, row_ind))]
    score = (1 - cost_matrix)[permutation[1:], range(1, len(row_ind))].prod()
    return self.permuted(permutation), score

# 从Markdown格式的字符串中解析出表格，并可选择是否转置表格。
def _parse_table_from_markdown(text, transposed = False):
  """Builds a table from a markdown representation."""
  lines = text.lower().splitlines()
  if not lines:
    return Table()
  if lines[0].startswith("title |"):
    title = lines[0][len("title |") :].strip()
    offset = 1
  else:
    title = None
    offset = 0
  if len(lines) < offset + 1:
    return Table(title=title)
  rows = []
  for line in lines[offset:]:
    rows.append(tuple(v.strip() for v in line.split(" | ")))
  if transposed:
    rows = [tuple(row) for row in itertools.zip_longest(*rows, fillvalue="")]
  return Table(title=title, headers=rows[0], rows=tuple(rows[1:]))

def _parse_table_from_list(text_list, transposed = False):
  """Builds a table from a list of lists."""
  if not text_list:
    return Table()
  text_list = [(row) for row in text_list]
  if transposed:
    text_list = [tuple(row) for row in itertools.zip_longest(*text_list, fillvalue="")]
  return Table(title=None, headers=text_list[0], rows=tuple(text_list[1:]))

# 从表格中提取数据点，并返回一个字典。
def _get_table_datapoints(table):
  """Extracts a dict of datapoints from a table."""
  datapoints = {}
  if table.title is not None:
    datapoints["title"] = table.title
  if not table.rows or len(table.headers) <= 1:
    return datapoints
  for row in table.rows:
    for header, cell in zip(table.headers[1:], row[1:]):
      datapoints[f"{row[0]} {header}"] = cell
  return datapoints

def _get_table_datapoints_v2(table):
  """Extracts a dict of datapoints from a table."""
  datapoints = defaultdict(dict)
  if table.title is not None:
    datapoints["title"] = table.title
  
  if not table.rows or len(table.headers) <= 1:
    return datapoints
  for row in table.rows:
    row_key = row[0]
    for header, cell in zip(table.headers[1:], row[1:]):
      # datapoints[f"{row[0]} {header}"] = cell
      datapoints[row_key][header] = cell
  return datapoints

# 计算两个数据点之间的相似度，考虑文本和数值的匹配。
def _get_datapoint_metric(
    target,
    prediction,
    text_theta=0.5,
    number_theta=0.1,
):
  """Computes a metric that scores how similar two datapoint pairs are."""
  key_metric = pix2struct_metrics.anls_metric(
      target[0], prediction[0], text_theta
  )
  pred_float = _to_float(prediction[1])
  target_float = _to_float(target[1])
  if pred_float is not None and target_float:
    return key_metric * (
        1 - _get_relative_distance(target_float, pred_float, number_theta)
    )
  elif target[1] == prediction[1]:
    return key_metric
  else:
    return key_metric * pix2struct_metrics.anls_metric(
        target[1], prediction[1], text_theta
    )

# 计算两个表格数据点的精确度、召回率和F1分数。
def _table_datapoints_precision_recall_f1_v2(
    target_table,
    prediction_table,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """
  计算两个表格之间的匹配相似度。

  Args:
      target_table (dict): 目标表格，字典形式。
      prediction_table (dict): 预测表格，字典形式。
      text_theta (float): 文本相似度阈值，默认为0.5。
      number_theta (float): 数字相似度阈值，默认为0.1。

  Returns:
      tuple: 包含精确率、召回率和F1分数的元组。

  """
  """Calculates matching similarity between two tables as dicts."""
  """
  计算两个表格之间的匹配相似度。

  Args:
      target_table (dict): 目标表格，字典形式。
      prediction_table (dict): 预测表格，字典形式。
      text_theta (float): 文本相似度阈值，默认为0.5。
      number_theta (float): 数字相似度阈值，默认为0.1。

  Returns:
      tuple: 包含精确率、召回率和F1分数的元组。

  """
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints_v2(target_table).items())
  prediction_datapoints = list(_get_table_datapoints_v2(prediction_table).items())
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints:
    return 1, 0, 0
  distance = []
  for t, _ in target_datapoints:
      distance.append([
          1 - pix2struct_metrics.anls_metric(t, p, text_theta)
          for p, _ in prediction_datapoints
      ])
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  score = 0
  for r, c in zip(row_ind, col_ind):
    t_r = target_datapoints[r][0]
    t_c_list = list(target_datapoints[r][1].items())
    p_r = prediction_datapoints[c][0]
    p_c_list = list(prediction_datapoints[c][1].items())
    score_r = pix2struct_metrics.anls_metric(
        t_r, p_r, text_theta
    )
    score_c_list = []
    for t_c_item in t_c_list:
      score_c_list.append([1-pix2struct_metrics.anls_metric(
          t_c_item[0], p_c_item[0], text_theta
      ) for p_c_item in p_c_list
      ])
    cost_socre_c = np.array(score_c_list)
    row_score_c, col_score_c = optimize.linear_sum_assignment(cost_socre_c)
    for r_score_c, c_score_c in zip(row_score_c, col_score_c):
      t_c = t_c_list[r_score_c][0].strip().strip('"')
      p_c = p_c_list[c_score_c][0].strip().strip('"')
      t_v = _to_float(t_c_list[r_score_c][1].strip('"'))
      p_v = _to_float(p_c_list[c_score_c][1].strip('"'))
      score_c = pix2struct_metrics.anls_metric(t_c, p_c, text_theta)
      if t_v and p_v is not None:
        score_v = 1 - _get_relative_distance(t_v, p_v, number_theta)
      elif t_c_list[r_score_c][1] == 'nan' and p_v == 0. or t_v == 0. and p_c_list[c_score_c][1] == 'nan':
        score_v = 1
      else:
        score_v = pix2struct_metrics.anls_metric(t_c_list[r_score_c][1], p_c_list[c_score_c][1], text_theta)
      score += (1+score_r) / 2 * (1+score_c) / 2 * score_v
    # score += _get_datapoint_metric(
    #     target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
    # )
  if score == 0:
    return 0, 0, 0
  # 考虑到预测到的表格会包含title，而真实表格可能不包含title，这会导致准确率计算偏低
  len_prediction_datapoints = len(prediction_datapoints) * len(prediction_datapoints[0][1])
  len_target_datapoints = len(target_datapoints) * len(target_datapoints[0][1])
  precision = score / len_prediction_datapoints
  recall = score / len_target_datapoints
  # precision = score / len(prediction_datapoints)
  # recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)

# 计算两个表格数据点的精确度、召回率和F1分数。
def _table_datapoints_precision_recall_f1(
    target_table,
    prediction_table,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """
  计算两个表格之间的匹配相似度。

  Args:
      target_table (dict): 目标表格，字典形式。
      prediction_table (dict): 预测表格，字典形式。
      text_theta (float): 文本相似度阈值，默认为0.5。
      number_theta (float): 数字相似度阈值，默认为0.1。

  Returns:
      tuple: 包含精确率、召回率和F1分数的元组。

  """
  """Calculates matching similarity between two tables as dicts."""
  target_datapoints = list(_get_table_datapoints(target_table).items())
  prediction_datapoints = list(_get_table_datapoints(prediction_table).items())
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints:
    return 1, 0, 0
  distance = []
  for t, _ in target_datapoints:
    distance.append(
        [
            1 - pix2struct_metrics.anls_metric(t, p, text_theta)
            for p, _ in prediction_datapoints
        ]
    )
  cost_matrix = np.array(distance)
  row_ind, col_ind = optimize.linear_sum_assignment(cost_matrix)
  score = 0
  for r, c in zip(row_ind, col_ind):
    score += _get_datapoint_metric(
        target_datapoints[r], prediction_datapoints[c], text_theta, number_theta
    )
  if score == 0:
    return 0, 0, 0
  # 考虑到预测到的表格会包含title，而真实表格可能不包含title，这会导致准确率计算偏低
  len_prediction_datapoints = len(prediction_datapoints)
  len_target_datapoints = len(target_datapoints)
  if target_datapoints[0][0] == 'title' and prediction_datapoints[0][0]!='title':
    len_target_datapoints-=1
  elif target_datapoints[0][0] != 'title' and prediction_datapoints[0][0]=='title':
    len_prediction_datapoints-=1
  precision = score / len_prediction_datapoints
  recall = score / len_target_datapoints
  # precision = score / len(prediction_datapoints)
  # recall = score / len(target_datapoints)
  return precision, recall, 2 * precision * recall / (precision + recall)


# 计算多个目标表格与预测表格之间的数据点精确度、召回率和F1分数，并返回一个包含这些分数的字典。
def table_datapoints_precision_recall_per_point(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
    version = 'v1'
):
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a dictionary of keys and values using row and column
  headers. Then we match keys between the two dicts as long as their relative
  levenshtein distance is below a threshold. Values are also compared with
  ANLS if strings or relative distance if they are numeric.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with per-point precision, recall and F1
  """
  assert len(targets) == len(predictions)
  per_point_scores = {"precision": [], "recall": [], "f1": []}

  if version == 'v1':
    metric_compute_function = _table_datapoints_precision_recall_f1
  elif version == 'v2':
    metric_compute_function = _table_datapoints_precision_recall_f1_v2
  else:
    raise ValueError("Unknown version {}".format(version))
  for pred, target in zip(predictions, targets):
    # 检查数据类型
    if type(pred) is str:
      _parse_table = _parse_table_from_markdown
    elif type(pred) is list:
      _parse_table = _parse_table_from_list
    all_metrics = []
    for transposed in [False, True]:
      pred_table = _parse_table(pred, transposed=transposed)
      target_table = _parse_table(target)
      
      # pylint:disable=g-complex-comprehension
      all_metrics.extend(
          [
              metric_compute_function(
                  target_table,
                  pred_table,
                  text_theta,
                  number_theta,
              )
          ]
      )
      # pylint:enable=g-complex-comprehension
    p, r, f = max(all_metrics, key=lambda x: x[-1])
    per_point_scores["precision"].append(p)
    per_point_scores["recall"].append(r)
    per_point_scores["f1"].append(f)
  return per_point_scores

# 聚合版本的table_datapoints_precision_recall_per_point，返回一个包含整体精确度、召回率和F1分数的字典。
def table_datapoints_precision_recall(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Aggregated version of table_datapoints_precision_recall_per_point().

  Same as table_datapoints_precision_recall_per_point() but returning aggregated
  scores instead of per-point scores.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Dictionary with aggregated precision, recall and F1
  """
  score_dict = table_datapoints_precision_recall_per_point(
      targets, predictions, text_theta, number_theta
  )
  return {
      "table_datapoints_precision": (
          100.0 * sum(score_dict["precision"]) / len(targets)
      ),
      "table_datapoints_recall": (
          100.0 * sum(score_dict["recall"]) / len(targets)
      ),
      "table_datapoints_f1": 100.0 * sum(score_dict["f1"]) / len(targets),
  }

# 从表格中提取行数据点
def _get_row_datapoints(table):
  """Extracts a list of datapoints from a table as rows."""
  if table.title is None:
    return table.rows
  return table.rows + (("title", table.title),)

# 计算两行数据点之间的相似度
def _get_row_metric(
    target_parts,
    prediction_parts,
    text_theta=0.5,
    number_theta=0.1,
):
  """Computes a metric that scores how similar two datapoint pairs are."""
  if len(target_parts) != len(prediction_parts) or not target_parts:
    return 0.0
  result = []
  for target, prediction in zip(target_parts, prediction_parts):
    pred_float = _to_float(prediction)
    target_float = _to_float(target)
    if target == prediction:
      result.append(1.0)
    elif pred_float is not None and target_float:
      result.append(
          1 - _get_relative_distance(target_float, pred_float, number_theta)
      )
    elif target_float is not None:
      result.append(0.0)
    else:
      result.append(
          pix2struct_metrics.anls_metric(target, prediction, text_theta)
      )
  return np.prod(result)

# 计算两个表格行数据点的精确度、召回率和F1分数。
def _row_datapoints_precision_recall_f1(
    target,
    prediction,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Calculates matching similarity between two tables as list of rows."""
  target_datapoints = _get_row_datapoints(target)
  aligned_prediction, aligned_score = prediction.aligned(
      target.headers, text_theta
  )
  prediction_datapoints = _get_row_datapoints(aligned_prediction)
  if not target_datapoints and not prediction_datapoints:
    return 1, 1, 1
  if not target_datapoints:
    return 0, 1, 0
  if not prediction_datapoints or not aligned_score:
    return 1, 0, 0
  metrics = []
  for t in target_datapoints:
    metrics.append(
        [
            aligned_score * _get_row_metric(t, p, text_theta, number_theta)
            for p in prediction_datapoints
        ]
    )
  metrics_matrix = np.array(metrics)
  row_ind, col_ind = optimize.linear_sum_assignment(1 - metrics_matrix)
  score = metrics_matrix[row_ind, col_ind].sum()
  if score == 0:
    return 0, 0, 0
  # 考虑到预测到的表格会包含title，而真实表格可能不包含title，这会导致准确率计算偏低
  len_prediction_datapoints = len(prediction_datapoints)
  len_target_datapoints = len(target_datapoints)
  print(target_datapoints[0])
  print(prediction_datapoints[0])
  # if target_datapoints[0][0] == 'title' and prediction_datapoints[0][0]!='title':
  #   len_target_datapoints-=1
  # elif target_datapoints[0][0] != 'title' and prediction_datapoints[0][0]=='title':
  #   len_prediction_datapoints-=1
  precision = score / len_prediction_datapoints
  recall = score / len_target_datapoints
  return precision, recall, 2 * precision * recall / (precision + recall)

# 计算每个目标表格与预测表格之间的行数据点精确度、召回率和F1分数，并返回一个包含这些分数的字典
def row_datapoints_precision_recall(
    targets,
    predictions,
    text_theta = 0.5,
    number_theta = 0.1,
):
  """Computes precisin recall and F1 metrics given two flattened tables.

  Parses each string into a list of rows using column headers. Then we match
  entries by their levenshtein / numeric relative distance is below a threshold.

  Args:
    targets: list of list of strings.
    predictions: list of strings.
    text_theta: relative edit distance above this is set to the maximum of 1.
    number_theta: relative error rate above this is set to the maximum of 1.

  Returns:
    Mapping with precision, recall and F1
  """
  if len(targets) != len(predictions):
    raise ValueError(
        f"Targets has length {len(targets)} and predictions has length "
        f"{len(predictions)}."
    )
  precision, recall, f1 = 0, 0, 0
  for pred, target in zip(predictions, targets):
    if type(pred) is str:
      _parse_table = _parse_table_from_markdown
    elif type(pred) is list:
      _parse_table = _parse_table_from_list
    all_metrics = []
    prediction_tables = [
        _parse_table(pred, transposed=transposed)
        for transposed in [True, False]
    ]
    for t in target:
      for target_transposed in [True, False]:
        target_table = _parse_table(t, transposed=target_transposed)
        for prediction_table in prediction_tables:
          if len(target_table.headers) != len(prediction_table.headers):
            continue
          all_metrics.append(
              _row_datapoints_precision_recall_f1(
                  target_table,
                  prediction_table,
                  text_theta,
                  number_theta,
              )
          )
    p, r, f = max(all_metrics, key=lambda x: x[-1], default=(0, 0, 0))
    precision += p
    recall += r
    f1 += f
  return {
      "row_datapoints_precision": 100.0 * precision / len(targets),
      "row_datapoints_recall": 100.0 * recall / len(targets),
      "row_datapoints_f1": 100.0 * f1 / len(targets),
  }
