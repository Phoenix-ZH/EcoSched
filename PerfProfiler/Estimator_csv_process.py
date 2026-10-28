import os
import pandas as pd
import re
import json


def read_csv_files(directory, operation_list):
    filtered_data = {}
    op_importance = {}
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_csv(file_path)
            total_cuda_time = df['self_cuda_time_total'].sum()
            for op in operation_list:
                filtered_rows = df[df['name'].str.contains(op, case=False, na=False)]
                filtered_data = filtered_rows[['name', 'self_cuda_time_total']]
                filtered_data_cuda_time = filtered_data['self_cuda_time_total'].sum()
                if (op not in op_importance):
                    op_importance[op] = 0
                op_importance[op] += filtered_data_cuda_time / total_cuda_time
    return op_importance


'''
Conv, GEMM (mm), Linear, Pool, Atc (relu, sigmoid), Softmax, Norm, Elemwise (add, mul, sub, div), Emb,Dropout, Reduce (sum, max, mean)
'''
operation_dict = {"Conv": ["Conv"], "GEMM": ["mm"], "linear": ["linear"], "Pool": ["Pool"], "Atc": ["relu", "sigmoid"],
                  "Softmax": ["Softmax"], "Norm": ["Norm"], "Elemwise": ["add", "mul", "sub", "div"], "Emb": ["Emb"],
                  "Dropout": ["Dropout"], "Reduce": ["sum", "max", "mean"]}

workload_list = ["cycleGAN-Pytorch", "ResNet50-Pytorch", "Transformers-Pytorch"]
base_path = './Estimator_data/'
operation_importance_dict = {}
for workload in workload_list:
    workload_path = os.path.join(base_path, workload)
    directory = os.path.join(base_path, workload)
    for operation_name, operation_list in operation_dict.items():
        op_importance = read_csv_files(directory, operation_list)
        operation_importance_sum = 0
        for op, importance in op_importance.items():
            op_importance[op] = op_importance[op] / 4  # 4个GPU
            operation_importance_sum += op_importance[op]
        if operation_name not in operation_importance_dict:
            operation_importance_dict[operation_name] = 0
        operation_importance_dict[operation_name] += operation_importance_sum

for operation_name, operation_importance in operation_importance_dict.items():
    operation_importance_dict[operation_name] = operation_importance_dict[operation_name] / 3.0 * 100
    operation_importance_dict[operation_name] = {
        "importance": max(round(operation_importance_dict[operation_name], 1), 0.1)}

print(operation_importance_dict)

operation_importance_dict_path = './Estimator_data/operation_importance_dict.json'
with open(operation_importance_dict_path, 'w') as f:
    json.dump(operation_importance_dict, f, indent=4)
