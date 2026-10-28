import os
import re

workload_list = ["cycleGAN-Pytorch", "ResNet50-Pytorch", "Transformers-Pytorch"]
batch_size_mapping = {
    "cycleGAN-Pytorch": [1, 2, 4],
    "ResNet50-Pytorch": [128, 256, 512],
    "Transformers-Pytorch": [512, 1024, 2048]
}
GPU_model_list = ["A16", "P100-SXM2-16GB", "V100-SXM2-16GB", "A100-PCIE-40GB"]


def rename_files(directory):
    # rename_files('./Estimator_data/Transformers-PyTorch')
    try:
        for file_name in os.listdir(directory):
            if "NVIDIA" in file_name:
                new_name = file_name.split("NVIDIA_", 1)[-1]
                old_path = os.path.join(directory, file_name)
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
            if "Tesla" in file_name:
                new_name = file_name.split("Tesla_", 1)[-1]
                old_path = os.path.join(directory, file_name)
                new_path = os.path.join(directory, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed: `{file_name}` -> `{new_name}`")
    except FileNotFoundError:
        print(f"Directory not found: `{directory}`")
    except Exception as e:
        print(f"Error occurred: {e}")


base_path = "./Estimator_data"
estimator_results = {}


def estimator_data_process():
    for workload in workload_list:
        estimator_results[workload] = {}
        batch_size_list = batch_size_mapping[workload]
        workload_path = os.path.join(base_path, workload)
        if os.path.exists(workload_path):
            for batch_size in batch_size_list:
                estimator_results[workload][batch_size] = {}
                for gpu_model in GPU_model_list:
                    matched_files = [
                        file for file in os.listdir(workload_path)
                        if (gpu_model + "_" + str(batch_size)) in file
                    ]

                    for file in matched_files:
                        file_path = os.path.join(workload_path, file)
                        iteration_data = {}
                        if file_path.endswith('.csv'):
                            continue
                        with open(file_path, "r") as f:
                            lines = f.readlines()
                            for i in range(0, len(lines), 2):
                                line1 = lines[i].strip()
                                line2 = lines[i + 1].strip()
                                line = line1 + line2
                                match1 = re.search(
                                    r"iteration (\d+) Time: ([\d.]+) second.*Throughput: ([\d.]+) tokens/second", line)
                                match2 = re.search(
                                    r"iteration (\d+) Time: ([\d.]+) second.*Throughput: ([\d.]+) samples/second", line)
                                if match1:
                                    iteration = int(match1.group(1))
                                    iteration_time = float(match1.group(2))
                                    iteration_throughput = float(match1.group(3))
                                    iteration_data[iteration] = (iteration_time, iteration_throughput)
                                if match2:
                                    iteration = int(match2.group(1))
                                    iteration_time = float(match2.group(2))
                                    iteration_throughput = float(match2.group(3))
                                    iteration_data[iteration] = (iteration_time, iteration_throughput)
                            max_throughput_group = max(iteration_data.items(), key=lambda x: x[1][1])
                            estimator_results[workload][batch_size][gpu_model] = {
                                "iteration_time": max_throughput_group[1][0],
                                "iteration_throughput": max_throughput_group[1][1]
                            }
    return estimator_results
