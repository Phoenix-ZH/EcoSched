# EcoSched 使用说明

[English](README_EN.md) | 简体中文

本文中的命令默认在项目根目录（本 README 所在目录）执行。文件路径均相对于项目根目录；明确标注 `cd` 的章节除外。

## PerfProfiler

PerfProfiler 对应论文中的 workload 性能与能耗分析组件：采集迭代、算子和 CUDA kernel 信息，依据 GPU 硬件规格预测迭代时间，结合工作量与通信开销计算执行时间，并通过功率积分或能耗模型得到 ECon。

### 1. 常用入口

使用已有记录和参数生成当前实验结果：

```bash
python PerfProfiler/predict_iteration_time.py
python PerfProfiler/run_econ.py predict
python PerfProfiler/exe_time.py
```

| 命令 | 功能 | 默认输出 |
|---|---|---|
| `python PerfProfiler/predict_iteration_time.py` | 从源 GPU 的算子/kernel profile 预测目标 GPU 时间 | `PerfProfiler/Predict_result/predictions.json` |
| `python PerfProfiler/run_econ.py predict` | 计算固定配置网格的两条 ECon 响应曲线 | `PerfProfiler/Predict_result/econ_predictions.json`、`econ_results.json` |
| `python PerfProfiler/run_econ.py evaluate` | 汇总已保存 ECon 预测的相对偏差 | 更新 `PerfProfiler/Predict_result/econ_results.json` |
| `python PerfProfiler/exe_time.py` | 计算执行时间、重复记录和统计量 | `PerfProfiler/Predict_result/exe_time_results.json` |
| `python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0` | 执行 workload 并记录 NVML 功率和能耗 | `PerfProfiler/example_measurements/` 下带时间戳的 JSON |

前三个命令使用保存的数据和参数，不需要启动 GPU workload。`exe_time.py` 直接执行即可，没有 `predict` 或 `evaluate` 子命令。

绘图时进入 `PerfProfiler/Exp_plot`，在 Jupyter 或 IDE 中从头执行对应 notebook：

```bash
cd PerfProfiler/Exp_plot
jupyter notebook plot_ECon.ipynb
```

`plot_iteration_time.ipynb` 生成迭代/请求时间对比图，`plot_ECon.ipynb` 生成 ECon 与归一化 T_exe 图。PDF 使用相对文件名保存，因此以 `Exp_plot` 为 notebook 工作目录时，图文件保存在该目录。

### 2. 运行环境与 workload 范围

建议使用 Python 3.10 或更高版本。按所执行的组件准备依赖：

| 场景 | 依赖与条件 |
|---|---|
| 读取已有记录进行时间和 ECon 计算 | Python 标准库及完整的输入 JSON/CSV |
| workload 执行、迭代计时 | PyTorch；CUDA 执行需要匹配的 NVIDIA 驱动和 CUDA 版 PyTorch |
| 算子/kernel 采集 | CUDA GPU、PyTorch Profiler 和 CUPTI |
| NVML 能耗记录 | 可用的 NVIDIA NVML；优先使用 `pynvml` 接口，也支持原生 NVML 动态库 |
| Qwen | 支持 Qwen3 的 `transformers`、本地模型权重；多 GPU 自动分配还需要 `accelerate` |
| notebook 作图 | NumPy、Matplotlib、Jupyter/IDE notebook 支持；当前样式启用 LaTeX，并使用 `inconsolata`、`newtxtext` |

Qwen 模型卡给出的 Transformers 兼容起点为 4.51.0。两个批量采集入口会通过 `conda run -n qwen3` 调用 Qwen，因此使用它们采集 Qwen 时应准备名为 `qwen3` 的环境。直接执行 Qwen 脚本时，使用当前具备所需依赖的环境。

| Workload | 实现与一次工作单元 | 支持的 batch size | 当前 ECon / T_exe 图使用的 batch size |
|---|---|---|---:|
| ResNet50 | ResNet-50 训练；前向、损失、反向、优化器更新 | 128 / 256 / 512 | 256 |
| CycleGAN | 图像转换训练；生成器和判别器更新 | 1 / 2 / 4 | 2 |
| Transformers | Transformer encoder 分类训练 | 512 / 1024 / 2048 | 1024 |
| Qwen | Qwen3-1.7B 完整推理请求；prefill 与固定长度 decode | 1 / 2 / 4 | 2 |

前三类 workload 构造固定形状的输入张量用于计算性能采集。批量入口默认使用 ResNet50 的 32×32 图像、CycleGAN 的 128×128 图像和 Transformers 的 64-token 序列。Qwen 默认输入 64 tokens、输出 32 tokens、FP16；固定输出长度便于比较相同工作量。

### 3. 目录与文件作用

```text
EcoSched/
├── README.md
├── README_EN.md
└── PerfProfiler/
    ├── predict_iteration_time.py
    ├── specs_scaling.py
    ├── profile_scaling.py
    ├── gpu_specs.json
    ├── run_econ.py
    ├── econ_energy_model.py
    ├── econ_measurement.py
    ├── exe_time.py
    ├── Predict_result/           # 固定参数、时间与能耗结果
    ├── Exp_plot/                 # 当前绘图 notebook 与 PDF
    └── Workloads/
        ├── profile_workloads.py
        ├── profile_workload_operators.py
        ├── Profiler/torch_profiler.py
        ├── ResNet50/resnet50_benchmark.py
        ├── CycleGAN/cyclegan_benchmark.py
        ├── Transformers/transformers_benchmark.py
        ├── Qwen/qwen3_1_7b_benchmark.py
        ├── Qwen/model/
        └── profile_results/
            ├── benchmark/
            └── profiler/
```

#### 3.1 PerfProfiler 主目录

| 文件 | 作用与调用方式 |
|---|---|
| `predict_iteration_time.py` | 批量预测入口；发现源 GPU profile，加载硬件规格与效率参数，写入目标 GPU 的时间预测。 |
| `specs_scaling.py` | 读取 kernel/profile，计算算子类别、硬件并发能力、计算/访存缩放和请求时间组合；供预测入口调用。 |
| `profile_scaling.py` | 将 kernel 按实现类别聚合，通过保存的效率系数修正硬件缩放结果；供预测入口调用。 |
| `gpu_specs.json` | A16、P100、V100、A100 的 SM 数量、时钟、带宽、算力和资源限制等输入规格。 |
| `run_econ.py` | ECon 命令入口：`predict`、`evaluate`、`measure`、`record`、`operator-predict`。 |
| `econ_energy_model.py` | 能耗响应曲线和算子级功率缩放函数；计算 GPU 数量变化、通信和空闲间隙的能耗贡献。 |
| `econ_measurement.py` | NVML 后台采样、窗口边界插值、梯形积分及四种 workload 的直接执行入口。 |
| `exe_time.py` | 读取迭代记录与执行时间参数，计算 1/2/4 GPU 的 T_exe、计算/通信分量以及重复记录统计。 |

`specs_scaling.py`、`profile_scaling.py`、`econ_energy_model.py` 是库文件，通常由相应入口调用，无需单独执行。

#### 3.2 Workloads 下的脚本

| 文件 | 作用与输出 |
|---|---|
| `profile_workloads.py` | 遍历 workload 与 batch size，调用计时脚本，保存逐次 CSV 和总表。 |
| `profile_workload_operators.py` | 预热后采集一次工作单元的算子和 kernel；支持 workload、batch size 和 Qwen scope 筛选。 |
| `Profiler/torch_profiler.py` | 通用 `profile_iteration()` 接口；导出 trace、算子、kernel、类别汇总和元数据；也有独立 ResNet50 示例入口。 |
| `ResNet50/resnet50_benchmark.py` | 定义 ResNet50 并执行训练计时；`--output` 保存汇总 CSV，`--raw-output` 保存逐次 CSV，同时写 `config.json`。 |
| `CycleGAN/cyclegan_benchmark.py` | 定义生成器/判别器并执行训练计时；`--output` 保存逐次 CSV，同时写 `config.json`。 |
| `Transformers/transformers_benchmark.py` | 定义 encoder 分类模型并执行训练计时；`--output` 保存逐次 CSV，同时写 `config.json`。 |
| `Qwen/qwen3_1_7b_benchmark.py` | 从本地权重加载 Qwen，记录固定输入/输出长度的完整请求；`--output` 保存逐次请求延迟。 |
| `Qwen/model/verify_gpu.py` | 在 GPU 上加载本地 Qwen 并完成一次生成，打印版本、设备、输出文本和显存使用；不保存实验结果文件。 |

#### 3.3 Predict_result 下的数据文件

| 文件/目录 | 作用 |
|---|---|
| `parameters.json` | 迭代/请求时间预测使用的固定效率参数、特征类别及硬件配置。 |
| `predictions.json` | `predict_iteration_time.py` 输出：workload、batch、源/目标 GPU、`predicted_ms`、类别贡献与缩放信息。 |
| `econ_model.json` | ECon 测量侧和估计侧响应曲线的固定系数及 GPU 数量适用范围。 |
| `econ_predictions.json` | `run_econ.py predict` 输出的配置网格与两条能耗曲线数值。 |
| `econ_results.json` | 在能耗预测基础上增加逐配置相对偏差、workload 汇总和最大偏差配置；供当前 ECon 图读取。 |
| `exe_time_model.json` | T_exe 的计算/通信系数、运行波动参数和迭代记录根目录；Qwen 还包含硬件缩放配置。 |
| `exe_time_results.json` | T_exe 的秒数、归一化均值、标准差、计算/通信分量及每组重复记录。 |
| `live_measurements/` | 已保存的现场 GPU 能耗 trace；通常按 workload、batch、GPU 数量命名。 |

当前 ECon 参数覆盖四种 workload、三种目标 GPU 和 1/2/4 GPU，共 36 个点。T_exe 额外包含 A16 基准配置，共 48 个点，其中 36 个点用于目标 GPU 图。

#### 3.4 记录、绘图与辅助文件

| 路径或文件模式 | 内容 |
|---|---|
| `Workloads/profile_results/benchmark/**/iterations.csv` | 逐次计时，常用列为 `seconds` 或 Qwen 的 `latency_seconds`。 |
| `benchmark/summary.csv` | 各 workload/batch 配置的去极值计时汇总。 |
| `profiler/**/trace.json` | PyTorch/CUPTI 时间线，可用 Chrome tracing 或 Perfetto 查看。 |
| `profiler/**/operators.csv` | 算子名称、类别、调用次数、输入形状、CPU/CUDA 时间与内存信息。 |
| `profiler/**/kernels.csv` | CUDA kernel、关联算子、时长及 trace 提供的 launch/resource 信息。 |
| `profiler/**/paper_operator_summary.csv` | Conv、GEMM、Elemwise、Pool、Softmax、Other 的时间和占比汇总。 |
| `profiler/**/metadata.json` | GPU、软件环境、scope、输入配置、预热和基准时间等信息。 |
| `profiler/profile_status.csv` | 各配置的采集状态与错误信息。 |
| `Exp_plot/plot_iteration_time.ipynb` | 从 `predictions.json` 与 benchmark CSV 读取预测值和记录均值，生成四种 workload 的时间对比 PDF。 |
| `Exp_plot/plot_ECon.ipynb` | 从 `econ_results.json` 与 `exe_time_results.json` 绘制 ECon 柱状图和 T_exe 均值曲线；当前不展示 T_exe 误差棒。 |
| `Exp_plot/*_iteration.pdf`、`*_ECon.pdf` | 四种 workload 的图文件。CycleGAN 文件前缀为 `cycleGAN`。 |

### 4. 采集迭代时间和算子信息

#### 4.1 批量计时

```bash
python PerfProfiler/Workloads/profile_workloads.py --device cuda:0 --iterations 10 --warmup 2
python PerfProfiler/Workloads/profile_workloads.py --workloads ResNet50 Transformers --device cuda:0
python PerfProfiler/Workloads/profile_workloads.py --workloads Qwen --device cuda:0 --qwen-input-tokens 64 --qwen-output-tokens 32 --qwen-dtype fp16
```

默认运行全部 12 个 workload/batch 组合。计时次数至少为 7；汇总会删除最短和最长各一次，再报告最小值、最大值与平均值，原始逐次记录仍完整保存。`--output-root` 可以指定其他输出根目录。

新采集记录默认组织如下，GPU 文件夹采用设备名称并将空格等字符替换为下划线：

```text
PerfProfiler/Workloads/profile_results/
├── benchmark/
│   ├── summary.csv
│   └── <Workload>/batch_<size>/<GPU_model>/
│       ├── iterations.csv
│       └── config.json          # 训练类计时脚本写出
└── profiler/
    ├── profile_status.csv
    └── <Workload>/batch_<size>/<GPU_model>/
        ├── trace.json
        ├── operators.csv
        ├── kernels.csv
        ├── paper_operator_summary.csv
        └── metadata.json
```

已有记录还使用 `<Workload>/<GPU简称>/batch_<size>/` 布局。`predict_iteration_time.py` 兼容这两种 profile 布局；当前 `exe_time.py` 和迭代时间 notebook 使用已有 benchmark 布局。补充新 benchmark 后，应按其读取布局整理文件，再用于这些入口。

#### 4.2 单独执行 workload

下列示例均在项目根目录运行，并显式指定输入形状与输出文件：

```bash
python PerfProfiler/Workloads/ResNet50/resnet50_benchmark.py --batch-sizes 256 --image-size 32 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/resnet50_summary.csv --raw-output PerfProfiler/Workloads/profile_results/manual/resnet50_iterations.csv
python PerfProfiler/Workloads/CycleGAN/cyclegan_benchmark.py --batch-size 2 --image-size 128 --base-channels 32 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/cyclegan_iterations.csv
python PerfProfiler/Workloads/Transformers/transformers_benchmark.py --batch-size 1024 --seq-len 64 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/transformers_iterations.csv
conda run -n qwen3 python PerfProfiler/Workloads/Qwen/qwen3_1_7b_benchmark.py --batch-size 2 --input-tokens 64 --output-tokens 32 --dtype fp16 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/qwen_iterations.csv
```

独立训练脚本省略输入形状参数时，部分 GPU 会使用内置配置；跨 GPU 比较应显式指定相同的图像、序列长度和模型宽度。训练脚本的 `config.json` 保存在相应输出 CSV 所在目录，同一目录内后一次执行会覆盖该配置文件。

#### 4.3 算子与 kernel 采集

```bash
python PerfProfiler/Workloads/profile_workload_operators.py --device cuda:0 --warmup 2
python PerfProfiler/Workloads/profile_workload_operators.py --workload ResNet50 --batch-size 256 --device cuda:0 --baseline-iterations 10
python PerfProfiler/Workloads/profile_workload_operators.py --workload Qwen --batch-size 2 --device cuda:0 --qwen-profile-scope decode
```

训练 workload 记录一次训练迭代。Qwen 的 scope 决定一次采集的范围：

| `--qwen-profile-scope` | 工作单元 | 默认输出根目录 |
|---|---|---|
| `decode` | 提前准备 prompt/KV cache，采集一个 decode token | `Workloads/profile_results/profiler/` |
| `prefill` | 输入 prompt 的 prefill | `Workloads/profile_results/profiler_prefill/` |
| `request` | prefill 加固定长度 decode 的完整请求 | `Workloads/profile_results/profiler_request/` |

表内路径均位于 `PerfProfiler/` 下。`--output-root` 可覆盖输出根目录。`--baseline-iterations` 可额外采集不带 profiler 的时间并写入元数据。带 profiler 的单次时间包含采集开销，分析端到端耗时时使用对应 benchmark 或非插桩基准记录。

#### 4.4 通用 profiler 与 Nsight Compute

独立 ResNet50 profiler 示例：

```bash
python PerfProfiler/Workloads/Profiler/torch_profiler.py --batch-size 128 --image-size 32 --warmup 3 --output PerfProfiler/Workloads/profile_results/profiler_manual/resnet50_b128
```

其他 workload 可以调用 `profile_iteration(step_fn, output_dir, warmup, metadata, prepare_fn, baseline_iterations)`。`step_fn` 是不带参数的一次工作单元；`prepare_fn` 用于提前准备状态。预热会执行该函数并更新模型/优化器状态。

`kernels.csv` 从 CUPTI activity 中提取可用的 grid/block、CTA 数量、寄存器和 shared memory 信息；occupancy、blocks/warps per SM 属于资源估计。硬件计数器分析可使用 Nsight Compute，时间线分析可使用 Nsight Systems。算子分类保留 `Other`，避免遗漏未归类操作。

安装 `ncu` 并获得性能计数器权限后，可在项目根目录运行：

```bash
ncu --nvtx --nvtx-include "TARGET_ITERATION" --set full -o resnet50_ncu --target-processes all python PerfProfiler/Workloads/Profiler/torch_profiler.py --ncu-ready --batch-size 128 --image-size 32 --warmup 3
```

该模式只执行 NVTX 标记迭代，避免与 PyTorch CUPTI profiler 同时采集；报告写为根目录下的 `resnet50_ncu.ncu-rep`。具体计数器支持取决于 GPU、驱动和权限。

### 5. 预测迭代与请求时间

论文的基本流程是先采集源 GPU 上的算子时间，再根据计算能力、带宽和并发资源缩放到目标 GPU。当前实现进一步按 kernel 类别聚合，并应用 `parameters.json` 中保存的效率参数。

```bash
python PerfProfiler/predict_iteration_time.py
python PerfProfiler/predict_iteration_time.py --workload ResNet50 --batch-size 256 --targets P100 V100 A100
python PerfProfiler/predict_iteration_time.py --workload Qwen --batch-size 2
```

默认源 GPU 为 A16，目标 GPU 为 P100、V100、A100。输入包括 `Workloads/profile_results/profiler/`、`gpu_specs.json` 和 `Predict_result/parameters.json`。参数化入口使用 `paper` 模式和保存的适用范围；修改 `--variant` 或 token/prefill 覆盖项不会自动绕过参数范围检查。

结果统一写入 `Predict_result/predictions.json`，时间单位为毫秒。训练 workload 的 `predicted_ms` 对应一次训练迭代；Qwen 从 decode profile 组合请求时间，需结合记录中的 `scope`、prefill/decode 分量及 token 配置理解结果。Qwen 的单 decode 步与完整请求不能直接当作同一种时间比较。

每次运行都会覆盖 `predictions.json`，包括使用 workload/batch 筛选的运行。生成四种 workload 的完整图之前，应执行一次不带筛选参数的命令。

### 6. ECon 计算与现场记录

#### 6.1 固定配置网格与算子能耗模型

论文中的 ECon 对 GPU 功率随时间积分；对于同型号多 GPU，总能耗是各设备积分结果之和。算子级估计将预测的算子时间与平均功率相乘，再汇总计算、通信和空闲间隙能耗。`econ_energy_model.py` 实现相应功率缩放和 GPU 数量响应函数。

```bash
python PerfProfiler/run_econ.py predict
python PerfProfiler/run_econ.py evaluate
```

`predict` 读取 `econ_model.json`，计算配置网格并更新两个结果 JSON；`evaluate` 读取已保存的 `econ_predictions.json`，重新计算逐点偏差、workload 汇总和最大偏差配置。两者不执行 workload。

| 字段 | 含义 |
|---|---|
| `predicted_measured_econ` | 能耗响应模型计算的测量侧归一化值。 |
| `predicted_estimated_econ` | 能耗响应模型计算的估计侧归一化值。 |
| `estimated_vs_measured_error_pct` | 两侧绝对差除以测量侧数值，再乘 100。 |
| `econ_j` | 功率 trace 积分或算子能耗接口得到的焦耳数，具体口径见相应输出记录。 |
| `econ_kwh` | `econ_j / 3,600,000`。 |

当前图中的两侧值来自保存的响应模型；直接 GPU 功率采样由下述 `econ_measurement.py` 完成，记录在独立 JSON 中。保存新的现场 trace 不会自动替换响应模型或现有图数据。

#### 6.2 执行 workload 并 trace ECon

以下命令在具备 GPU 的机器上执行，将结果显式保存在 `Predict_result/live_measurements/`：

```bash
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu1.json
python PerfProfiler/econ_measurement.py --workload CycleGAN --batch-size 2 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/CycleGAN_batch2_gpu1.json
python PerfProfiler/econ_measurement.py --workload Transformers --batch-size 1024 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/Transformers_batch1024_gpu1.json
conda run -n qwen3 python PerfProfiler/econ_measurement.py --workload Qwen --batch-size 2 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/Qwen_batch2_gpu1.json
```

多 GPU 示例：

```bash
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0,1 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu2.json
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0,1,2,3 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu4.json
```

`--gpu-indices` 使用物理 NVML 设备编号；`--batch-size` 为每张 GPU 的 batch size，输出同时记录全局 batch size。训练类 workload 使用数据并行；Qwen 多 GPU 使用自动设备映射，并将实际映射写入结果。

默认预热 10 次、记录 10 次、采样间隔 20 ms，可通过 `--warmup`、`--iterations`、`--sampling-ms` 调整。采集过程同步设备后标记迭代窗口，使用窗口边界插值和梯形积分累计各 GPU 能耗。输出包含总焦耳数、逐迭代时间与能耗、平均功率、设备信息和功率样本。

省略 `--output` 时，当前代码会保存到 `PerfProfiler/example_measurements/<Workload>_batch_<size>_gpu<count>_<timestamp>.json`。若希望集中放在 `Predict_result`，使用上面的显式输出参数；同名文件会被覆盖。

#### 6.3 自定义 trace、进程记录与算子输入

`run_econ.py` 还提供以下接口，输入 JSON 的键对应被调用函数的参数：

| 命令 | 输入要点 | 默认输出 |
|---|---|---|
| `python PerfProfiler/run_econ.py measure --input trace_input.json` | `traces`：各 GPU 的 `[时间秒, 功率瓦]`；`windows`：`[开始秒, 结束秒, 步数]`；`serial_iterations`；可选 `baseline_j` | `Predict_result/trace_measurement.json` |
| `python PerfProfiler/run_econ.py record --input record_config.json` | `command` 参数列表、`gpu_indices`；工作目录、步数及采样设置等见 `record_subprocess()` | `Predict_result/econ_record_<timestamp>.json`；可用 `--output` 指定 |
| `python PerfProfiler/run_econ.py operator-predict --input operators.json` | `operators`、`source`、`target`、`gpu_count`、`serial_iterations`、`baseline_j`；可选通信和空闲参数 | `Predict_result/operator_prediction.json` |

以上输出路径均位于 `PerfProfiler/` 下。`measure` 输入时间戳应使用相同单调时钟且覆盖窗口边界；`serial_iterations` 表示固定本地 batch 口径下的单 GPU 等价工作量。`record` 覆盖被启动进程的整个生命周期，包含加载和初始化等阶段；逐迭代采集使用 `econ_measurement.py`。

### 7. T_exe 计算与结果统计

令 N 为单 GPU 等价迭代总数，g 为 GPU 数量，固定本地 batch 下的论文执行时间关系为：

```text
N = epochs × iterations_per_epoch
T_iter(g) = T_compute + T_comm × log2(g)
T_exe(g) = (N / g) × T_iter(g)
normalized_T_exe(g) = T_exe(g) / T_exe(A16, 1 GPU)
```

`exe_time.py` 读取 `exe_time_model.json` 和已有 benchmark 迭代记录，计算时间分量及重复记录。当前前三种 workload 使用固定的归一化计算/通信系数；Qwen 统一以 A16 请求记录为基准，通过配置的串行占比、算力和带宽比例得到目标 GPU 时间。实际 GPU trace 的并行实现和请求口径应与模型配置分别记录。

```bash
python PerfProfiler/exe_time.py
python PerfProfiler/exe_time.py --epochs 2 --iterations-per-epoch 100 --repeats 30 --seed 42
```

默认单 GPU 等价工作量为 100 次、每组 30 条重复记录。运行波动由配置参数和迭代记录共同决定；`std_exe_seconds` 与 `std_normalized_exe_time` 是这些记录的样本标准差。增加 GPU 数量时，通信阶段贡献进入相对波动模型。保存的 T_exe 是基于记录和参数推算的执行时间，现场采集窗口的总耗时使用对应 trace 的 `elapsed_s`。

`--model` 和 `--output` 可指定其他输入/输出文件。输出默认覆盖 `Predict_result/exe_time_results.json`，并保留每次记录的计算时间、通信时间和总时间。标准差数据继续保存；当前 `plot_ECon.ipynb` 只画均值曲线。

### 8. 绘图的数据依赖和输出

| Notebook | 输入 | 生成文件 |
|---|---|---|
| `Exp_plot/plot_iteration_time.ipynb` | `Predict_result/predictions.json`；`Workloads/profile_results/benchmark/` 下的逐次 CSV | `cycleGAN_iteration.pdf`、`ResNet50_iteration.pdf`、`Transformers_iteration.pdf`、`Qwen_iteration.pdf` |
| `Exp_plot/plot_ECon.ipynb` | `Predict_result/econ_results.json`；`Predict_result/exe_time_results.json` | `cycleGAN_ECon.pdf`、`ResNet50_ECon.pdf`、`Transformers_ECon.pdf`、`Qwen_ECon.pdf` |

在 `Exp_plot` 工作目录中执行，以上 PDF 就保存在 `PerfProfiler/Exp_plot/`。更新结果 JSON 后，从头运行 notebook；只执行后面的绘图单元格可能继续使用内存中的旧数据。保存 notebook 后，文件内会保留本次图像和文本输出。

## CostPlanner

CostPlanner 使用合成的 24 小时 workload 和区域资源数据运行四种 baseline 与 EcoSched 调度器。通常从项目根目录执行本节命令；路径均相对于 EcoSched 根目录。

### 1. 数据流程

CostPlanner 的数据流程是：读取 Alibaba PAI trace 中的任务及作业信息，结合 PerfProfiler 已保存的 workload 迭代时间记录和合成配置，生成 EcoSched 使用的 workload、GPU 执行时间、能耗参数、deadline、epoch/iteration 工作量、区域容量、电价及跨区域传输属性。调度器在这些输入上构造可行的 GPU 与区域放置，再输出任务级调度和按小时汇总的能耗及电费。

默认合成器读取 CostPlanner/cluster_trace/data 中的 pai_task_table.csv 和 pai_job_table.csv，并从 PerfProfiler/Workloads/profile_results/benchmark 中读取 benchmark 迭代记录。默认配置位于 CostPlanner/datasets/synthesize_config.json。合成器使用已有记录，不会为了生成数据而启动 GPU workload。

CostPlanner 的合成、调度和敏感性实验脚本使用 Python 标准库及已保存的 CSV/JSON 数据，不需要启动 GPU workload。绘图需要 NumPy、Matplotlib 和 Jupyter。

### 2. 核心文件

| 文件 | 作用 |
|---|---|
| CostPlanner/datasets/synthesize.py | 从 trace、PerfProfiler benchmark 记录和配置构造 24 小时 EcoSched 输入数据。 |
| CostPlanner/datasets/synthesize_config.json | 描述每小时 workload 数量、workload 模板和 batch size、epoch 工作量、GPU 功率、区域及传输相关的生成参数。 |
| CostPlanner/scheduling_common.py | 读取共享 workload/区域输入，计算 deadline 可行的执行时间、GPU 能耗与区域电费，维护调度容量并写出通用结果。 |
| CostPlanner/baseline_selection.py | 为 baseline 枚举满足 deadline 的 GPU 类型、GPU 数量、区域和开始时间，并加载各算法的策略参数。 |
| CostPlanner/schedule_results/baseline_parameters.json | 保存 LCF、LEF、TanGo 和 MAST 使用的可配置策略权重及算法参数。 |
| CostPlanner/LowestCostFirst.py | LCF 在可行放置中结合区域电价和预计电费进行选择。 |
| CostPlanner/LowestEConFirst.py | LEF 在可行放置中结合执行能耗和预计电费进行选择。 |
| CostPlanner/TanGo.py | TanGo 将 workload 的 GPU 执行时间与 GPU 供给情况共同纳入区域放置评分，并考虑预计电费。 |
| CostPlanner/MAST.py | MAST 根据 GPU 供给规模评估 workload 的 GPU 时间，并结合预计电费选择可行放置。 |
| CostPlanner/CostPlannerFlow.py | EcoSched 的主要入口。它按到达批次建立 workload 到区域/GPU 池的候选分配图，用最小费用流求解分配，再通过资源日历验证 GPU 并行预留与 deadline；遇到资源冲突时重新规划。跨区域传输时间会影响任务可开始时间。 |
| CostPlanner/costplanner_flow.py | 实现 CostPlannerFlow 使用的最小费用流图和求解器。 |
| CostPlanner/GeoCollector.py | 提供基于时间区间的 GPU 容量日历，用于检查多 GPU 预留、可用时段及冲突。 |
| CostPlanner/network_sensitivity_experiments.py | 在第 14 小时的 6000-workload 场景下，分别改变输入数据量和跨区域带宽，运行五种调度器并统计 GPU 与网络能耗/电费。 |
| CostPlanner/Exp_plot/plot_CostPlanner.ipynb | 从 schedule_results 中读取小时汇总与传输敏感性结果，绘制 CostPlanner 实验图。 |

调度主流程使用 CostPlanner/datasets/workloads_24h/workloads.csv 和 CostPlanner/datasets/workloads_24h/df_regions.csv。前者包含各 workload 的到达时间、deadline、总迭代量、epochs、不同 GPU 的执行时间与功率，以及跨区域传输所需的来源区域、数据量和带宽；后者包含各区域的 GPU 容量和电价。生成的 ML_workloads.csv、prefix_views.csv 和 overhead_* 文件用于补充分析场景。

CostPlannerFlow 的常规 ECon/ECost 按 GPU 执行能耗和任务所在区域电价计算。network_sensitivity_experiments.py 另外根据跨区域传输时长和网络功率假设核算网络能耗，并在传输实验结果中分别记录 GPU、网络及合计费用。

### 3. 运行方式

先生成或更新合成数据：

~~~bash
python CostPlanner/datasets/synthesize.py
~~~

然后可分别运行调度器：

~~~bash
python CostPlanner/LowestCostFirst.py
python CostPlanner/LowestEConFirst.py
python CostPlanner/TanGo.py
python CostPlanner/MAST.py
python CostPlanner/CostPlannerFlow.py
~~~

单独运行 EcoSched 时执行 CostPlannerFlow.py。运行网络敏感性实验：

~~~bash
python CostPlanner/network_sensitivity_experiments.py
~~~

默认情况下，敏感性脚本使用第 14 小时的 6000 个 workload；数据量实验采用 1× 至 5×，固定带宽为 50 Gbps；带宽实验采用 10、20、30、40 Gbps，并以 1× 数据量为基准。

查看图表时，从 CostPlanner/Exp_plot 目录启动 notebook：

~~~bash
cd CostPlanner/Exp_plot
jupyter notebook plot_CostPlanner.ipynb
~~~

### 4. 输入、结果与图表

| 路径 | 内容 |
|---|---|
| CostPlanner/datasets/workloads_24h/ | 合成后的 24 小时 workload、区域容量及补充分析数据。 |
| CostPlanner/schedule_results/{算法}_results.csv | 按小时汇总输入/完成任务数、ECon、ECost、等待时间和相关下界。 |
| CostPlanner/schedule_results/{算法}_schedule.csv | 逐 workload 的调度结果，包括 GPU、区域、GPU 数量、开始/完成时间、ECon 和电费。 |
| CostPlanner/schedule_results/transfer_results/transfer_summary.csv | 各数据量/带宽情景下五种算法的完成情况、GPU 与网络能耗及费用汇总。 |
| CostPlanner/schedule_results/transfer_results/transfer_hourly_results.csv | 传输敏感性实验的逐 workload 结果及情景信息。 |
| CostPlanner/Exp_plot/ | 绘图 notebook 和生成的 PDF 图。 |

重新运行合成器会更新 datasets/workloads_24h 中同名数据文件；重新运行某个调度器会覆盖其同名 results/schedule CSV。绘图前应先运行与图表对应的调度或敏感性实验，以便 notebook 读取最新 CSV。
