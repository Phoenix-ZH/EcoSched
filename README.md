# EcoSched: Usage Guide

[English](README.md) | [简体中文](README_zh.md)

Unless a section explicitly uses `cd`, run all commands from the project root, where this README is located. File paths are relative to that root.

## PerfProfiler

PerfProfiler implements the workload performance and energy analysis component described in the paper. It collects iteration, operator, and CUDA kernel information; predicts iteration times from GPU specifications; computes execution time from the workload budget and communication overhead; and obtains ECon through power integration or energy models.

### 1. Common entry points

Generate the current experiment results from the saved records and parameters:

```bash
python PerfProfiler/predict_iteration_time.py
python PerfProfiler/run_econ.py predict
python PerfProfiler/exe_time.py
```

| Command | Purpose | Default output |
|---|---|---|
| `python PerfProfiler/predict_iteration_time.py` | Predict target GPU times from source GPU operator/kernel profiles | `PerfProfiler/Predict_result/predictions.json` |
| `python PerfProfiler/run_econ.py predict` | Compute both ECon response curves for the configured grid | `PerfProfiler/Predict_result/econ_predictions.json` and `econ_results.json` |
| `python PerfProfiler/run_econ.py evaluate` | Summarize relative differences in saved ECon predictions | Updates `PerfProfiler/Predict_result/econ_results.json` |
| `python PerfProfiler/exe_time.py` | Compute execution times, repeated records, and statistics | `PerfProfiler/Predict_result/exe_time_results.json` |
| `python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0` | Execute a workload and record NVML power and energy | A timestamped JSON file under `PerfProfiler/example_measurements/` |

The three commands in the initial code block consume saved records and parameters without launching a GPU workload. Run `exe_time.py` directly; it has no `predict` or `evaluate` subcommand.

For plotting, enter `PerfProfiler/Exp_plot` and execute the corresponding notebook from the beginning in Jupyter or an IDE:

```bash
cd PerfProfiler/Exp_plot
jupyter notebook plot_ECon.ipynb
```

`plot_iteration_time.ipynb` produces iteration/request time comparisons. `plot_ECon.ipynb` plots ECon and normalized T_exe. PDFs use relative filenames, so they are saved in `Exp_plot` when that directory is the notebook working directory.

### 2. Environment and workload scope

Python 3.10 or later is recommended. Prepare dependencies for the components you intend to run:

| Operation | Dependencies and requirements |
|---|---|
| Time and ECon computation from saved records | Python standard library and complete input JSON/CSV files |
| Workload execution and iteration timing | PyTorch; CUDA execution requires a compatible NVIDIA driver and CUDA-enabled PyTorch |
| Operator/kernel collection | CUDA GPU, PyTorch Profiler, and CUPTI |
| NVML energy recording | Available NVIDIA NVML; the recorder prefers the `pynvml` interface and also supports the native NVML library |
| Qwen | A `transformers` version supporting Qwen3 and local model weights; automatic multi-GPU placement also requires `accelerate` |
| Notebook plotting | NumPy, Matplotlib, and Jupyter/IDE notebook support; the current style enables LaTeX and uses `inconsolata` and `newtxtext` |

The Qwen model card lists Transformers 4.51.0 as the compatibility starting point. Both batch collection entry points invoke Qwen through `conda run -n qwen3`, so prepare an environment named `qwen3` when using them for Qwen. Standalone Qwen scripts use the active environment and its installed dependencies.

| Workload | Implementation and work unit | Supported batch sizes | Batch size in current ECon / T_exe figures |
|---|---|---|---:|
| ResNet50 | ResNet-50 training: forward pass, loss, backward pass, and optimizer update | 128 / 256 / 512 | 256 |
| CycleGAN | Image translation training: generator and discriminator updates | 1 / 2 / 4 | 2 |
| Transformers | Transformer encoder classification training | 512 / 1024 / 2048 | 1024 |
| Qwen | Complete Qwen3-1.7B inference request: prefill and fixed-length decode | 1 / 2 / 4 | 2 |

The first three workloads construct input tensors with fixed shapes for compute performance collection. The batch entry points default to 32×32 images for ResNet50, 128×128 images for CycleGAN, and 64-token sequences for Transformers. Qwen defaults to 64 input tokens, 32 output tokens, and FP16. A fixed output length makes the request work comparable.

### 3. Directory and file roles

```text
EcoSched/
├── README.md
├── README_zh.md
└── PerfProfiler/
    ├── predict_iteration_time.py
    ├── specs_scaling.py
    ├── profile_scaling.py
    ├── gpu_specs.json
    ├── run_econ.py
    ├── econ_energy_model.py
    ├── econ_measurement.py
    ├── exe_time.py
    ├── Predict_result/           # Fixed parameters, timing and energy results
    ├── Exp_plot/                 # Current plotting notebooks and PDFs
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

#### 3.1 PerfProfiler root files

| File | Purpose and usage |
|---|---|
| `predict_iteration_time.py` | Batch prediction entry point. Discovers source GPU profiles, loads hardware specifications and efficiency parameters, and writes target GPU time predictions. |
| `specs_scaling.py` | Reads kernel/profile records and computes operator categories, hardware concurrency, compute/memory scaling, and request time composition. Called by the prediction entry point. |
| `profile_scaling.py` | Groups kernels by implementation family and applies saved efficiency coefficients to hardware scaling results. Called by the prediction entry point. |
| `gpu_specs.json` | Input specifications for A16, P100, V100, and A100, including SM count, clocks, bandwidth, compute throughput, and resource limits. |
| `run_econ.py` | ECon command entry point: `predict`, `evaluate`, `measure`, `record`, and `operator-predict`. |
| `econ_energy_model.py` | Energy response curves and operator power scaling functions, including GPU-count, communication, and idle-gap contributions. |
| `econ_measurement.py` | Background NVML sampling, window boundary interpolation, trapezoidal integration, and direct execution of the four workloads. |
| `exe_time.py` | Reads iteration records and execution-time parameters to compute T_exe, compute/communication components, and repeated-record statistics for 1/2/4 GPUs. |

`specs_scaling.py`, `profile_scaling.py`, and `econ_energy_model.py` are library modules. Normally they are called by their entry points rather than executed separately.

#### 3.2 Scripts under Workloads

| File | Purpose and output |
|---|---|
| `profile_workloads.py` | Iterates over workloads and batch sizes, invokes timing scripts, and saves per-iteration CSV files, logs, and a summary table. |
| `profile_workload_operators.py` | Collects operators and kernels for one work unit after warmup. Supports workload, batch-size, and Qwen scope filters. |
| `Profiler/torch_profiler.py` | Provides `profile_iteration()` and exports traces, operators, kernels, category summaries, and metadata. Also includes a standalone ResNet50 example. |
| `ResNet50/resnet50_benchmark.py` | Defines ResNet50 and measures training iterations. `--output` saves a summary CSV; `--raw-output` saves per-iteration CSV data. Also writes `config.json`. |
| `CycleGAN/cyclegan_benchmark.py` | Defines the generators/discriminators and measures training iterations. `--output` saves a per-iteration CSV. Also writes `config.json`. |
| `Transformers/transformers_benchmark.py` | Defines an encoder classifier and measures training iterations. `--output` saves a per-iteration CSV. Also writes `config.json`. |
| `Qwen/qwen3_1_7b_benchmark.py` | Loads local Qwen weights and measures complete requests with fixed input/output lengths. `--output` saves per-request latency. |
| `Qwen/model/verify_gpu.py` | Loads local Qwen on a GPU, completes one generation, and prints versions, device information, generated text, and memory usage. Does not write experiment result files. |

#### 3.3 Data files under Predict_result

| File/directory | Purpose |
|---|---|
| `parameters.json` | Fixed efficiency parameters, feature families, and hardware configuration for iteration/request time prediction. |
| `predictions.json` | Output of `predict_iteration_time.py`: workload, batch size, source/target GPU, `predicted_ms`, family contributions, and scaling details. |
| `econ_model.json` | Fixed coefficients and GPU-count domains for the measurement-side and estimation-side ECon response curves. |
| `econ_predictions.json` | Configuration grid and both energy curves produced by `run_econ.py predict`. |
| `econ_results.json` | Adds per-configuration relative differences, workload summaries, and the worst configuration to the energy predictions. Read by the current ECon figure. |
| `exe_time_model.json` | T_exe compute/communication coefficients, run-variability parameters, and iteration-record root. Also includes Qwen hardware scaling settings. |
| `exe_time_results.json` | T_exe in seconds, normalized means, standard deviations, compute/communication components, and repeated records for each configuration. |
| `live_measurements/` | Saved GPU energy traces, usually named by workload, batch size, and GPU count. |

The current ECon parameters cover four workloads, three target GPUs, and GPU counts of 1/2/4, giving 36 points. T_exe also includes A16 baseline configurations, giving 48 points in total; 36 are used in the target GPU figures.

#### 3.4 Records, plots, and supporting files

| Path or file pattern | Content |
|---|---|
| `Workloads/profile_results/benchmark/**/iterations.csv` | Per-iteration timing, usually in `seconds`, or `latency_seconds` for Qwen. |
| `benchmark/summary.csv` | Timing summaries with the shortest and longest samples removed for each workload/batch configuration. |
| `profiler/**/trace.json` | PyTorch/CUPTI timeline, viewable in Chrome tracing or Perfetto. |
| `profiler/**/operators.csv` | Operator names, categories, call counts, input shapes, CPU/CUDA times, and memory information. |
| `profiler/**/kernels.csv` | CUDA kernels, associated operators, durations, and launch/resource fields available in the trace. |
| `profiler/**/paper_operator_summary.csv` | Time and share summaries for Conv, GEMM, Elemwise, Pool, Softmax, and Other. |
| `profiler/**/metadata.json` | GPU, software environment, scope, input configuration, warmup, and baseline timing details. |
| `profiler/profile_status.csv` | Collection status and error details for each configuration. |
| `Exp_plot/plot_iteration_time.ipynb` | Reads predictions and benchmark means to produce time comparison PDFs for the four workloads. |
| `Exp_plot/plot_ECon.ipynb` | Reads `econ_results.json` and `exe_time_results.json` to plot ECon bars and T_exe mean curves. T_exe error bars are currently hidden. |
| `Exp_plot/*_iteration.pdf`, `*_ECon.pdf` | Figure files for the four workloads. The CycleGAN filename prefix is `cycleGAN`. |

### 4. Collecting iteration times and operator information

#### 4.1 Batch timing

```bash
python PerfProfiler/Workloads/profile_workloads.py --device cuda:0 --iterations 10 --warmup 2
python PerfProfiler/Workloads/profile_workloads.py --workloads ResNet50 Transformers --device cuda:0
python PerfProfiler/Workloads/profile_workloads.py --workloads Qwen --device cuda:0 --qwen-input-tokens 64 --qwen-output-tokens 32 --qwen-dtype fp16
```

By default, the script runs all 12 workload/batch combinations. At least seven timed iterations are required. The summary removes the shortest and longest samples before reporting minimum, maximum, and mean times; the original per-iteration records are preserved. Use `--output-root` to choose another output root.

New records use the following layout. GPU directory names are based on the device name, with spaces and similar characters replaced by underscores:

```text
PerfProfiler/Workloads/profile_results/
├── benchmark/
│   ├── summary.csv
│   └── <Workload>/batch_<size>/<GPU_model>/
│       ├── iterations.csv
│       └── config.json          # Written by training timing scripts
└── profiler/
    ├── profile_status.csv
    └── <Workload>/batch_<size>/<GPU_model>/
        ├── trace.json
        ├── operators.csv
        ├── kernels.csv
        ├── paper_operator_summary.csv
        └── metadata.json
```

Existing records also use `<Workload>/<GPU_short_name>/batch_<size>/`. `predict_iteration_time.py` accepts both profile layouts. The current `exe_time.py` and iteration-time notebook use the existing benchmark layout. Organize new benchmark files into the layout expected by these readers before using them there.

#### 4.2 Running individual workloads

Run these examples from the project root. They explicitly specify input shapes and output files:

```bash
python PerfProfiler/Workloads/ResNet50/resnet50_benchmark.py --batch-sizes 256 --image-size 32 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/resnet50_summary.csv --raw-output PerfProfiler/Workloads/profile_results/manual/resnet50_iterations.csv
python PerfProfiler/Workloads/CycleGAN/cyclegan_benchmark.py --batch-size 2 --image-size 128 --base-channels 32 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/cyclegan_iterations.csv
python PerfProfiler/Workloads/Transformers/transformers_benchmark.py --batch-size 1024 --seq-len 64 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/transformers_iterations.csv
conda run -n qwen3 python PerfProfiler/Workloads/Qwen/qwen3_1_7b_benchmark.py --batch-size 2 --input-tokens 64 --output-tokens 32 --dtype fp16 --device cuda:0 --output PerfProfiler/Workloads/profile_results/manual/qwen_iterations.csv
```

When input-shape arguments are omitted, standalone training scripts use built-in settings for some GPUs. For comparisons across GPUs, explicitly select the same image size, sequence length, and model width. Training scripts write `config.json` beside the output CSV; subsequent runs in the same directory overwrite that configuration file.

#### 4.3 Operator and kernel collection

```bash
python PerfProfiler/Workloads/profile_workload_operators.py --device cuda:0 --warmup 2
python PerfProfiler/Workloads/profile_workload_operators.py --workload ResNet50 --batch-size 256 --device cuda:0 --baseline-iterations 10
python PerfProfiler/Workloads/profile_workload_operators.py --workload Qwen --batch-size 2 --device cuda:0 --qwen-profile-scope decode
```

Training workloads record one training iteration. For Qwen, the scope determines the recorded work unit:

| `--qwen-profile-scope` | Work unit | Default output root |
|---|---|---|
| `decode` | Prepare the prompt/KV cache first, then collect one decode token | `Workloads/profile_results/profiler/` |
| `prefill` | Prefill for the input prompt | `Workloads/profile_results/profiler_prefill/` |
| `request` | Complete request: prefill and fixed-length decode | `Workloads/profile_results/profiler_request/` |

Paths in this table are under `PerfProfiler/`. `--output-root` overrides the output root. `--baseline-iterations` collects additional timings without profiler instrumentation and stores them in metadata. A profiled iteration includes collection overhead; use matching benchmark or uninstrumented baseline records when analyzing end-to-end latency.

#### 4.4 Shared profiler and Nsight Compute

Standalone ResNet50 profiling example:

```bash
python PerfProfiler/Workloads/Profiler/torch_profiler.py --batch-size 128 --image-size 32 --warmup 3 --output PerfProfiler/Workloads/profile_results/profiler_manual/resnet50_b128
```

Other workloads can call `profile_iteration(step_fn, output_dir, warmup, metadata, prepare_fn, baseline_iterations)`. `step_fn` is a zero-argument callable for one work unit. `prepare_fn` prepares state before that unit. Warmup invokes the callable and updates model/optimizer state.

`kernels.csv` extracts available grid/block dimensions, CTA counts, registers, and shared-memory information from CUPTI activity. Occupancy and blocks/warps per SM are resource estimates. Use Nsight Compute for hardware-counter analysis and Nsight Systems for timeline analysis. Operator summaries retain the `Other` category to include unclassified work.

After installing `ncu` and obtaining performance-counter access, run this command from the project root:

```bash
ncu --nvtx --nvtx-include "TARGET_ITERATION" --set full -o resnet50_ncu --target-processes all python PerfProfiler/Workloads/Profiler/torch_profiler.py --ncu-ready --batch-size 128 --image-size 32 --warmup 3
```

This mode executes the NVTX-marked iteration without running the PyTorch CUPTI profiler concurrently. It writes `resnet50_ncu.ncu-rep` in the project root. Counter availability depends on the GPU, driver, and permissions.

### 5. Predicting iteration and request times

The paper's basic workflow collects operator times on a source GPU and scales them to a target GPU using compute capability, bandwidth, and concurrency resources. The implementation also aggregates kernel families and applies efficiency parameters saved in `parameters.json`.

```bash
python PerfProfiler/predict_iteration_time.py
python PerfProfiler/predict_iteration_time.py --workload ResNet50 --batch-size 256 --targets P100 V100 A100
python PerfProfiler/predict_iteration_time.py --workload Qwen --batch-size 2
```

The default source GPU is A16; target GPUs are P100, V100, and A100. Inputs include `Workloads/profile_results/profiler/`, `gpu_specs.json`, and `Predict_result/parameters.json`. The parameterized entry point uses `paper` mode and the saved parameter scope. Changing `--variant` or token/prefill overrides does not bypass parameter-scope validation.

Results are written to `Predict_result/predictions.json`, with times in milliseconds. For training workloads, `predicted_ms` represents one training iteration. Qwen composes request time from a decode profile; interpret its output together with `scope`, prefill/decode components, and token settings. A single decode step and a complete request represent different timing scopes.

Every run overwrites `predictions.json`, including runs filtered by workload or batch size. Before producing complete figures for all four workloads, run the command once without filters.

### 6. ECon computation and GPU recording

#### 6.1 Fixed configuration grid and operator energy model

The paper defines ECon by integrating GPU power over time. For multiple GPUs of the same model, total energy is the sum of the device integrals. Operator-level estimation multiplies predicted operator times by average power, then combines compute, communication, and idle-gap contributions. `econ_energy_model.py` implements the corresponding power scaling and GPU-count response functions.

```bash
python PerfProfiler/run_econ.py predict
python PerfProfiler/run_econ.py evaluate
```

`predict` reads `econ_model.json`, computes the configuration grid, and updates both result JSON files. `evaluate` reads the saved `econ_predictions.json` and recomputes per-point differences, workload summaries, and the worst configuration. Neither command executes the workload.

| Field | Meaning |
|---|---|
| `predicted_measured_econ` | Normalized measurement-side value computed by the energy response model. |
| `predicted_estimated_econ` | Normalized estimation-side value computed by the energy response model. |
| `estimated_vs_measured_error_pct` | Absolute difference between the two values, divided by the measurement-side value, multiplied by 100. |
| `econ_j` | Energy in joules obtained through power-trace integration or the operator energy interface; the output record identifies its scope. |
| `econ_kwh` | `econ_j / 3,600,000`. |

Both curves in the current figure come from saved response models. Direct GPU power sampling is performed by `econ_measurement.py`, described below, and stored in separate JSON records. Saving a new GPU trace does not automatically replace the response model or existing figure data.

#### 6.2 Executing workloads and tracing ECon

Run the following commands on a machine with GPUs. They explicitly save results under `Predict_result/live_measurements/`:

```bash
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu1.json
python PerfProfiler/econ_measurement.py --workload CycleGAN --batch-size 2 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/CycleGAN_batch2_gpu1.json
python PerfProfiler/econ_measurement.py --workload Transformers --batch-size 1024 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/Transformers_batch1024_gpu1.json
conda run -n qwen3 python PerfProfiler/econ_measurement.py --workload Qwen --batch-size 2 --gpu-indices 0 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/Qwen_batch2_gpu1.json
```

Multi-GPU examples:

```bash
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0,1 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu2.json
python PerfProfiler/econ_measurement.py --workload ResNet50 --batch-size 256 --gpu-indices 0,1,2,3 --iterations 10 --output PerfProfiler/Predict_result/live_measurements/ResNet50_batch256_gpu4.json
```

`--gpu-indices` specifies physical NVML device indices. `--batch-size` is the per-GPU batch size, and the output also records the global batch size. Training workloads use data parallelism. Multi-GPU Qwen uses automatic device placement and records the actual device map.

Defaults are 10 warmup iterations, 10 recorded iterations, and a 20 ms sampling interval. Adjust them with `--warmup`, `--iterations`, and `--sampling-ms`. The recorder synchronizes devices when marking iteration windows, interpolates window boundaries, and integrates each GPU's power with the trapezoidal rule. Outputs include total joules, per-iteration durations and energies, average power, device details, and power samples.

Without `--output`, the current implementation saves to `PerfProfiler/example_measurements/<Workload>_batch_<size>_gpu<count>_<timestamp>.json`. To keep results under `Predict_result`, use an explicit output path as above. Reusing a filename overwrites it.

#### 6.3 Custom traces, process recording, and operator inputs

`run_econ.py` also exposes the following interfaces. Input JSON keys correspond to the called function's parameters:

| Command | Required input and options | Default output |
|---|---|---|
| `python PerfProfiler/run_econ.py measure --input trace_input.json` | `traces`: per-GPU `[time_seconds, power_watts]`; `windows`: `[start_seconds, end_seconds, steps]`; `serial_iterations`; optional `baseline_j` | `Predict_result/trace_measurement.json` |
| `python PerfProfiler/run_econ.py record --input record_config.json` | A `command` argument list and `gpu_indices`; see `record_subprocess()` for working-directory, step-count, and sampling options | `Predict_result/econ_record_<timestamp>.json`; override with `--output` |
| `python PerfProfiler/run_econ.py operator-predict --input operators.json` | `operators`, `source`, `target`, `gpu_count`, `serial_iterations`, `baseline_j`; optional communication and idle parameters | `Predict_result/operator_prediction.json` |

These output paths are under `PerfProfiler/`. Timestamps supplied to `measure` must share a monotonic clock and bracket the window boundaries. `serial_iterations` expresses the single-GPU-equivalent workload with a fixed local batch size. `record` covers the launched process's lifetime, including loading and initialization. Use `econ_measurement.py` for iteration-window recording.

### 7. T_exe computation and statistics

Let N be the total number of single-GPU-equivalent iterations and g the GPU count. With a fixed local batch size, the paper's execution-time relationship is:

```text
N = epochs × iterations_per_epoch
T_iter(g) = T_compute + T_comm × log2(g)
T_exe(g) = (N / g) × T_iter(g)
normalized_T_exe(g) = T_exe(g) / T_exe(A16, 1 GPU)
```

`exe_time.py` reads `exe_time_model.json` and existing benchmark iteration records to compute timing components and repeated records. The first three workloads use fixed normalized compute/communication coefficients. Qwen uses A16 request records as its baseline and derives target GPU times from configured serial fractions, compute throughput, and bandwidth ratios. Record the parallel execution method and request scope of GPU traces separately from the model configuration.

```bash
python PerfProfiler/exe_time.py
python PerfProfiler/exe_time.py --epochs 2 --iterations-per-epoch 100 --repeats 30 --seed 42
```

Defaults are a single-GPU-equivalent budget of 100 iterations and 30 repeated records per configuration. Run variability is determined by the configured parameters and iteration records. `std_exe_seconds` and `std_normalized_exe_time` are the sample standard deviations of these records. Communication stages contribute to relative variability as GPU count increases. Saved T_exe values are execution-time estimates derived from records and parameters; the total duration of recorded GPU windows is available as `elapsed_s` in the corresponding trace.

Use `--model` and `--output` to select other input/output files. By default, the script overwrites `Predict_result/exe_time_results.json`, retaining compute, communication, and total times for each repeated record. Standard-deviation data remains saved; `plot_ECon.ipynb` currently displays mean curves only.

### 8. Plot inputs and outputs

| Notebook | Inputs | Generated files |
|---|---|---|
| `Exp_plot/plot_iteration_time.ipynb` | `Predict_result/predictions.json`; per-iteration CSV files under `Workloads/profile_results/benchmark/` | `cycleGAN_iteration.pdf`, `ResNet50_iteration.pdf`, `Transformers_iteration.pdf`, `Qwen_iteration.pdf` |
| `Exp_plot/plot_ECon.ipynb` | `Predict_result/econ_results.json`; `Predict_result/exe_time_results.json` | `cycleGAN_ECon.pdf`, `ResNet50_ECon.pdf`, `Transformers_ECon.pdf`, `Qwen_ECon.pdf` |

Run the notebooks with `Exp_plot` as the working directory to save these PDFs under `PerfProfiler/Exp_plot/`. After updating result JSON files, execute the notebook from the beginning. Running only later plotting cells can reuse old data in memory. Saving the notebook preserves its current figures and text output.


## CostPlanner

CostPlanner runs four baselines and the EcoSched scheduler on a synthesized 24-hour workload and regional resource data. Run the commands in this section from the project root unless a command explicitly changes directory. Paths are relative to the EcoSched root.

### 1. Data flow

The data flow reads task and job information from the Alibaba PAI trace, combines it with saved workload iteration timings from PerfProfiler and the synthesis configuration, and generates EcoSched inputs. These inputs describe workloads, GPU execution times and power, deadlines, epoch/iteration budgets, regional capacities and electricity prices, and inter-region transfer attributes. The schedulers use the inputs to construct feasible GPU and regional placements, then write task-level schedules and hourly energy and cost summaries.

By default, the synthesizer reads pai_task_table.csv and pai_job_table.csv under CostPlanner/cluster_trace/data and benchmark iteration records under PerfProfiler/Workloads/profile_results/benchmark. Its default configuration is CostPlanner/datasets/synthesize_config.json. It uses the saved records and does not launch GPU workloads to generate the dataset.

The CostPlanner synthesis, scheduling, and sensitivity scripts use the Python standard library and saved CSV/JSON data; they do not launch GPU workloads. Plotting requires NumPy, Matplotlib, and Jupyter.

### 2. Core files

| File | Purpose |
|---|---|
| CostPlanner/datasets/synthesize.py | Builds the 24-hour EcoSched input dataset from the trace, PerfProfiler benchmark records, and configuration. |
| CostPlanner/datasets/synthesize_config.json | Defines hourly workload counts, workload templates and batch sizes, epoch work, GPU power, regions, and transfer-related generation parameters. |
| CostPlanner/scheduling_common.py | Loads the shared workload and regional inputs, computes deadline-feasible execution times, GPU energy and regional electricity bills, maintains scheduling capacity, and writes common results. |
| CostPlanner/baseline_selection.py | Enumerates baseline placements that meet deadlines across GPU type, GPU count, region, and start time, and loads each algorithm's policy parameters. |
| CostPlanner/schedule_results/baseline_parameters.json | Stores the configurable policy weights and algorithm parameters used by LCF, LEF, TanGo, and MAST. |
| CostPlanner/LowestCostFirst.py | LCF selects feasible placements using regional tariffs and estimated electricity bills. |
| CostPlanner/LowestEConFirst.py | LEF selects feasible placements using execution energy and estimated electricity bills. |
| CostPlanner/TanGo.py | TanGo scores regional placements using workload GPU execution time and GPU supply, together with estimated electricity bills. |
| CostPlanner/MAST.py | MAST evaluates workload GPU time relative to GPU supply and combines that score with estimated electricity bills. |
| CostPlanner/CostPlannerFlow.py | Main EcoSched entry point. It builds a workload-to-region/GPU-pool candidate graph for each arrival batch, solves the assignment with minimum-cost flow, and validates GPU gang reservations and deadlines with a resource calendar; it replans when reservations conflict. Inter-region transfer time affects the earliest task start. |
| CostPlanner/costplanner_flow.py | Implements the minimum-cost flow graph and solver used by CostPlannerFlow. |
| CostPlanner/GeoCollector.py | Provides a time-interval GPU capacity calendar for checking multi-GPU reservations, available intervals, and conflicts. |
| CostPlanner/network_sensitivity_experiments.py | Varies input data volume and inter-region bandwidth for the hour-14 scenario with 6,000 workloads, runs all five schedulers, and reports GPU and network energy/cost. |
| CostPlanner/Exp_plot/plot_CostPlanner.ipynb | Reads hourly summaries and transfer-sensitivity results from schedule_results and creates the CostPlanner experiment plots. |

The main schedulers read CostPlanner/datasets/workloads_24h/workloads.csv and CostPlanner/datasets/workloads_24h/df_regions.csv. The workload file contains arrival times, deadlines, total iterations, epochs, per-GPU execution time and power, plus origin region, data volume, and bandwidth for inter-region transfers. The region file contains GPU capacity and electricity prices for each region. ML_workloads.csv, prefix_views.csv, and overhead_* files are generated for supplementary analyses.

For standard CostPlannerFlow results, ECon/ECost are calculated from GPU execution energy and the task's regional electricity price. network_sensitivity_experiments.py separately accounts for network energy from transfer duration and an assumed network power, and records GPU, network, and combined costs in the transfer experiment results.

### 3. Running the code

Generate or refresh the synthesized dataset:

~~~bash
python CostPlanner/datasets/synthesize.py
~~~

Run the schedulers individually:

~~~bash
python CostPlanner/LowestCostFirst.py
python CostPlanner/LowestEConFirst.py
python CostPlanner/TanGo.py
python CostPlanner/MAST.py
python CostPlanner/CostPlannerFlow.py
~~~

Run CostPlannerFlow.py by itself to execute EcoSched. Run the network sensitivity experiments with:

~~~bash
python CostPlanner/network_sensitivity_experiments.py
~~~

By default, the sensitivity script uses 6,000 workloads from hour 14. Its data-volume experiment uses 1× through 5× at a fixed 50 Gbps; its bandwidth experiment uses 10, 20, 30, and 40 Gbps at the 1× data volume. 

To view the plots, start the notebook from CostPlanner/Exp_plot:

~~~bash
cd CostPlanner/Exp_plot
jupyter notebook plot_CostPlanner.ipynb
~~~

### 4. Inputs, results, and plots

| Path | Contents |
|---|---|
| CostPlanner/datasets/workloads_24h/ | Synthesized 24-hour workloads, regional capacity, and supplementary analysis data. |
| CostPlanner/schedule_results/{algorithm}_results.csv | Hourly input/completed task counts, ECon, ECost, wait times, and related lower bounds. |
| CostPlanner/schedule_results/{algorithm}_schedule.csv | Per-workload placement, including GPU, region, GPU count, start/finish times, ECon, and electricity cost. |
| CostPlanner/schedule_results/transfer_results/transfer_summary.csv | Completion, GPU/network energy, and cost summaries for the five algorithms across data-volume and bandwidth scenarios. |
| CostPlanner/schedule_results/transfer_results/transfer_hourly_results.csv | Per-workload results and scenario details for the transfer sensitivity experiments. |
| CostPlanner/Exp_plot/ | Plotting notebook and generated PDF figures. |

Running the synthesizer updates files with matching names under datasets/workloads_24h. Running a scheduler overwrites its corresponding results and schedule CSV files. Run the scheduler or sensitivity experiment required by a plot before executing the notebook so it reads the latest CSV data.
