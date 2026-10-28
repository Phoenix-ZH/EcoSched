#!/usr/bin/env python3
"""Capture ATen operators and CUDA kernels for one or all training workloads."""
import argparse
import csv
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BATCHES = {"ResNet50": (128, 256, 512), "CycleGAN": (1, 2, 4),
           "Transformers": (512, 1024, 2048), "Qwen": (1, 2, 4)}
DEFAULT_ORDER = ("ResNet50", "CycleGAN", "Transformers", "Qwen")


def gpu_identity(device):
    name = torch.cuda.get_device_name(device)
    folder = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name, folder or "Unknown_GPU"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_step(workload, batch_size, device, image_size, seq_len):
    if workload == "ResNet50":
        mod = load_module("resnet50_benchmark", ROOT / "ResNet50" / "resnet50_benchmark.py")
        model = mod.ResNet50(num_classes=1000).to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)
        labels = torch.randint(0, 1000, (batch_size,), device=device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        return step, None

    if workload == "CycleGAN":
        mod = load_module("cyclegan_benchmark", ROOT / "CycleGAN" / "cyclegan_benchmark.py")
        model = mod.CycleGAN(base=32, blocks=9).to(device)
        optim_g = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
                                   lr=2e-4, betas=(0.5, 0.999))
        optim_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()),
                                   lr=2e-4, betas=(0.5, 0.999))
        shape = (batch_size, 3, image_size, image_size)
        real_a = torch.rand(shape, device=device).mul_(2).sub_(1)
        real_b = torch.rand(shape, device=device).mul_(2).sub_(1)
        return lambda: mod.train_iteration(model, optim_g, optim_d, real_a, real_b), None

    mod = load_module("transformers_benchmark", ROOT / "Transformers" / "transformers_benchmark.py")
    model = mod.TransformerClassifier(vocab_size=4096, seq_len=seq_len,
                                      d_model=128, nhead=4, layers=2,
                                      num_classes=16, dropout=0.1).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    tokens = torch.randint(4096, (batch_size, seq_len), device=device)
    labels = torch.randint(16, (batch_size,), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(tokens), labels)
        loss.backward()
        optimizer.step()
    return step, None


def make_qwen_step(batch_size, device, model_dir, input_tokens, dtype_name,
                   profile_scope="decode", output_tokens=32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=dtype_map[dtype_name], local_files_only=True,
        attn_implementation="eager").to(device).eval()
    prompt = "Explain how GPUs accelerate machine learning in simple terms."
    token_ids = tokenizer(prompt, add_special_tokens=True, truncation=True,
                          max_length=input_tokens)["input_ids"]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or tokenizer.bos_token_id or 0]
    token_ids = (token_ids * math.ceil(input_tokens / len(token_ids)))[:input_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = input_ids.expand(batch_size, -1).contiguous()
    attention_mask = torch.ones_like(input_ids)
    decode_ids = input_ids[:, -1:]
    decode_attention_mask = torch.ones((batch_size, input_tokens + 1), dtype=torch.long, device=device)
    cache_position = torch.tensor([input_tokens], dtype=torch.long, device=device)
    state = {"past_key_values": None}

    def prefill():
        return model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=True, cache_position=torch.arange(input_tokens, device=device))

    def prefill_step():
        with torch.inference_mode():
            return prefill()

    def request_step():
        # Match the uninstrumented benchmark scope: prefill produces token 1,
        # followed by output_tokens - 1 decode forwards and greedy selection.
        with torch.inference_mode():
            output = prefill()
            past = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for decode_index in range(output_tokens - 1):
                context = input_tokens + decode_index
                mask = torch.ones((batch_size, context + 1), dtype=torch.long, device=device)
                position = torch.tensor([context], dtype=torch.long, device=device)
                model_inputs = model.prepare_inputs_for_generation(
                    next_token, past_key_values=past, attention_mask=mask, cache_position=position)
                output = model(**model_inputs, use_cache=True)
                past = output.past_key_values
                next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            return next_token

    if profile_scope == "request":
        return request_step, None
    if profile_scope == "prefill":
        return prefill_step, None
    if profile_scope != "decode":
        raise ValueError("Unknown Qwen profiling scope: {}".format(profile_scope))

    def prepare():
        with torch.inference_mode():
            state["past_key_values"] = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                cache_position=torch.arange(input_tokens, device=device)).past_key_values

    def step():
        with torch.inference_mode():
            model_inputs = model.prepare_inputs_for_generation(
                decode_ids, past_key_values=state["past_key_values"],
                attention_mask=decode_attention_mask, cache_position=cache_position)
            model(**model_inputs, use_cache=True)
    return step, prepare


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", choices=DEFAULT_ORDER,
                        help="Default: all workloads in ResNet50, CycleGAN, Transformers, Qwen order.")
    parser.add_argument("--workload", choices=DEFAULT_ORDER,
                        help="Optional single-workload shorthand.")
    parser.add_argument("--batch-size", type=int,
                        help="Optional single-batch filter; requires --workload.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--baseline-iterations", type=int, default=0,
                        help="Optional uninstrumented source timings stored in profile metadata.")
    parser.add_argument("--resnet-image-size", type=int, default=32)
    parser.add_argument("--cycle-image-size", type=int, default=128)
    parser.add_argument("--transformer-seq-len", type=int, default=64)
    parser.add_argument("--qwen-model-dir", type=Path, default=ROOT / "Qwen" / "model")
    parser.add_argument("--qwen-input-tokens", type=int, default=64)
    parser.add_argument("--qwen-output-tokens", type=int, default=32)
    parser.add_argument("--qwen-profile-scope", choices=("decode", "prefill", "request"), default="decode")
    parser.add_argument("--qwen-dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--output-root", type=Path,
                        help="Defaults to profiler, profiler_prefill, or profiler_request by scope.")
    parser.add_argument("--qwen-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_root is None:
        suffix = "" if args.qwen_profile_scope == "decode" else "_" + args.qwen_profile_scope
        args.output_root = ROOT / "profile_results" / ("profiler" + suffix)
    if args.workload and args.workloads:
        parser.error("use either --workload or --workloads, not both")
    if args.batch_size is not None and not args.workload:
        parser.error("--batch-size requires --workload")
    workloads = [args.workload] if args.workload else (args.workloads or list(DEFAULT_ORDER))
    if args.qwen_worker and (workloads != ["Qwen"] or args.batch_size is None):
        parser.error("internal Qwen worker requires --workload Qwen and --batch-size")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.baseline_iterations < 0:
        parser.error("--baseline-iterations must be >= 0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("operator and CUDA-kernel profiling requires an available CUDA device")
    torch.cuda.set_device(device)
    if args.cycle_image_size < 32 or args.cycle_image_size % 4:
        parser.error("--cycle-image-size must be >= 32 and divisible by 4")
    if args.qwen_input_tokens < 1 or args.qwen_output_tokens < 1:
        parser.error("Qwen input/output token counts must be positive")
    return args, workloads, device


def profile_qwen_in_child(args, batch_size, device, output_dir):
    command = ["conda", "run", "-n", "qwen3", "python", str(Path(__file__).resolve()),
               "--workload", "Qwen", "--batch-size", str(batch_size), "--device", str(device),
               "--warmup", str(args.warmup), "--qwen-model-dir", str(args.qwen_model_dir),
               "--qwen-input-tokens", str(args.qwen_input_tokens), "--qwen-dtype", args.qwen_dtype,
               "--qwen-output-tokens", str(args.qwen_output_tokens),
               "--qwen-profile-scope", args.qwen_profile_scope,
               "--baseline-iterations", str(args.baseline_iterations),
               "--output-root", str(args.output_root), "--qwen-worker"]
    result = subprocess.run(command, check=False)
    if result.returncode:
        return "failed", "qwen3 profiler subprocess exit code {}".format(result.returncode)
    return "ok", ""


def main():
    args, workloads, device = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    gpu_model, gpu_folder = gpu_identity(device)
    profiler_path = ROOT / "Profiler"
    sys.path.insert(0, str(profiler_path))
    from torch_profiler import profile_iteration

    status_rows = []
    for workload in workloads:
        batch_sizes = (args.batch_size,) if args.batch_size is not None else BATCHES[workload]
        for batch_size in batch_sizes:
            if batch_size not in BATCHES[workload]:
                raise SystemExit("batch size for {} must be one of {}".format(workload, BATCHES[workload]))
            output_dir = output_root / workload / "batch_{}".format(batch_size) / gpu_folder
            output_dir.mkdir(parents=True, exist_ok=True)
            print("[{} batch={}] profiling one iteration".format(workload, batch_size), flush=True)
            if workload == "Qwen" and not args.qwen_worker:
                try:
                    status, error = profile_qwen_in_child(args, batch_size, device, output_dir)
                except OSError as exc:
                    status, error = "failed", str(exc)
            else:
                try:
                    if workload == "Qwen":
                        step, prepare_fn = make_qwen_step(
                            batch_size, device, args.qwen_model_dir, args.qwen_input_tokens, args.qwen_dtype,
                            args.qwen_profile_scope, args.qwen_output_tokens)
                    else:
                        image_size = args.resnet_image_size if workload == "ResNet50" else args.cycle_image_size
                        step, prepare_fn = make_step(workload, batch_size, device,
                                                     image_size, args.transformer_seq_len)
                    profile_iteration(
                        step, output_dir=str(output_dir), warmup=args.warmup,
                        metadata={"workload": workload, "gpu_model": gpu_model, "batch_size": batch_size,
                                  "device": str(device), "resnet_image_size": args.resnet_image_size,
                                  "cycle_image_size": args.cycle_image_size,
                                  "transformer_seq_len": args.transformer_seq_len,
                                  "qwen_input_tokens": args.qwen_input_tokens,
                                  "qwen_output_tokens": args.qwen_output_tokens,
                                  "qwen_profile_scope": args.qwen_profile_scope,
                                  "qwen_dtype": args.qwen_dtype},
                        prepare_fn=prepare_fn, baseline_iterations=args.baseline_iterations)
                    status, error = "ok", ""
                    del step, prepare_fn
                    torch.cuda.empty_cache()
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    status, error = "cuda_oom", str(exc).splitlines()[0]
                    print("  CUDA OOM; continuing with next batch.", file=sys.stderr, flush=True)
                    if "step" in locals():
                        del step
                    if "prepare_fn" in locals():
                        del prepare_fn
                    torch.cuda.empty_cache()
            status_rows.append({"workload": workload, "gpu_model": gpu_model, "batch_size": batch_size,
                                "status": status, "output_dir": str(output_dir), "error": error})

    if not args.qwen_worker:
        status_path = output_root / "profile_status.csv"
        with status_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["workload", "gpu_model", "batch_size", "status", "output_dir", "error"])
            writer.writeheader()
            writer.writerows(status_rows)
        print("Profile status: {}".format(status_path))
    return 1 if any(row["status"] != "ok" for row in status_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
