#!/usr/bin/env python3
"""Measure fixed-length Qwen3-1.7B autoregressive requests on CPU or CUDA.

Each timed iteration includes prompt prefill plus autoregressive decode for an
exact number of output tokens. EOS stopping and sampling are disabled so every
request has the same input and output lengths.
"""
import argparse
import csv
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path(__file__).parent / 'model')
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--input-tokens', type=int, default=64)
    parser.add_argument('--output-tokens', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=('fp16', 'bf16', 'fp32'), default='fp16')
    parser.add_argument('--prompt', default='Explain how GPUs accelerate machine learning in simple terms.')
    parser.add_argument('--output', type=Path, help='Optional CSV path for per-request latency.')
    return parser.parse_args()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values, q):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def main():
    args = parse_args()
    if args.input_tokens < 1 or args.output_tokens < 1:
        raise SystemExit('--input-tokens and --output-tokens must be positive')
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit('--iterations must be >= 1 and --warmup >= 0')
    if not args.model_dir.is_dir():
        raise SystemExit('Model directory not found: {}'.format(args.model_dir))

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
        attn_implementation='eager',
    ).to(device).eval()

    # Use an exact-length, fully attended prefix for every request.
    token_ids = tokenizer(args.prompt, add_special_tokens=True,
                          truncation=True, max_length=args.input_tokens)['input_ids']
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or tokenizer.bos_token_id or 0]
    token_ids = (token_ids * math.ceil(args.input_tokens / len(token_ids)))[:args.input_tokens]
    input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
    input_ids = input_ids.expand(args.batch_size, -1).contiguous()
    initial_attention_mask = torch.ones_like(input_ids)

    def run_request():
        # Include prefill and all decode forwards in this request's timed interval.
        output = model(
            input_ids=input_ids,
            attention_mask=initial_attention_mask,
            use_cache=True,
            cache_position=torch.arange(args.input_tokens, dtype=torch.long, device=device),
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Prefill logits produce token 1. Each forward below consumes the prior
        # generated token and produces the next one; never stop early on EOS.
        for decode_index in range(args.output_tokens - 1):
            context_length = args.input_tokens + decode_index
            attention_mask = torch.ones(
                (args.batch_size, context_length + 1), dtype=torch.long, device=device)
            cache_position = torch.tensor([context_length], dtype=torch.long, device=device)
            model_inputs = model.prepare_inputs_for_generation(
                next_token,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
            )
            output = model(**model_inputs, use_cache=True)
            past_key_values = output.past_key_values
            next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return next_token

    durations = []
    all_durations = []
    for step in range(args.warmup + args.iterations):
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            _ = run_request()
        sync(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            durations.append(elapsed)
            all_durations.append(elapsed)

    mean_s = statistics.fmean(durations)
    median_s = statistics.median(durations)
    gpu_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    throughput = args.batch_size * args.output_tokens / mean_s
    print('device: {} ({})'.format(device, gpu_name))
    print('model: {}'.format(args.model_dir))
    print('workload: full request (prefill + decode)')
    print('batch_size: {}, input_tokens: {}, output_tokens: {}, dtype: {}'.format(
        args.batch_size, args.input_tokens, args.output_tokens, args.dtype))
    print('iterations: {}, warmup: {}'.format(args.iterations, args.warmup))
    print('min_latency_ms: {:.3f}'.format(min(durations) * 1000.0))
    print('max_latency_ms: {:.3f}'.format(max(durations) * 1000.0))
    print('mean_latency_ms: {:.3f}'.format(mean_s * 1000.0))
    print('median_latency_ms: {:.3f}'.format(median_s * 1000.0))
    print('p95_latency_ms: {:.3f}'.format(percentile(durations, 0.95) * 1000.0))
    print('generated_tokens_per_second: {:.3f}'.format(throughput))

    if device.type == 'cuda':
        print('peak_allocated_gpu_memory_mib: {:.1f}'.format(
            torch.cuda.max_memory_allocated(device) / 1024**2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['iteration', 'latency_seconds', 'batch_size', 'input_tokens',
                             'output_tokens', 'generated_tokens', 'dtype', 'device', 'gpu'])
            for index, elapsed in enumerate(all_durations, 1):
                writer.writerow([index, '{:.9f}'.format(elapsed), args.batch_size,
                                 args.input_tokens, args.output_tokens,
                                 args.batch_size * args.output_tokens,
                                 args.dtype, str(device), gpu_name])
        print('csv: {}'.format(args.output))


if __name__ == '__main__':
    main()
