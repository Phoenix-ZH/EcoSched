#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet50(nn.Module):
    """Standard ResNet-50 (3, 4, 6, 3 bottleneck blocks), randomly initialized."""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, planes, blocks, stride=1):
        outplanes = planes * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes),
            )
        layers = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = outplanes
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_config(batch_size, device, image_size=None):
    use_p100_defaults = device.type == "cuda" and "P100" in torch.cuda.get_device_name(device) and image_size is None
    height, width = {128: (32, 68), 256: (32, 55), 512: (30, 42)}[batch_size] if use_p100_defaults else (image_size or 32, image_size or 32)
    torch.backends.cudnn.benchmark = device.type == "cuda" and not (use_p100_defaults and batch_size == 128)
    torch.backends.cudnn.deterministic = use_p100_defaults and batch_size == 128
    return {"image_height": height, "image_width": width,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}

def save_config(config, output):
    if output:
        path = Path(output).with_name("config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--image-size", type=int, default=None,
                        help="Input H=W (224 is standard ImageNet ResNet-50; use 32 for CIFAR-sized input)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=10,
                        help="Timed training iterations; report mean and median")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--output", default="benchmark_results.csv")
    parser.add_argument("--raw-output", default=None, help="Optional CSV with one row per timed iteration")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    if args.amp and device.type != "cuda":
        raise SystemExit("--amp requires CUDA")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"GPU/device: {gpu_name}; torch={torch.__version__}; input=3x{args.image_size}x{args.image_size}")
    print("Timing includes forward, loss, backward, and optimizer.step; CUDA is synchronized.")

    import csv
    rows = []
    raw_rows = []
    for batch_size in args.batch_sizes:
        try:
            config = resolve_config(batch_size, device, args.image_size)
            save_config(config, args.raw_output or args.output)
            print(json.dumps(config), flush=True)
            model = ResNet50(num_classes=1000).to(device).train()
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
            x = torch.randn(batch_size, 3, config["image_height"], config["image_width"], device=device)
            y = torch.randint(0, 1000, (batch_size,), device=device)

            def step():
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            for _ in range(args.warmup):
                step()
            sync(device)
            times = []
            for _ in range(args.iterations):
                sync(device)
                start = time.perf_counter()
                step()
                sync(device)
                times.append(time.perf_counter() - start)
            raw_rows.extend({"iteration": i + 1, "seconds": f"{elapsed:.9f}",
                             "batch_size": batch_size, "device": str(device),
                             "gpu": gpu_name}
                            for i, elapsed in enumerate(times))
            times_ms = sorted(t * 1000 for t in times)
            mean_ms = sum(times_ms) / len(times_ms)
            median_ms = times_ms[len(times_ms) // 2]
            throughput = batch_size / (mean_ms / 1000)
            print(f"batch={batch_size}: mean={mean_ms:.3f} ms, median={median_ms:.3f} ms, throughput={throughput:.1f} samples/s")
            rows.append({"device": gpu_name, "torch": torch.__version__, "image_size": str((config["image_height"], config["image_width"])),
                         "batch_size": batch_size, "iterations": args.iterations,
                         "mean_ms": f"{mean_ms:.3f}", "median_ms": f"{median_ms:.3f}",
                         "samples_per_second": f"{throughput:.1f}", "amp": args.amp})
            del model, optimizer, x, y
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"batch={batch_size}: CUDA out of memory; skipped. Try --amp or --image-size 32.", file=sys.stderr)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output}")
    else:
        raise SystemExit("No batch size completed successfully.")
    if args.raw_output and raw_rows:
        with open(args.raw_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=raw_rows[0].keys())
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"Saved raw iterations: {args.raw_output}")


if __name__ == "__main__":
    main()
