from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """ResNet generator from the original CycleGAN design."""
    def __init__(self, in_channels=3, out_channels=3, base=64, blocks=9):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_channels, base, 7),
                  nn.InstanceNorm2d(base), nn.ReLU(inplace=True)]
        channels = base
        for _ in range(2):
            layers += [nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                       nn.InstanceNorm2d(channels * 2), nn.ReLU(inplace=True)]
            channels *= 2
        layers += [ResidualBlock(channels) for _ in range(blocks)]
        for _ in range(2):
            layers += [nn.ConvTranspose2d(channels, channels // 2, 3, stride=2,
                                          padding=1, output_padding=1),
                       nn.InstanceNorm2d(channels // 2), nn.ReLU(inplace=True)]
            channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, out_channels, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN discriminator."""
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        channels = base
        for multiplier, stride in ((2, 2), (4, 2), (8, 1)):
            next_channels = base * multiplier
            layers += [nn.Conv2d(channels, next_channels, 4, stride=stride, padding=1),
                       nn.InstanceNorm2d(next_channels), nn.LeakyReLU(0.2, True)]
            channels = next_channels
        layers += [nn.Conv2d(channels, 1, 4, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CycleGAN(nn.Module):
    def __init__(self, base=64, blocks=9):
        super().__init__()
        self.G_AB = Generator(base=base, blocks=blocks)
        self.G_BA = Generator(base=base, blocks=blocks)
        self.D_A = Discriminator(base=base)
        self.D_B = Discriminator(base=base)


def set_requires_grad(modules, enabled):
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)


def train_iteration(model, optim_g, optim_d, real_a, real_b, lambda_cycle=10.0, lambda_identity=0.5):
    """Run one full generator + discriminator update; inputs are normalized to [-1, 1]."""
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    generators = (model.G_AB, model.G_BA)
    discriminators = (model.D_A, model.D_B)

    set_requires_grad(discriminators, False)
    optim_g.zero_grad(set_to_none=True)
    fake_b, fake_a = model.G_AB(real_a), model.G_BA(real_b)
    rec_a, rec_b = model.G_BA(fake_b), model.G_AB(fake_a)
    id_a, id_b = model.G_BA(real_a), model.G_AB(real_b)
    pred_fake_b, pred_fake_a = model.D_B(fake_b), model.D_A(fake_a)
    loss_g = (mse(pred_fake_b, torch.ones_like(pred_fake_b)) +
              mse(pred_fake_a, torch.ones_like(pred_fake_a)) +
              lambda_cycle * (l1(rec_a, real_a) + l1(rec_b, real_b)) +
              lambda_cycle * lambda_identity * (l1(id_a, real_a) + l1(id_b, real_b)))
    loss_g.backward()
    optim_g.step()

    set_requires_grad(discriminators, True)
    optim_d.zero_grad(set_to_none=True)
    pred_real_a, pred_fake_a = model.D_A(real_a), model.D_A(fake_a.detach())
    pred_real_b, pred_fake_b = model.D_B(real_b), model.D_B(fake_b.detach())
    loss_d_a = 0.5 * (mse(pred_real_a, torch.ones_like(pred_real_a)) +
                      mse(pred_fake_a, torch.zeros_like(pred_fake_a)))
    loss_d_b = 0.5 * (mse(pred_real_b, torch.ones_like(pred_real_b)) +
                      mse(pred_fake_b, torch.zeros_like(pred_fake_b)))
    loss_d = loss_d_a + loss_d_b
    loss_d.backward()
    optim_d.step()
    return loss_g.detach(), loss_d.detach()


def resolve_config(batch_size, device, image_size=None, base_channels=None):
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cudnn.deterministic = False
    use_p100_defaults = device.type == "cuda" and "P100" in torch.cuda.get_device_name(device) and image_size is None and base_channels is None
    size = {1:284,2:284,4:276}[batch_size] if use_p100_defaults else (image_size or 128)
    return {"image_size": size, "base_channels": 64 if use_p100_defaults else (base_channels or 32)}

def save_config(config, output):
    if output:
        path = Path(output).with_name("config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=1)
    p.add_argument('--image-size', type=int, default=None, help='Square input size divisible by 4; omitted uses the per-batch P100 preset or 128 on other devices.')
    p.add_argument('--base-channels', type=int, default=None, help='Feature width; omitted uses 64 for the P100 preset or 32 otherwise.')
    p.add_argument('--residual-blocks', type=int, default=9)
    p.add_argument('--iterations', type=int, default=10, help='Timed iterations after warmup.')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--output', type=Path, help='Optional CSV path for per-iteration timings.')
    return p.parse_args()


def main():
    args = parse_args()
    config = resolve_config(args.batch_size, torch.device(args.device), args.image_size, args.base_channels)
    args.image_size, args.base_channels = config["image_size"], config["base_channels"]
    save_config(config, args.output)
    print(json.dumps(config), flush=True)
    if args.image_size < 32 or args.image_size % 4:
        raise SystemExit('--image-size must be >= 32 and divisible by 4')
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit('--iterations must be >= 1 and --warmup >= 0')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
    model = CycleGAN(args.base_channels, args.residual_blocks).to(device)
    optim_g = torch.optim.Adam(list(model.G_AB.parameters()) + list(model.G_BA.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
    optim_d = torch.optim.Adam(list(model.D_A.parameters()) + list(model.D_B.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
    
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    real_a = torch.rand(shape, device=device).mul_(2).sub_(1)
    real_b = torch.rand(shape, device=device).mul_(2).sub_(1)
    times, last_losses = [], (None, None)
    total = args.warmup + args.iterations
    for step in range(total):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        last_losses = train_iteration(model, optim_g, optim_d, real_a, real_b)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            times.append(elapsed)
    mean_ms = 1000 * sum(times) / len(times)
    print(f'device: {device} ({torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"})')
    print(f'batch_size: {args.batch_size}, image_size: {args.image_size}x{args.image_size}, '
          f'base_channels: {args.base_channels}, residual_blocks: {args.residual_blocks}')
    print(f'iterations: {args.iterations}, warmup: {args.warmup}')
    print(f'mean_iteration_ms: {mean_ms:.3f}; mean_iteration_s: {mean_ms / 1000:.6f}')
    print(f'last_generator_loss: {last_losses[0].item():.4f}; last_discriminator_loss: {last_losses[1].item():.4f}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as f:
            f.write('iteration,seconds,batch_size,image_size,device\n')
            for i, seconds in enumerate(times, 1):
                f.write(f'{i},{seconds:.9f},{args.batch_size},{args.image_size},{device}\n')
        print(f'csv: {args.output}')


if __name__ == '__main__':
    main()
