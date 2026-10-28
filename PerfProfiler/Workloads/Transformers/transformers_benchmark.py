import argparse
import json
import csv
import time
from pathlib import Path

import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """Token embedding + learned positions + Transformer encoder + classifier."""
    def __init__(self, vocab_size, seq_len, d_model, nhead, layers, num_classes, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)
        self.register_buffer('positions', torch.arange(seq_len), persistent=False)

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding(self.positions)
        x = self.encoder(x)
        return self.classifier(self.norm(x.mean(dim=1)))


def resolve_config(batch_size, device, seq_len=None):
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.backends.cudnn.deterministic = False
    use_p100_defaults = device.type == "cuda" and "P100" in torch.cuda.get_device_name(device) and seq_len is None
    return {"seq_len": {512:196,1024:135,2048:111}[batch_size] if use_p100_defaults else (seq_len or 64)}

def save_config(config, output):
    if output:
        path = Path(output).with_name("config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=2))

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size', type=int, choices=(512, 1024, 2048), default=512)
    p.add_argument('--seq-len', type=int, default=None)
    p.add_argument('--vocab-size', type=int, default=4096)
    p.add_argument('--num-classes', type=int, default=16)
    p.add_argument('--d-model', type=int, default=128)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--iterations', type=int, default=10, help='Timed iterations after warmup.')
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--output', type=Path, help='Optional CSV path for per-iteration timings.')
    return p.parse_args()


def main():
    args = parse_args()
    config = resolve_config(args.batch_size, torch.device(args.device), args.seq_len)
    args.seq_len = config["seq_len"]
    save_config(config, args.output)
    print(json.dumps(config), flush=True)
    if args.seq_len < 1 or args.layers < 1 or args.d_model < 1:
        raise SystemExit('--seq-len, --layers and --d-model must be positive')
    if args.d_model % args.nhead:
        raise SystemExit('--d-model must be divisible by --nhead')
    if args.iterations < 1 or args.warmup < 0:
        raise SystemExit('--iterations must be >= 1 and --warmup >= 0')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA requested but unavailable')
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model = TransformerClassifier(args.vocab_size, args.seq_len, args.d_model,
                                  args.nhead, args.layers, args.num_classes,
                                  args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # Keep synthetic batches on device so timing measures model work, not input I/O.
    tokens = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
    labels = torch.randint(args.num_classes, (args.batch_size,), device=device)
    durations = []
    model.train()
    for step in range(args.warmup + args.iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            durations.append(elapsed)
    mean_ms = 1000.0 * sum(durations) / len(durations)
    gpu_name = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'
    print('device: {} ({})'.format(device, gpu_name))
    print('batch_size: {}, seq_len: {}, d_model: {}, nhead: {}, layers: {}'.format(
        args.batch_size, args.seq_len, args.d_model, args.nhead, args.layers))
    print('iterations: {}, warmup: {}'.format(args.iterations, args.warmup))
    print('last_loss: {:.5f}'.format(loss.item()))
    print('mean_iteration_ms: {:.3f}; mean_iteration_s: {:.6f}'.format(mean_ms, mean_ms / 1000.0))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iteration', 'seconds', 'batch_size', 'seq_len', 'd_model',
                             'nhead', 'layers', 'device', 'gpu'])
            for i, seconds in enumerate(durations, 1):
                writer.writerow([i, '{:.9f}'.format(seconds), args.batch_size, args.seq_len,
                                 args.d_model, args.nhead, args.layers, str(device), gpu_name])
        print('csv: {}'.format(args.output))


if __name__ == '__main__':
    main()
