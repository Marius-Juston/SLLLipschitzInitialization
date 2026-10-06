"""Recompute initialization activation statistics in float64 (training stays float32)."""
import json
from pathlib import Path
import torch
from .runner import data, loader, seed_all, freeze_for_evaluation, write_json
from .models import build, ColumnLinear


def run():
    records={}
    base=json.loads(Path('configs/revision/cifar10-sll-d15-zero-s0.json').read_text())
    _,validation,_,inputs,classes=data(base,Path('data'))
    x,_=next(iter(loader(validation,128,workers=0)))
    for bias in ('zero','corrected'):
        for seed in range(3):
            cfg=dict(base,bias=bias,seed=seed);seed_all(seed)
            model=build(cfg,inputs,classes).double().to('cuda:1').eval()
            freeze_for_evaluation(model);rows={};y=x.double().to('cuda:1')
            with torch.no_grad():
                for i,layer in enumerate(model.net):
                    y=layer(y)
                    if isinstance(layer,ColumnLinear):
                        rows[f'net.{i}']=dict(second_moment=y.square().mean().item(),
                                             input_variance=y.var(dim=0,unbiased=False).mean().item())
            records[f'cifar10-sll-d15-{bias}-s{seed}']=rows
    write_json(Path('artifacts/campaign/precision-diagnostics.json'),dict(dtype='float64',
        purpose='Post hoc numerical-precision check of the same float32-drawn initial parameters; no retraining.',
        batch_size=128,records=records))
    print('Saved float64 initialization check',flush=True)


if __name__=='__main__':run()
