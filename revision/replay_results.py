"""Independent checkpoint replay and bounded PGD geometry audit (no retraining)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .models import build
from .runner import data, loader, seed_all, freeze_for_evaluation, pgd, write_json


class GeometryChecked(nn.Module):
    def __init__(self, model, original, eps):
        super().__init__();self.model=model;self.original=original;self.eps=eps
        self.max_norm=0.;self.calls=0
    def forward(self, x):
        with torch.no_grad():
            norm=(x-self.original).flatten(1).norm(dim=1).max().item()
            if not torch.isfinite(x).all() or x.min() < -1e-6 or x.max() > 1+1e-6 or norm > self.eps+2e-5:
                raise ValueError('PGD evaluated an invalid perturbation')
            self.max_norm=max(self.max_norm,norm);self.calls+=1
        return self.model(x)


def replay(root, output, device):
    seed_all(901)
    jobs=json.loads((root/'campaign.json').read_text())['jobs']
    datasets={}; records={}
    for name in jobs:
        p=root/name;cfg=json.loads((p/'config.json').read_text())
        ev=json.loads((p/'evaluation.json').read_text())
        if cfg['dataset'] not in datasets:
            _,_,test,inputs,classes=data(cfg,Path('data'));datasets[cfg['dataset']]=(test,inputs,classes)
        test,inputs,classes=datasets[cfg['dataset']]
        seed_all(cfg['seed']);model=build(cfg,inputs,classes).to(device)
        parameter_count=sum(v.numel() for v in model.parameters())
        trainable_count=sum(v.numel() for v in model.parameters() if v.requires_grad)
        best=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
        last=torch.load(p/'last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==cfg['epochs'] and best['epoch']==ev['checkpoint_epoch']
        assert all(torch.isfinite(t).all() for t in last['model'].values())
        assert all(torch.isfinite(t).all() for t in best['model'].values())
        del last
        model.load_state_dict(best['model']);del best
        model.eval();freeze_for_evaluation(model)
        correct=[];radii=[];predictions=[]
        batches=loader(test,cfg['batch'],workers=0)
        with torch.no_grad():
            for x,y in batches:
                x,y=x.to(device),y.to(device); z=model(x)
                assert torch.isfinite(z).all()
                predictions.extend(z.argmax(1).cpu().tolist())
                correct.extend(z.argmax(1).eq(y).cpu().tolist())
                if cfg['dataset']=='cifar10': radii.extend(model.certificate(z,y).cpu().tolist())
        assert abs(np.mean(correct)-ev['clean']['accuracy'])<1e-12, name+' clean mismatch'
        rec=dict(checkpoint_epoch=ev['checkpoint_epoch'],parameters=parameter_count,trainable_parameters=trainable_count,
                 clean_accuracy=float(np.mean(correct)),prediction_counts=np.bincount(predictions,minlength=classes).tolist())
        if cfg['dataset']=='cifar10':
            saved=np.load(p/'test-margins.npz');radii=np.asarray(radii)
            assert np.array_equal(correct,saved['correct'])
            np.testing.assert_allclose(radii,saved['radii'],rtol=2e-5,atol=2e-6)
            rec['max_certificate_replay_error']=float(np.max(np.abs(radii-saved['radii'])))
            # Audit every evaluated PGD iterate, on eight fixed examples per model.
            indices=np.random.default_rng(123).permutation(len(test))[:8]
            xs,ys=zip(*(test[int(i)] for i in indices)); x=torch.stack(xs).to(device);y=torch.tensor(ys,device=device)
            geometry={}
            for eps in (.25,.5,1.):
                checked=GeometryChecked(model,x,eps)
                robust=pgd(checked,x,y,eps,steps=100,restarts=5)
                certified=torch.tensor(radii[indices]>eps,device=device)&model(x).argmax(1).eq(y)
                assert not (certified & ~robust).any(),name+' certificate violated by replay attack'
                geometry[str(eps)]=dict(calls=checked.calls,max_norm=checked.max_norm,
                                         certified_count=int(certified.sum()),robust_count=int(robust.sum()))
            rec['pgd_geometry_replay']=geometry
        records[name]=rec
        write_json(output/'checkpoint-replay.json',dict(complete=len(records)==len(jobs),checked=len(records),total=len(jobs),
                   attack_check_examples=8,attack_check_steps=100,attack_check_restarts=5,runs=records))
        print(f'{len(records)}/{len(jobs)} {name}: clean/certificate/checkpoint replay passed',flush=True)
        del model
    return records


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('review-runs'))
    p.add_argument('--output',type=Path,default=Path('artifacts/campaign'));p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);replay(a.root,a.output,a.device)
