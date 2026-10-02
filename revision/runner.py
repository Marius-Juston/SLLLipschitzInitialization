"""Train/evaluate revision studies; all CLI paths are relative to repository root."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from .models import build, ColumnLinear, UPSTREAM_COMMIT


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def data(config, root, download=False):
    if config['dataset'] == 'covertype':
        from sklearn.datasets import fetch_covtype
        ds = fetch_covtype(data_home=str(root), download_if_missing=download)
        x = torch.tensor(ds.data,dtype=torch.float32)
        y = torch.tensor(ds.target-1,dtype=torch.long)
        perm = np.random.default_rng(0).permutation(len(y))
        train_end, val_end = int(.72*len(y)),int(.8*len(y))
        tr, va, te = perm[:train_end],perm[train_end:val_end],perm[val_end:]
        # Train-only scaling, retained inside all evaluated input coordinates.
        mean, std = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
        x = (x-mean)/std
        full = TensorDataset(x,y)
        return Subset(full,tr),Subset(full,va),Subset(full,te),54,7
    cls = datasets.CIFAR10 if config['dataset']=='cifar10' else datasets.CIFAR100
    training = transforms.Compose([
        transforms.RandomCrop(32,padding=4,padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        *([transforms.ColorJitter(brightness=.1,contrast=(.5,2.),saturation=(.3,2.),hue=.02)]
          if config['model']=='residual' else []),
        transforms.ToTensor()])
    train_full = cls(str(root),train=True,transform=training,download=download)
    val_full = cls(str(root),train=True,transform=transforms.ToTensor(),download=False)
    test = cls(str(root),train=False,transform=transforms.ToTensor(),download=download)
    perm = np.random.default_rng(0).permutation(50000)
    return Subset(train_full,perm[:45000]),Subset(val_full,perm[45000:]),test,3072,len(train_full.classes)


def loader(ds, batch, shuffle=False, seed=0, workers=4):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,
                      pin_memory=torch.cuda.is_available(),generator=gen)


def objective(logits, labels, residual=False):
    if residual:
        logits = (logits-math.sqrt(2)*1.5*F.one_hot(labels,logits.shape[1]))/.25
        return .25*F.cross_entropy(logits,labels)
    return F.cross_entropy(logits,labels)


@torch.no_grad()
def clean_eval(model, batches, device):
    model.eval()
    correct = count = 0
    loss = 0.
    for x,y in batches:
        x,y = x.to(device),y.to(device)
        z = model(x)
        loss += F.cross_entropy(z,y,reduction='sum').item()
        correct += (z.argmax(1)==y).sum().item()
        count += len(y)
    return dict(accuracy=correct/count,loss=loss/count)


def diagnostics(model, batches, device):
    model.eval()
    rows, handles = {}, []
    def hook(name):
        def capture(module, inputs, output):
            flat = output.detach().float().flatten(1)
            rows[name] = dict(second_moment=flat.square().mean().item(),
                              input_variance=flat.var(dim=0,unbiased=False).mean().item())
            if output.requires_grad:
                output.register_hook(lambda g: rows[name].update(gradient_rms=g.detach().square().mean().sqrt().item()))
        return capture
    for name,m in model.named_modules():
        if isinstance(m,ColumnLinear) or m.__class__.__name__.startswith('SDPBased'):
            handles.append(m.register_forward_hook(hook(name)))
    x,y = next(iter(batches))
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model(x.to(device)),y.to(device)).backward()
    for handle in handles:
        handle.remove()
    model.zero_grad(set_to_none=True)
    return rows


def train(config, output, root, device, pilot_steps=0, resume=False, stop_after=0):
    from torch.utils.tensorboard import SummaryWriter
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    seed_all(config['seed'])
    tr,va,_,inputs,classes = data(config,root)
    model = build(config,inputs,classes).to(device)
    residual = config['model']=='residual'
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'],betas=(.5,.9) if residual else (.9,.999))
    existing = output/'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError('Output directory contains a different configuration')
    if (output/'last.pt').exists() and not resume:
        raise ValueError('Existing checkpoint; use --resume')
    write_json(existing,config)
    provenance=dict(upstream_commit=UPSTREAM_COMMIT,
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)),
        torch=torch.__version__,cuda=torch.version.cuda,device=str(device),
        packages={p:importlib.metadata.version(p) for p in ['torch','torchvision','numpy','scipy','scikit-learn']},
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('revision').rglob('*.py')})
    previous=output/'provenance.json'
    if previous.exists():
        history=output/'provenance-history.json'
        records=json.loads(history.read_text()) if history.exists() else []
        records.append(json.loads(previous.read_text()));write_json(history,records)
    write_json(previous,provenance)
    start_epoch, best, step = 0,-1.,0
    checkpoint = output/'last.pt'
    if resume and checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_epoch,best,step = saved['epoch'],saved['best'],saved['step']
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
    writer = SummaryWriter(str(output/'tensorboard'))
    val_loader = loader(va,config['batch'],seed=0,workers=config['workers'])
    if not pilot_steps and start_epoch == 0:
        write_json(output/'initialization.json',diagnostics(model,val_loader,device))
    started = time.monotonic()
    timings=[]
    total_steps = config['epochs']*math.ceil(len(tr)/config['batch'])
    write_json(output/'status.json',dict(state='training',epoch=start_epoch,epochs=config['epochs']))
    for epoch in range(start_epoch,config['epochs']):
        model.train()
        loss_sum=count=0
        epoch_start=time.monotonic()
        batches = loader(tr,config['batch'],True,config['seed']*10000+epoch,config['workers'])
        for x,y in batches:
            x,y=x.to(device),y.to(device)
            if residual:
                lr=float(np.interp(step,[0,total_steps*.4,total_steps*.8,total_steps],[0,config['lr'],config['lr']/20,0]))
                for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            if pilot_steps and str(device).startswith('cuda'): torch.cuda.synchronize()
            t=time.monotonic()
            z=model(x)
            loss=objective(z,y,residual)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            step+=1
            if pilot_steps and str(device).startswith('cuda'): torch.cuda.synchronize()
            timings.append(time.monotonic()-t)
            loss_sum+=loss.item()*len(y); count+=len(y)
            if pilot_steps and len(timings)>=pilot_steps:
                steady=float(np.median(timings[min(3,len(timings)-1):]))
                result=dict(config=config,step_seconds=steady,
                            projected_training_hours=steady*total_steps/3600,
                            peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0,
                            elapsed_seconds=time.monotonic()-started)
                write_json(output/'pilot.json',result); print(json.dumps(result),flush=True)
                writer.close(); return
        validation=clean_eval(model,val_loader,device)
        record=dict(epoch=epoch+1,train_loss=loss_sum/count,validation=validation,
                    epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-started)
        for key,value in [('loss/train',record['train_loss']),('accuracy/validation',validation['accuracy'])]:
            writer.add_scalar(key,value,epoch+1)
        writer.flush()
        # Checkpoint is the source of truth; history is truncated to it on resume.
        history_path=output/'history.json'
        history=json.loads(history_path.read_text()) if history_path.exists() else []
        history=[r for r in history if r['epoch']<=epoch]+[record]
        improved=validation['accuracy']>best
        best=max(best,validation['accuracy'])
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,step=step,
                   torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   numpy_rng=np.random.get_state(),python_rng=random.getstate())
        torch.save(state,output/'last.tmp'); (output/'last.tmp').replace(checkpoint)
        if improved:
            torch.save(dict(model=model.state_dict(),epoch=epoch+1),output/'best.tmp')
            (output/'best.tmp').replace(output/'best.pt')
        write_json(history_path,history)
        write_json(output/'status.json',dict(state='training',**record,epochs=config['epochs'],
                   remaining_hours=(config['epochs']-epoch-1)*record['epoch_seconds']/3600))
        print(json.dumps(record),flush=True)
        if stop_after and epoch+1 >= stop_after:
            write_json(output/'status.json',dict(state='paused',epoch=epoch+1,epochs=config['epochs']))
            writer.close(); return
    write_json(output/'final-diagnostics.json',diagnostics(model,val_loader,device))
    write_json(output/'status.json',dict(state='trained',epoch=config['epochs'],elapsed_seconds=time.monotonic()-started))
    writer.close()


def pgd(model,x,y,eps,steps=100,restarts=5):
    """Keep every successful iterate; include clean errors in robust error."""
    correct=model(x).argmax(1).eq(y)
    for _ in range(restarts):
        delta=torch.randn_like(x)
        delta*=eps*torch.rand(len(x),device=x.device).reshape(-1,1,1,1)/delta.flatten(1).norm(dim=1).clamp_min(1e-30).reshape(-1,1,1,1)
        adv=(x+delta).clamp(0,1)
        for _ in range(steps):
            adv=adv.detach().requires_grad_(True)
            logits=model(adv)
            correct &= logits.argmax(1).eq(y).detach()
            grad=torch.autograd.grad(F.cross_entropy(logits,y),adv)[0]
            adv=adv.detach()+2.5*eps/steps*grad/grad.flatten(1).norm(dim=1).clamp_min(1e-30).reshape(-1,1,1,1)
            delta=adv-x
            delta*= (eps/delta.flatten(1).norm(dim=1).clamp_min(1e-30)).clamp_max(1).reshape(-1,1,1,1)
            adv=(x+delta).clamp(0,1)
        correct &= model(adv).argmax(1).eq(y).detach()
    return correct


def freeze_for_evaluation(model):
    """Cache parameter-only normalizers; preserve gradients with respect to input."""
    for parameter in model.parameters(): parameter.requires_grad_(False)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer,ColumnLinear):
                w=layer.normalized_weight().detach()
                layer.normalized_weight=lambda w=w:w
            elif hasattr(layer,'compute_t'):
                t=layer.compute_t().detach()
                layer.compute_t=lambda t=t:t
    return model


def evaluate(output, root, device, attack=False, autoattack=False, subset=1000, allow_incomplete=False):
    output=Path(output); config=json.loads((output/'config.json').read_text()); seed_all(config['seed']+50000)
    history=json.loads((output/'history.json').read_text())
    training_complete=bool(history and history[-1]['epoch']==config['epochs'])
    if not training_complete and not allow_incomplete:
        raise ValueError('Training is incomplete; use --allow-incomplete only for a pilot')
    _,_,test,inputs,classes=data(config,root)
    model=build(config,inputs,classes).to(device)
    saved=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(saved['model']); model.eval()
    freeze_for_evaluation(model)
    batches=loader(test,config['batch'],workers=config['workers'])
    result=dict(checkpoint_epoch=saved['epoch'],clean=clean_eval(model,batches,device),
        protocol_complete=training_complete and (config['dataset']=='covertype' or
            (attack and subset==1000 and (config['model']!='residual' or autoattack))))
    # Covertype contains categorical features: do not report image perturbation metrics.
    if config['dataset']=='covertype':
        write_json(output/'evaluation.json',result); return
    radii,correct,margins=[],[],[]; noisy={str(s):0 for s in [.01,.03,.05]}
    with torch.no_grad():
        for x,y in batches:
            x,y=x.to(device),y.to(device); z=model(x)
            radii.extend(model.certificate(z,y).cpu().tolist()); correct.extend(z.argmax(1).eq(y).cpu().tolist())
            other=z.clone().scatter_(1,y[:,None],-torch.inf).amax(1)
            margins.extend((z.gather(1,y[:,None]).squeeze(1)-other).cpu().tolist())
            for s in [.01,.03,.05]:
                noisy[str(s)]+=model((x+s*torch.randn_like(x)).clamp(0,1)).argmax(1).eq(y).sum().item()
    result.update(certified={str(e):float(np.mean(np.asarray(correct)&(np.asarray(radii)>e))) for e in [.25,.5,1.]},
                  noise={s:v/len(test) for s,v in noisy.items()},margin_mean=float(np.mean(margins)),
                  attack_subset_size=min(subset,len(test)))
    # Fixed, seed-independent subset; all models evaluated on identical examples.
    indices=np.random.default_rng(123).permutation(len(test))[:subset]
    attacked=loader(Subset(test,indices),min(64,config['batch']),workers=config['workers'])
    if attack:
        result['pgd']={}
        for eps in [.25,.5,1.]:
            success=0
            for x,y in attacked:
                success+=pgd(model,x.to(device),y.to(device),eps).sum().item()
            result['pgd'][str(eps)]=success/len(indices)
    if autoattack:
        from autoattack import AutoAttack
        xs,ys=zip(*list(attacked)); x,y=torch.cat(xs).to(device),torch.cat(ys).to(device)
        result['autoattack']={}
        for eps in [.25,.5,1.]:
            adversary=AutoAttack(model,norm='L2',eps=eps,version='standard',device=str(device),seed=123)
            adv=adversary.run_standard_evaluation(x,y,bs=64)
            with torch.no_grad(): result['autoattack'][str(eps)]=float(model(adv).argmax(1).eq(y).float().mean())
    write_json(output/'evaluation.json',result)
    np.savez_compressed(output/'test-margins.npz',radii=radii,correct=correct,margins=margins)
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['download','train','pilot','evaluate'])
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=Path('review-runs/smoke'))
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--attack',action='store_true')
    p.add_argument('--autoattack',action='store_true')
    p.add_argument('--subset',type=int,default=1000)
    p.add_argument('--allow-incomplete',action='store_true')
    args=p.parse_args()
    if args.action=='evaluate':
        evaluate(args.output,args.data,args.device,args.attack,args.autoattack,args.subset,args.allow_incomplete); return
    if args.action=='download':
        for name in ['cifar10','covertype']:
            data(dict(dataset=name,model='aol'),args.data,download=True)
        return
    config=json.loads(args.config.read_text())
    try: train(config,args.output,args.data,args.device,args.steps if args.action=='pilot' else 0,args.resume,args.stop_after)
    except Exception as exc:
        write_json(args.output/'status.json',dict(state='failed',error=repr(exc)))
        raise


if __name__=='__main__': main()
