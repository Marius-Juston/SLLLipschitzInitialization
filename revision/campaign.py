"""Four-GPU queue, restartable progress reporting, and result aggregation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .runner import write_json


def names(configs, profile):
    selected=json.loads((configs/'manifest.json').read_text())
    if profile=='compact':
        selected=[n for n in selected if not (n.startswith('covertype') and int(n.rsplit('s',1)[1])>=3)
                  and not ('cifar10' in n and '-d15-' in n)]
    return selected


def completed(path):
    file=path/'evaluation.json'
    return file.exists() and json.loads(file.read_text()).get('protocol_complete',False)


def status(root):
    manifest=root/'campaign.json'
    jobs=json.loads(manifest.read_text())['jobs'] if manifest.exists() else [p.name for p in root.iterdir() if p.is_dir()]
    states=defaultdict(int); active=[]
    for name in jobs:
        p=root/name
        info=json.loads((p/'status.json').read_text()) if (p/'status.json').exists() else {'state':'queued'}
        state='evaluated' if completed(p) else info['state']
        states[state]+=1
        if state in ['training','evaluating','failed']:
            active.append(dict(name=name,**info))
    return dict(total=len(jobs),counts=dict(states),active=active)


def run(args):
    args.root.mkdir(parents=True,exist_ok=True)
    # Prevent two schedulers from assigning the same runs.
    import fcntl
    lock=(args.root/'scheduler.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    selected=names(args.configs,args.profile)
    campaign=dict(profile=args.profile,jobs=selected,devices=args.devices,started=time.time())
    existing=args.root/'campaign.json'
    if existing.exists():
        old=json.loads(existing.read_text())
        if old['jobs']!=selected: raise ValueError('Campaign profile changed; use another output root')
    write_json(existing,campaign)
    # Residual models first; shorter runs backfill devices as they become free.
    queue=[]
    for name in selected:
        config=json.loads((args.configs/f'{name}.json').read_text())
        out=args.root/name
        info=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if not completed(out):
            queue.append((name,config,'evaluate' if info.get('state') in ['trained','evaluating'] or info.get('action')=='evaluate' else 'train'))
    active={}; failures=[]
    import signal
    def stop(signum,frame):
        for proc,log,*_ in active.values(): proc.terminate()
        for proc,log,*_ in active.values():
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired: proc.kill()
            log.close()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    while queue or active:
        for gpu in args.devices:
            if gpu not in active and queue:
                name,cfg,action=queue.pop(0);out=args.root/name;out.mkdir(parents=True,exist_ok=True)
                log=(out/f'{action}.log').open('a')
                cmd=[sys.executable,'-m','revision.runner',action,'--output',str(out),'--data',str(args.data),'--device',f'cuda:{gpu}']
                if action=='train': cmd+=['--config',str(args.configs/f'{name}.json'),'--resume']
                else:
                    write_json(out/'status.json',dict(state='evaluating'))
                    if cfg['dataset']!='covertype': cmd+=['--attack']
                    if cfg['model']=='residual': cmd+=['--autoattack']
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
                active[gpu]=(proc,log,name,cfg,action)
        for gpu,(proc,log,name,cfg,action) in list(active.items()):
            code=proc.poll()
            if code is None: continue
            log.close(); del active[gpu]
            if code:
                failures.append(dict(name=name,action=action,returncode=code))
                write_json(args.root/name/'status.json',dict(state='failed',action=action,returncode=code))
            elif action=='train':
                # Evaluate immediately on the same queue priority; no test-driven selection.
                queue.insert(0,(name,cfg,'evaluate'))
        write_json(args.root/'progress.json',dict(**status(args.root),failures=failures,
                   elapsed_hours=(time.time()-campaign['started'])/3600))
        if queue or active: time.sleep(10)
    print(json.dumps(status(args.root),indent=2))
    if failures: raise SystemExit(1)


def report(root, output):
    manifest=json.loads((root/'campaign.json').read_text()); jobs=manifest['jobs']
    missing=[name for name in jobs if not completed(root/name)]
    groups=defaultdict(list)
    for name in jobs:
        p=root/name
        if not completed(p): continue
        cfg=json.loads((p/'config.json').read_text());ev=json.loads((p/'evaluation.json').read_text())
        key=(cfg['dataset'],cfg['model'],cfg['depth'],cfg['bias'])
        row=dict(seed=cfg['seed'],clean=ev['clean']['accuracy'])
        for metric in ['certified','noise','pgd','autoattack']:
            for eps,val in ev.get(metric,{}).items(): row[f'{metric}_{eps}']=val
        groups[key].append(row)
    summary=[]
    for key,rows in groups.items():
        metrics={}
        for metric in rows[0]:
            if metric=='seed':continue
            values=[r[metric] for r in rows]
            metrics[metric]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None)
        summary.append(dict(dataset=key[0],model=key[1],depth=key[2],bias=key[3],seeds=rows,metrics=metrics))
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'summary.json',dict(complete=not missing,missing=missing,groups=summary))
    if missing:
        print(f'Partial JSON saved; {len(missing)} runs missing. Publication table not generated.')
        return
    lines=[r'\begin{table*}',r'\centering',r'\caption{Held-out accuracy and certified accuracy (percent; mean $\pm$ sample standard deviation).}',
           r'\begin{tabular}{llllrr}',r'Dataset & Model & Depth & Bias & Clean & Certified $\epsilon=0.5$ \\',r'\hline']
    def fmt(m): return '--' if m is None else f"{100*m['mean']:.2f}"+(f" $\\pm$ {100*m['std']:.2f}" if m['std'] is not None else '')
    for g in summary:
        lines.append(f"{g['dataset']} & {g['model']} & {g['depth']} & {g['bias']} & {fmt(g['metrics']['clean'])} & {fmt(g['metrics'].get('certified_0.5'))} \\\\")
    lines += [r'\end{tabular}',r'\label{tab:revision-results}',r'\end{table*}']
    (output/'revision-results.tex').write_text('\n'.join(lines)+'\n')
    print(output/'revision-results.tex')


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','status','report'])
    p.add_argument('--root',type=Path,default=Path('review-runs'))
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--configs',type=Path,default=Path('configs/revision'))
    p.add_argument('--output',type=Path,default=Path('artifacts/campaign'))
    p.add_argument('--profile',choices=['full','compact'],default='full')
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3])
    args=p.parse_args()
    if args.action=='run':run(args)
    elif args.action=='status':print(json.dumps(status(args.root),indent=2))
    else:report(args.root,args.output)

if __name__=='__main__':main()
