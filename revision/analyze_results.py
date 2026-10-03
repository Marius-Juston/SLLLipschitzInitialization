"""Audit completed runs and export seed-level evidence without changing evaluations.

Run from the repository root with ``uv run --frozen python -m revision.analyze_results``.
The audit reconstructs certificates on the exact attack subset; it never compares
full-test certification to subset attack accuracy as though denominators matched.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess
from functools import lru_cache

import numpy as np
from .runner import write_json

EPS = ('0.25', '0.5', '1.0')
NOISE = ('0.01', '0.03', '0.05')


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)) if len(a)>1 else None)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(None)
def committed_hash(commit, name):
    return hashlib.sha256(subprocess.check_output(["git", "show", f"{commit}:{name}"])).hexdigest()


def audit_run(path, expected):
    config = json.loads((path/'config.json').read_text())
    require(config == expected, f'{path.name}: config differs from manifest')
    ev = json.loads((path/'evaluation.json').read_text())
    history = json.loads((path/'history.json').read_text())
    require(ev['protocol_complete'], f'{path.name}: incomplete evaluation')
    require([r['epoch'] for r in history] == list(range(1, config['epochs']+1)),
            f'{path.name}: missing/duplicate epochs')
    selected = max(history, key=lambda r:r['validation']['accuracy'])['epoch']
    require(ev['checkpoint_epoch'] == selected, f'{path.name}: checkpoint selection mismatch')
    for r in history:
        require(np.isfinite([r['train_loss'], r['validation']['loss'], r['validation']['accuracy']]).all(),
                f'{path.name}: nonfinite training record')
    require(0 <= ev['clean']['accuracy'] <= 1 and np.isfinite(ev['clean']['loss']),
            f'{path.name}: invalid clean metrics')
    provenance = json.loads((path/'provenance.json').read_text())
    require({'revision/models.py','revision/runner.py','revision/math_checks.py','revision/vendor/sll_layers.py'} <= set(provenance['source_hashes']),
            f'{path.name}: incomplete source provenance')
    for name, digest in provenance['source_hashes'].items():
        require(committed_hash(provenance['code_commit'], name) == digest,
                f'{path.name}: original source changed: {name}')
    row = dict(run=path.name, **config, checkpoint_epoch=selected, clean=ev['clean']['accuracy'])
    if config['dataset'] != 'covertype':
        saved = np.load(path/'test-margins.npz')
        radii, correct, margins = saved['radii'], saved['correct'], saved['margins']
        require(len(radii)==len(correct)==len(margins)==10000, f'{path.name}: test size mismatch')
        require(np.isfinite(radii).all() and (radii>=0).all() and np.isfinite(margins).all(),
                f'{path.name}: invalid margins/radii')
        require(abs(correct.mean()-row['clean'])<1e-12, f'{path.name}: clean count mismatch')
        require(abs(margins.mean()-ev['margin_mean'])<1e-10, f'{path.name}: margin mean mismatch')
        indices = np.random.default_rng(123).permutation(10000)[:1000]
        row['subset_clean'] = float(correct[indices].mean())
        require(ev['attack_subset_size']==1000, f'{path.name}: incorrect attack subset size')
        for eps in EPS:
            certified = float(np.mean(correct & (radii > float(eps))))
            require(abs(certified-ev['certified'][eps])<1e-12, f'{path.name}: certificate count mismatch')
            subset_cert = float(np.mean(correct[indices] & (radii[indices]>float(eps))))
            row['certified_'+eps] = certified
            row['subset_certified_'+eps] = subset_cert
            for attack in ('pgd','autoattack') if config['model']=='residual' else ('pgd',):
                acc=ev[attack][eps]
                require(subset_cert-1e-6 <= acc <= row['subset_clean']+1e-6,
                        f'{path.name}: same-subset certificate/attack/clean ordering fails')
                require(abs(acc*1000-round(acc*1000))<1e-3, f'{path.name}: attack denominator mismatch')
                row[attack+'_'+eps] = acc
        require(all(ev['certified'][a]>=ev['certified'][b] for a,b in zip(EPS,EPS[1:])),
                f'{path.name}: certificates not monotone')
        for level in NOISE:
            acc=ev['noise'][level]
            require(0<=acc<=1 and abs(acc*10000-round(acc*10000))<1e-7,
                    f'{path.name}: invalid noise count')
            row['noise_'+level]=acc
        if config['model']=='residual':
            log=(path/'evaluate.log').read_text()
            checks=re.findall(r'max L2 perturbation: ([\d.]+), nan in tensor: (\d+), max: ([\d.]+), min: ([\d.]+)',log)
            require(len(checks)==3, f'{path.name}: missing AutoAttack geometry logs')
            for values,eps in zip(checks,EPS):
                norm,nan,hi,lo=map(float,values)
                require(norm<=float(eps)+1e-5 and nan==0 and hi<=1 and lo>=0,
                        f'{path.name}: AutoAttack geometry failure')
    row['training_hours'] = sum(r['epoch_seconds'] for r in history)/3600
    record = dict(config=config, evaluation=ev, history=history, provenance=provenance,
                  initialization=json.loads((path/'initialization.json').read_text()),
                  final_diagnostics=json.loads((path/'final-diagnostics.json').read_text()))
    if (path/'provenance-history.json').exists():
        record['provenance_history']=json.loads((path/'provenance-history.json').read_text())
    return row, record


def analyze(root, output):
    manifest=json.loads((root/'campaign.json').read_text())
    require(manifest['jobs']==json.loads(Path('configs/revision/manifest.json').read_text()), 'campaign manifest mismatch')
    rows=[]; records={}
    for name in manifest['jobs']:
        row, record=audit_run(root/name,json.loads((Path('configs/revision')/(name+'.json')).read_text()))
        rows.append(row); records[name]=record
    groups=defaultdict(list)
    for r in rows: groups[(r['dataset'],r['model'],r['depth'],r['bias'])].append(r)
    identity={'run','dataset','model','depth','width','bias','seed','epochs','batch','lr','workers','checkpoint_epoch'}
    summary=[]
    for key, group in groups.items():
        metrics={k:stats([r[k] for r in group]) for k in group[0] if k not in identity}
        summary.append(dict(dataset=key[0],model=key[1],depth=key[2],bias=key[3],metrics=metrics))
    pairs=[]
    for key, group in groups.items():
        if key[3]!='corrected': continue
        baseline='default' if key[1]=='residual' else 'zero'
        comparison={r['seed']:r for r in groups[(*key[:3],baseline)]}
        require(set(comparison)=={r['seed'] for r in group}, 'unmatched seeds')
        differences={k:[100*(r[k]-comparison[r['seed']][k]) for r in sorted(group,key=lambda r:r['seed'])]
                     for k in group[0] if k not in identity and k!='training_hours'}
        pairs.append(dict(dataset=key[0],model=key[1],depth=key[2],baseline=baseline,
                          units='percentage points', seeds=sorted(comparison), differences=differences,
                          metrics={k:stats(v) for k,v in differences.items()}))
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'validated-summary.json',dict(groups=summary,paired_comparisons=pairs))
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (output/'seed-results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader();writer.writerows(rows)
    (output/'campaign-records.json.gz').write_bytes(gzip.compress(json.dumps(records,sort_keys=True).encode(),mtime=0))
    # Preserve compact per-example evidence for rechecking aggregate certificate bounds.
    arrays={}
    for name in records:
        path=root/name/'test-margins.npz'
        if path.exists():
            with np.load(path) as saved:
                for k in saved.files: arrays[name+'__'+k]=saved[k]
    np.savez_compressed(output/'test-margins.npz',**arrays)
    files={}
    for name in records:
        for leaf in ('config.json','evaluation.json','history.json','best.pt','initialization.json','final-diagnostics.json','provenance.json','test-margins.npz'):
            p=root/name/leaf
            if p.exists(): files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(output/'source-manifest.json',files)
    write_json(output/'audit.json',dict(runs=len(rows),checks_passed=True,
        checks=['manifest/config match','complete contiguous histories','first best validation checkpoint',
                'finite losses','original source hashes against recorded Git commit','full-test margin/certificate counts',
                'same-subset certificate <= attack accuracy <= clean accuracy',
                'certificate monotonicity','metric denominators','AutoAttack perturbation logs'],
        source_commits=sorted({r['provenance']['code_commit'] for r in records.values()}),
        source_difference='52d2f2d to 551d1c1 adds only a paused-status write in the optional stop_after path; training/evaluation arithmetic unchanged.',
        resumed_runs=[n for n,r in records.items() if 'provenance_history' in r],
        limitations=['Aggregate checks do not prove per-example attack/certificate consistency.',
                     'Original PGD adversarial tensors were not retained; projector requires separate replay check.',
                     'Early pilot provenance lacks source hashes; resumed runs retain available provenance history.'],
        training_gpu_hours=sum(r['training_hours'] for r in rows)))
    print(json.dumps(dict(runs=len(rows),groups=len(groups),paired_comparisons=len(pairs),output=str(output)),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('review-runs'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/campaign'))
    args=parser.parse_args();analyze(args.root,args.output)
