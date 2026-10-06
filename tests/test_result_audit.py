import json
import numpy as np
import pytest
from revision.analyze_results import audit_run
from revision.runner import write_json


def fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr("revision.analyze_results.committed_hash", lambda *args: "fixture-hash")
    config=dict(dataset='cifar10',model='aol',depth=1,bias='zero',seed=0,epochs=2)
    history=[dict(epoch=i,train_loss=1.,validation=dict(loss=1.,accuracy=.6),epoch_seconds=1.) for i in (1,2)]
    correct=np.zeros(10000,dtype=bool)
    subset=np.random.default_rng(123).permutation(10000)[:1000]
    correct[subset]=True  # Deliberately different full-test and subset rates.
    radii=np.where(correct,2.,0.);margins=np.where(correct,1.,-1.)
    ev=dict(protocol_complete=True,checkpoint_epoch=1,clean=dict(accuracy=.1,loss=1.),
            certified={e:.1 for e in ('0.25','0.5','1.0')},noise={s:.1 for s in ('0.01','0.03','0.05')},
            margin_mean=float(margins.mean()),attack_subset_size=1000,pgd={e:1. for e in ('0.25','0.5','1.0')})
    for leaf,value in [('config',config),('evaluation',ev),('history',history),
                       ('provenance',{'code_commit':'fixture','source_hashes':{n:'fixture-hash' for n in ('revision/models.py','revision/runner.py','revision/math_checks.py','revision/vendor/sll_layers.py')}}),('initialization',{}),('final-diagnostics',{})]:
        write_json(tmp_path/(leaf+'.json'),value)
    np.savez(tmp_path/'test-margins.npz',radii=radii,correct=correct,margins=margins)
    return config,ev


def test_audit_uses_identical_subset_for_attack_bounds(tmp_path,monkeypatch):
    cfg,ev=fixture_run(tmp_path,monkeypatch)
    row,_=audit_run(tmp_path,cfg)
    assert row['clean']==.1 and row['subset_clean']==1.
    ev['pgd']['0.5']=.9  # Above full-test certificate, but below same-subset certificate.
    write_json(tmp_path/'evaluation.json',ev)
    with pytest.raises(ValueError,match='same-subset'):
        audit_run(tmp_path,cfg)


def test_audit_requires_first_best_validation_checkpoint(tmp_path,monkeypatch):
    cfg,ev=fixture_run(tmp_path,monkeypatch)
    ev['checkpoint_epoch']=2
    write_json(tmp_path/'evaluation.json',ev)
    with pytest.raises(ValueError,match='checkpoint selection'):
        audit_run(tmp_path,cfg)
