import json
from pathlib import Path
from revision.campaign import report,status
from revision.runner import write_json


def test_report_rejects_incomplete_table(tmp_path):
    root=tmp_path/'runs'; out=tmp_path/'out'
    write_json(root/'campaign.json',{'jobs':['missing']})
    report(root,out)
    assert not (out/'revision-results.tex').exists()
    assert json.loads((out/'summary.json').read_text())['missing']==['missing']
    assert status(root)['counts']=={'queued':1}


def test_completed_report(tmp_path):
    root=tmp_path/'runs';out=tmp_path/'out'
    write_json(root/'campaign.json',{'jobs':['run0','run1']})
    for i in range(2):
        write_json(root/f'run{i}/config.json',dict(dataset='cifar10',model='aol',depth=5,bias='zero',seed=i))
        write_json(root/f'run{i}/evaluation.json',dict(protocol_complete=True,clean={'accuracy':.5+i*.1},certified={'0.5':.1}))
    report(root,out)
    assert (out/'revision-results.tex').exists()
    assert json.loads((out/'summary.json').read_text())['complete']
