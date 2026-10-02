import math
import numpy as np
import pytest
import torch
from revision.models import ColumnLinear, FeedForward, ResidualSmall
from revision.math_checks import gaussian_variance, mean_field
from revision.runner import pgd

@pytest.mark.parametrize('shape',[(3,7),(7,3),(4,4)])
def test_normalization(shape):
    torch.manual_seed(12)
    layer=ColumnLinear(*shape,train_q=True).double()
    with torch.no_grad(): layer.log_q.normal_()
    w=layer.normalized_weight()
    assert torch.linalg.matrix_norm(w,ord=2)<=1+1e-12
    original=w.detach().clone()
    with torch.no_grad(): layer.weight.mul_(13)
    torch.testing.assert_close(layer.normalized_weight(),original)
    layer(torch.randn(5,shape[0],dtype=torch.double)).sum().backward()
    assert torch.isfinite(layer.weight.grad).all()


def test_zero_weights_and_bias():
    layer=ColumnLinear(3,4)
    assert torch.count_nonzero(layer.bias)==0
    with torch.no_grad(): layer.weight.zero_()
    layer(torch.randn(2,3)).sum().backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.normalized_weight().count_nonzero()==0


def test_exact_cases():
    for n in [1,2,8,64]:
        assert gaussian_variance(1,n)==1/n
    for d in [1,2,16]:
        assert gaussian_variance(d,1)==1/d
    assert mean_field(1,2)>gaussian_variance(1,2)
    assert abs(gaussian_variance(8,8)-gaussian_variance(8,8,256))<1e-6


def test_linear_margin_certificate_and_pgd():
    class Linear(FeedForward):
        def __init__(self):
            torch.nn.Module.__init__(self)
        def forward(self,x):
            t=x.flatten(1)[:,0]-.5
            return torch.stack([t,-t],1)/math.sqrt(2)
    model=Linear()
    x=torch.tensor([.8,.2]).reshape(2,1,1,1)
    y=torch.tensor([0,1])
    torch.testing.assert_close(model.certificate(model(x),y),torch.full((2,),.3))
    assert pgd(model,x,y,.2,20,2).all()
    assert not pgd(model,x,y,.4,30,2).any()


def test_bias_does_not_change_weight_draws():
    torch.manual_seed(3); zero=FeedForward(8,3,2,8,'zero')
    torch.manual_seed(3); corrected=FeedForward(8,3,2,8,'corrected')
    for a,b in zip(zero.net,corrected.net):
        if isinstance(a,ColumnLinear): torch.testing.assert_close(a.weight,b.weight)


def test_symmetric_pairwise_inequality():
    rng=np.random.default_rng(91)
    u,v=rng.normal(size=(2,10000))
    for a in [-1,-.5,0,.5,1]:
        def f(z): return np.maximum(z,0)+a*np.minimum(z,0)
        lhs=(f(u)-f(v))**2+(f(-u)-f(-v))**2
        assert np.all(lhs<=(1+a*a)*(u-v)**2+1e-12)


def test_bias_draws_independent_across_layers():
    model=FeedForward(8,3,3,8,'corrected')
    assert not torch.equal(model.net[1].bias,model.net[2].bias)


def test_nontrivial_closed_form():
    from revision.math_checks import gaussian_variance_adaptive
    assert abs(gaussian_variance_adaptive(2,2)-math.log(1+math.sqrt(2))/(2*math.sqrt(2)))<1e-9


def test_checkpoint_resume_matches_uninterrupted(tmp_path,monkeypatch):
    from revision import runner
    from torch.utils.data import TensorDataset
    gen=torch.Generator().manual_seed(54)
    ds=TensorDataset(torch.randn(16,3,generator=gen),torch.arange(16)%2)
    monkeypatch.setattr(runner,'data',lambda *args,**kwargs:(ds,ds,ds,3,2))
    cfg=dict(dataset='synthetic',model='aol',depth=2,width=4,bias='corrected',seed=4,epochs=2,batch=4,lr=.001,workers=0)
    runner.train(cfg,tmp_path/'whole',tmp_path,'cpu')
    runner.train(cfg,tmp_path/'resumed',tmp_path,'cpu',stop_after=1)
    runner.train(cfg,tmp_path/'resumed',tmp_path,'cpu',resume=True)
    whole=torch.load(tmp_path/'whole/last.pt',weights_only=False)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key,value in whole['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)


def test_cached_evaluation_preserves_input_gradient():
    from revision.runner import freeze_for_evaluation
    model=FeedForward(8,3,2,8,'corrected')
    x=torch.randn(2,8,requires_grad=True)
    y=model(x);g=torch.autograd.grad(y.sum(),x)[0]
    freeze_for_evaluation(model)
    z=model(x);h=torch.autograd.grad(z.sum(),x)[0]
    torch.testing.assert_close(y,z)
    torch.testing.assert_close(g,h)


@pytest.mark.parametrize('distribution,expected',[('gaussian',2/math.pi),('laplace',.5),('uniform',.75)])
def test_product_moment_identity(distribution,expected):
    from revision.math_checks import product_sum_absolute_moment
    assert abs(product_sum_absolute_moment(1,distribution)-expected)<2e-7


def test_gaussian_sum_moment_closed_form():
    from revision.math_checks import product_sum_absolute_moment
    from scipy.special import gammaln
    for n in [2,5,20]:
        expected=2/math.sqrt(math.pi)*math.exp(gammaln((n+1)/2)-gammaln(n/2))
        assert abs(product_sum_absolute_moment(n,'gaussian')-expected)<2e-7
