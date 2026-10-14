import json, hashlib
from pathlib import Path
import torch
from ri_gcn_adapter import make_model, CANON_EDGES
from ri_gcn_upstream.classifier import RotationInvariantGCNClassifier

torch.set_num_threads(2)
torch.manual_seed(1701)
x = torch.randn(2, 150, 18, 6)
m = make_model().eval()
ref = RotationInvariantGCNClassifier(num_joints=18,in_channels=18,num_classes=1,
    connectivity=list(CANON_EDGES),strategy='uniform',max_hop=1,device='cpu').eval()
ref.load_state_dict({k:v for k,v in m.state_dict().items() if k!='A'})
xyz=x[...,:3];xyz=xyz-xyz.mean(dim=2,keepdim=True)
with torch.no_grad():
    y=m(x);z=ref(xyz).squeeze(-1)
assert y.shape==(2,) and torch.equal(y,z)
assert 'A' in m.state_dict() and not any(k.startswith('head.') for k in m.state_dict())
# Gram is invariant to one common orthogonal rotation; tolerance is FP32.
q,_=torch.linalg.qr(torch.randn(3,3));rot=x.clone();rot[...,:3]=x[...,:3]@q
with torch.no_grad():err=float((m(rot)-y).abs().max())
assert err<1e-5
m.train();m(x).square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
m.zero_grad(set_to_none=True);m.set_linear_probe().train()
bn_before={k:v.clone() for k,v in m.named_buffers() if 'running_' in k or 'num_batches_tracked' in k}
m(x).square().mean().backward()
assert [n for n,p in m.named_parameters() if p.requires_grad]==['fcn.weight','fcn.bias']
assert all(p.grad is None for n,p in m.named_parameters() if not n.startswith('fcn.'))
assert all(torch.equal(dict(m.named_buffers())[k],v) for k,v in bn_before.items())
assert not m.data_bn.training and not m.st_gcn_networks.training
m.set_linear_probe(False).train();assert all(p.requires_grad for p in m.parameters())
print(json.dumps({'status':'PASS','device':'cpu','torch':torch.__version__,'threads':torch.get_num_threads(),
 'shape':list(y.shape),'official_forward_max_abs_error':float((y-z).abs().max()),
 'rotation_max_abs_error':err,'parameters':sum(p.numel() for p in m.parameters()),
 'lp_parameters':sum(p.numel() for p in m.fcn.parameters()),'checks':['official_forward_equal','registered_A',
 'no_duplicate_head','rotation_invariance','finite_backward','LP_head_only','LP_BN_frozen','FT_restore']}))
