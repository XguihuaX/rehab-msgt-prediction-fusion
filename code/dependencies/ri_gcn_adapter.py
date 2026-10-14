"""Protocol-adapted RI-GCN; upstream e6c3d21ead98136878c97ad420c6e0da4a87cb56.
No data access, label conversion, training, or random reset occurs on import.
"""
import sys
from pathlib import Path
import torch

# Allows the existing file-based model loader as well as normal module imports.
_DEPENDENCY_DIR = str(Path(__file__).resolve().parent)
if _DEPENDENCY_DIR not in sys.path:
    sys.path.insert(0, _DEPENDENCY_DIR)
from ri_gcn_upstream.classifier import RotationInvariantGCNClassifier

CANON_TREE = ((0, 1, 17, 2), (0, 11, 12, 13), (0, 14, 15, 16),
              (17, 3, 4, 5, 6), (17, 7, 8, 9, 10))
CANON_EDGES = tuple((a, b) for chain in CANON_TREE for a, b in zip(chain[:-1], chain[1:]))

class RIGCNAdapter(RotationInvariantGCNClassifier):
    def __init__(self):
        super().__init__(num_joints=18, in_channels=18, num_classes=1,
                         connectivity=list(CANON_EDGES), strategy='uniform',
                         max_hop=1, device='cpu')
        adjacency = self.A
        del self.A
        self.register_buffer('A', adjacency)
        self._linear_probe = False

    @property
    def head(self):
        # Property avoids registering the same parameters under two paths.
        return self.fcn

    def set_linear_probe(self, enabled=True):
        self._linear_probe = bool(enabled)
        for parameter in self.parameters():
            parameter.requires_grad_(not self._linear_probe)
        for parameter in self.fcn.parameters():
            parameter.requires_grad_(True)
        self.train(self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, '_linear_probe', False):
            self.data_bn.eval()
            self.st_gcn_networks.eval()
            self.fcn.train(mode)
        return self

    def forward(self, x):
        if x.ndim != 4 or x.shape[2:] != (18, 6):
            raise ValueError('Expected B,T,18,6 canonical input')
        xyz = x[..., :3]
        # Preserve the official deterministic frame-wise centering, using the
        # shared protocol's temporal sampling and participant split.
        xyz = xyz - xyz.mean(dim=2, keepdim=True)
        return super().forward(xyz).squeeze(-1)

def make_model():
    """Caller controls seeds, device, sigmoid/BCE/MSE, and all data budgets."""
    return RIGCNAdapter()
