"""Closed-vocabulary tag-ID recognizer: one crop in, two characters out.

The ID is always exactly two characters from A-Z0-9 (CONOPS v1.0 5.2.3 items 7
and 8; see RULE-002 for the open question about the vocabulary). The problem
therefore becomes two 36-way classifications instead of open-ended OCR, so the
model has no sequence decoding to get wrong and returns a calibrated
probability for every possible ID. That probability is what multi-frame
voting needs.
"""
import numpy as np
import torch
import torch.nn as nn

from synth import CHARSET, to_input  # noqa: F401  (to_input re-exported)


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class TagNet(nn.Module):
    """~1.2M parameters; small enough for a Jetson alongside the deer detector."""

    def __init__(self, n_cls=len(CHARSET)):
        super().__init__()
        self.features = nn.Sequential(
            _block(1, 32), nn.MaxPool2d(2),     # 48x32
            _block(32, 64), nn.MaxPool2d(2),    # 24x16
            _block(64, 128), nn.MaxPool2d(2),   # 12x8
            _block(128, 192),
            nn.AdaptiveAvgPool2d((3, 2)),        # keep left/right layout for the two characters
        )
        self.neck = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(192 * 6, 256), nn.ReLU(inplace=True))
        self.head1 = nn.Linear(256, n_cls)
        self.head2 = nn.Linear(256, n_cls)

    def forward(self, x):
        z = self.neck(self.features(x))
        return self.head1(z), self.head2(z)


def batch_tensor(uint8_imgs, device="cpu"):
    x = torch.as_tensor(np.asarray(uint8_imgs), device=device).float().div_(127.5).sub_(1)
    return x.unsqueeze(1)


@torch.no_grad()
def predict(model, crops):
    """Return [(id, confidence, (p1, p2))] for a list of grayscale crops."""
    model.eval()
    dev = next(model.parameters()).device
    l1, l2 = model(batch_tensor([to_input(c) for c in crops], dev))
    p1, p2 = l1.softmax(1), l2.softmax(1)
    c1, i1 = p1.max(1)
    c2, i2 = p2.max(1)
    p1, p2 = p1.cpu(), p2.cpu()
    return [(CHARSET[a] + CHARSET[b], float(x * y), (p1[k].numpy(), p2[k].numpy()))
            for k, (a, b, x, y) in enumerate(zip(i1.tolist(), i2.tolist(), c1, c2))]


def load(path, device="cpu"):
    m = TagNet()
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()
