import torch
from random import randint

@torch.no_grad()
def mfunc(t, flip_percentage, seed, **kwargs):
    t = t[0]
    if t.min() >= 0:
        return t
    n = t.numel()
    torch.manual_seed(seed + n + len(kwargs['layer_key']))
    p = int(n * flip_percentage / 100)
    if p == 0:
        return t, None
    s = t.shape
    t = t.view(-1)
    d = torch.zeros_like(t)
    d[:p] = 1
    d = d[torch.randperm(n)]
    t[d == 1] *= -1
    return t.view(s), None

widgets = {
            "required": {
                "flip_percentage": ("FLOAT", {"default": 5.0, "min": 0, "max": 100, "step": 1/100, "round": 1/1000}),
                "seed": ("INT", {"default": randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
                }
            }

function_properties = {
    "function": mfunc,
    "widgets": widgets,
    }
