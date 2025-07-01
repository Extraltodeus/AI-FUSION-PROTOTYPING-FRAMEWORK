import torch
from random import randint

@torch.no_grad()
def random_model(t, seed, method, layer_key, **kwargs):
    if t[0].numel() < 100 or len(t) < 2:
        return t[0], None
    if method == "random pick":
        pick = torch.randint_like(t[0],t.shape[0])
        res = t[0].clone()
        for i in range(t.shape[0]):
            res[pick == i] = t[i][pick == i]
        return res, None
    tmax = t.max(dim=0).values
    tmin = t.min(dim=0).values
    if torch.equal(tmax,tmin):
        return t[0], None
    torch.manual_seed(seed + len(layer_key))
    res = torch.rand_like(t[0], device=t.device, dtype=t.dtype)
    neq = tmax != tmin
    res[neq] = res[neq] * (tmax[neq] - tmin[neq]) + tmin[neq]
    res[~neq] = t[0][~neq]
    return res, None

widgets = {
            "required": {
                "method" : (["uniform min max", "random pick"], {"default": "random pick"}),
                "seed": ("INT", {"default": randint(0, 0xffffffffffffffff), "min": 0, "max": 0xffffffffffffffff}),
                }
            }

function_properties = {
    "function": random_model,
    "widgets": widgets,
    }
