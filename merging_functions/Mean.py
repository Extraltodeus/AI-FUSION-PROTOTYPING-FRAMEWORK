import torch

@torch.no_grad()
def mean(t, method, **kwargs):
    if t.shape[0] == 1:
        return t[0], None
    t = t.to(dtype=torch.float32)
    if method == "arithmetic":
        return t.sum(dim=0).div(t.shape[0]), None
    elif method == "torch.mean":
        return t.mean(dim=0), None

widgets = {
            "required": {
                "method" : (["torch.mean", "arithmetic"], {"default": "torch.mean"}),
                }
            }

function_properties = {
    "function": mean,
    "widgets": widgets,
    }
