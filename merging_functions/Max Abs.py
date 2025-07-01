import torch

@torch.no_grad()
def maxabsmerge(t, match_first_model_sign, **kwargs):
    if t.shape[0] == 1:
        return t[0], None
    if match_first_model_sign:
        t[t.sign() != t[0].sign()] = 0
    tmin = t.min(dim=0).values
    tmax = t.max(dim=0).values
    bt = tmin.abs() > tmax.abs()
    tmax[bt] = tmin[bt]
    return tmax, None

widgets = {
            "required": {
                "match_first_model_sign" : ("BOOLEAN", {"default": False}),
                }
            }

function_properties = {
    "function": maxabsmerge,
    "widgets": widgets,
    }
