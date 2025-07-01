import torch
eps16 = torch.finfo(torch.float16).eps

def slerp(d, b, t):
    dn, bn = d.div(torch.linalg.norm(d.view(-1))), b.div(torch.linalg.norm(b.view(-1)))
    dot = (dn * bn).sum().clamp(min=-1.0 + eps16, max=1.0 - eps16)
    theta = dot.arccos()
    sin_theta = theta.sin()
    if sin_theta.sum() == 0:
        return d.mul(1 - t).add(b.mul(t))
    f1 = theta.mul(1 - t).sin().div(sin_theta)
    f2 = theta.mul(t).sin().div(sin_theta)
    return d.mul(f1).add(b.mul(f2))

def merge_function(first_layer, get_layer, total_models, **kwargs):
    result = first_layer.to(dtype=torch.float32)
    j = 2
    for i in range(1, total_models):
        t = get_layer(i)
        if t is not None:
            s = 1 / j
            result = slerp(result, t.to(dtype=torch.float32), s)
            j += 1
    return result, None

widgets = {"required": {}}

function_properties = {
    "function": merge_function,
    "widgets": widgets,
    'is_low_memory': True,
    }
