import torch
eps16 = torch.finfo(torch.float16).eps

def slerp(d, b, t):
    dn, bn = d.div(torch.linalg.norm(d)), b.div(torch.linalg.norm(b))
    dot = (dn * bn).sum().clamp(min=-1.0 + eps16, max=1.0 - eps16)
    theta = dot.arccos()
    sin_theta = theta.sin()
    if sin_theta.sum() == 0:
        return d.mul(1 - t).add(b.mul(t))
    f1 = theta.mul(1 - t).sin().div(sin_theta)
    f2 = theta.mul(t).sin().div(sin_theta)
    return d.mul(f1).add(b.mul(f2))

nrms = {
    "minmax": lambda x: (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values).add(eps16),
    "max":    lambda x: x / x.max(dim=0).values.add(eps16)
}
cmps = {
    "one_with_two":   1,
    "one_with_three": 2
}

def mergeDiff(t, normalization, power, less_different_is_more_2_in_1, comparison, use_slerp, loaded_list, **kwargs):
    cmid = cmps[comparison]
    d = nrms[normalization]((t[0] - t[cmid]).abs())
    if less_different_is_more_2_in_1:
        d = 1 - d
    if power > 1:
        d = d.pow(power)
    if use_slerp:
        return slerp(t[0], t[1], d)
    return (t[0] * (1 - d) + t[1] * d)

# For a triple dose of pimpin
def adddiff(t, normalization, power, less_different_is_more_2_in_1, comparison, use_slerp, atan_clamp, loaded_list, **kwargs):
    cmid = cmps[comparison]
    if t.shape[0] < (2 + (comparison == "one_with_three")) or (t.shape[0] == 3 and torch.equal(t[0], t[cmid])):
        return t[0], None
    s = t[0].shape
    t = t.view(t.shape[0], -1).to(dtype=torch.float32)
    if atan_clamp:
        n = torch.linalg.norm(t, dim=1, keepdim=True)
        t = t.div(n.add(eps16))
        y, x = t.sin().mul(n), t.cos().mul(n)
        y = mergeDiff(y, normalization, power, less_different_is_more_2_in_1, comparison, use_slerp, loaded_list, **kwargs)
        x = mergeDiff(x, normalization, power, less_different_is_more_2_in_1, comparison, use_slerp, loaded_list, **kwargs)
        return torch.atan2(y, x).mul(n[0]).view(s), None
    return mergeDiff(t, normalization, power, less_different_is_more_2_in_1, comparison, use_slerp, loaded_list, **kwargs).view(s), None

widgets = {"required": {
    "normalization" : ([n for n in nrms],),
    "power": ("INT", {"default": 1, "min": 1, "max": 1000}),
    "less_different_is_more_2_in_1": ("BOOLEAN", {"default": True}),
    "comparison" : ([n for n in cmps],),
    "use_slerp":  ("BOOLEAN", {"default": True}),
    "atan_clamp": ("BOOLEAN", {"default": False}),
    "strength": ("FLOAT", {"default": 1.0, "min": -1e3, "max": 1e3, "step": 1e-2, "round": 1e-4}),
}}

function_properties = {
    "function": adddiff,
    "widgets": widgets,
    }