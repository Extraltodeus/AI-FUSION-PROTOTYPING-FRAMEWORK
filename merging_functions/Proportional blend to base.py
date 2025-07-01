import torch
eps16 = torch.finfo(torch.float16).eps
eps32 = torch.finfo(torch.float32).eps
maxnorm16 = lambda x: x / x.max(dim=0).values.add(eps16)

def normd(d, normalization, power_after_normalization):
    if normalization == "individual":
        s = d.shape
        d = d.view(d.shape[0], -1)
        d = d.div(d.max(dim=1, keepdim=True).values.add(eps16)).view(s)
    elif normalization == "global" and power_after_normalization > 1:
        d = maxnorm16(d)
    if power_after_normalization > 1:
        d = d.pow(power_after_normalization)
    return d

def mergeDiff(t, normalization, power_after_normalization, multiply_by_diff_to_first, reverse_diff_to_first, loaded_list, **kwargs):
    if t.shape[0] < 3 or not loaded_list[-1][1]:
        return t[0], None
    t = t.to(dtype=torch.float32)
    t, b = t[:-1], t[-1]
    d = (t - b).pow(2)
    if d.max(dim=0).values.sum() == 0:
        return t[0], None
    d = normd(d, normalization, power_after_normalization)
    if multiply_by_diff_to_first:
        a = (t[0] - t).pow(2)
        a = normd(d, normalization, power_after_normalization)
        if reverse_diff_to_first:
            a = 1 - a
        a[0] = 1
        d = d.mul(a)
    s = d.sum(dim=0)
    m = s >= eps16
    d = d.div(s.add(eps32))
    r = t[0].clone()
    r[m] = t.mul(d).sum(dim=0)[m]
    return r, None

widgets = {"required": {
    "normalization" : (["global","individual"],),
    "power_after_normalization": ("INT", {"default": 1, "min": 1, "max": 1000}),
    "multiply_by_diff_to_first" : ("BOOLEAN", {"default": True}),
    "reverse_diff_to_first" : ("BOOLEAN", {"default": True}),
}}
function_properties = {
    "function": mergeDiff,
    "widgets": widgets,
    }