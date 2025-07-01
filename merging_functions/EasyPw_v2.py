import torch
eps = torch.finfo(torch.float32).eps
eps16 = torch.finfo(torch.float16).eps
maxnorm16 = lambda x: x / x.max(dim=0).values.add(eps16)
mmnorme16 = lambda x: (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values).add(eps16)

def abspow(d, p):
    if p <= 1 or p%2 == 1:
        d = d.abs()
    if p > 1:
        return d.pow(p)
    return d

@torch.no_grad()
def simple_pw_weights(t, power_before, power_after, use_base, sign_filter, b=None):
    d = torch.zeros_like(t)
    if use_base:
        s = torch.zeros_like(t)
    for i in range(t.shape[0]):
        d[i] += abspow(t.sub(t[i]), power_before).sum(dim=0)
        if use_base:
            s[i] += abspow(t[i].sub(b), power_before)
            d[i] += s[i]
    d = 1 - maxnorm16(d)
    if power_after > 1:
        d = d.pow(power_after)
    if use_base:
        d[s <= (eps16 * 3)] = 0
        del s
    if sign_filter:
        h = torch.gather(t, dim=0, index=d.sort(dim=0, descending=True).indices)[0]
        d[t.sign() != h.sign()] = 0
    z = d.max(dim=0).values <= eps
    r = t.mul(d).sum(dim=0)
    if z.sum() > 0:
        r[z] = t[0][z]
        r[~z] = r[~z].div(d.sum(dim=0)[~z])
    else:
        r = r.div(d.sum(dim=0))
    return r

@torch.no_grad()
def merging_function(t, power_before, power_after, recursions, last_is_base, sign_majority_filter, loaded_list, **kwargs):
    use_base = loaded_list[:-1][1] and last_is_base
    if t.shape[0] < (3 + use_base) or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        return t[0], None

    t = t.to(dtype=torch.float32)
    b = None
    if use_base:
        t, b = t[:-1], t[-1]
    for i in range(recursions):
        r = simple_pw_weights(t, power_before=power_before, power_after=power_after, use_base=use_base, sign_filter=sign_majority_filter, b=b)
        if i < (recursions - 1):
            t = torch.cat([t, r.unsqueeze(0)])
    return r, None

widgets = {
            "required": {
                "power_before": ("INT", {"default": 1, "min": 1, "max": 16}),
                "power_after": ("INT", {"default": 1, "min": 1, "max": 16}),
                "recursions": ("INT", {"default": 1, "min": 1, "max": 16}),
                "last_is_base" : ("BOOLEAN", {"default": False}),
                "sign_majority_filter" : ("BOOLEAN", {"default": False}),
                }
            }

function_properties = {
    "function": merging_function,
    "widgets": widgets,
    }
