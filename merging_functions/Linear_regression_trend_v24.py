import torch
from math import floor, ceil, pi

eps16 = torch.finfo(torch.float16).eps
mmnorm16  = lambda x: (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values).add(eps16)
maxnorm16 = lambda x: x / x.max(dim=0).values.add(eps16)
pytmag = lambda y, x: (y.pow(2) + x.pow(2)).sqrt()

def abspow(d, p):
    if p <= 1 or p%2 == 1:
        d = d.abs()
    if p > 1:
        return d.pow(p)
    return d

def divnodiv(x, y):
    if y > 1:
        return x.div(y)
    return x

def get_mag(mi, norms):
    if mi is None:
        return None
    low  = floor(mi)
    high = ceil(mi)
    mag  = norms[low] * (1 - (mi - low)) + norms[high] * (mi - low)
    return mag

def slerp_score(d, b, t):
    dn, bn = d.div(torch.linalg.norm(d.view(-1))), b.div(torch.linalg.norm(b.view(-1)))
    dot = (dn * bn).sum().clamp(min=-1.0 + eps16, max=1.0 - eps16)
    theta = dot.arccos()
    sin_theta = theta.sin()
    if sin_theta.sum() == 0:
        return d.mul(1 - t).add(b.mul(t))
    f1 = theta.mul(1 - t).sin().div(sin_theta)
    f2 = theta.mul(t).sin().div(sin_theta)
    return d.mul(f1).add(b.mul(f2))

def apply_influence(d, p, i, m):
    if (p.max(dim=0).values + p.min(dim=0).values.abs()).sum() == 0:
        return d
    if m == "multiply":
        if i == 1:
            return d.mul(p)
        return slerp_score(d, d.mul(p), i)
    if i == 1:
        return p
    return slerp_score(d, p, i)

@torch.no_grad()
def toyx(t, n):
    if n is not None:
        t = t.div(n)
        y, x = t.sin().mul(n), t.cos().mul(n)
    else:
        y, x = t.sin(), t.cos()
    return y, x, None

def ramswap(t, device, to_side):
    if t is None:
        return t
    if str(device) == "cpu" or t.numel() < 10000:
        return t
    if to_side:
        if str(t.device) == "cpu":
            return t
        return t.cpu()
    return t.to(device=device)

@torch.no_grad()
def unified_regression(t, degree, step):
    if degree == 0:
        x = torch.arange(1, t.size(0) + 1, device=t.device).float().unsqueeze(1) # avoid log(0)
        X = torch.cat([torch.ones_like(x), torch.log(x)], dim=1)
    else:
        x = torch.arange(t.size(0), device=t.device).float().unsqueeze(1)
        X = torch.cat([x**i for i in range(degree + 1)], dim=1)
    y = t.view(t.size(0), -1)
    beta = torch.linalg.pinv(X) @ y
    if degree == 0:
        next_x = torch.tensor([[1.0, torch.log(torch.tensor([t.size(0) + step], device=t.device))]], device=t.device)
    else:
        next_x = torch.tensor([[(t.size(0) + step - 1) ** i for i in range(degree + 1)]], device=t.device, dtype=X.dtype)
    pred = next_x @ beta
    return pred.view_as(t[0])

def pythaquickie(y1,x1,y2,x2):
    return ((x1 - x2).pow(2) + (y1 - y2).pow(2)).sqrt()

@torch.no_grad()
def reorganize_2d(y, x, f_index, n_index, influence_method, based, estimated_base_distance, sqrt_base, mean_dist, influence, get_d_mean):
    if based:
        y, yb = y[:-1], y[-1]
        x, xb = x[:-1], x[-1]
    if n_index > 0:
        y, yc = y[:-n_index], y[-n_index:]
        x, xc = x[:-n_index], x[-n_index:]
        n = torch.zeros_like(y)
    if f_index > 0:
        p = torch.zeros_like(y)

    d = torch.zeros_like(y)
    for i in range(y.shape[0]):
        d[i] += pythaquickie(y[i], x[i], y[torch.arange(y.shape[0]) != i], x[torch.arange(x.shape[0]) != i]).sum(dim=0)
    for i in range(f_index):
        p += pythaquickie(y, x, y[i], x[i])
    for i in range(n_index):
        n += pythaquickie(y, x, yc[i], xc[i])
    if n_index > 0:
        del yc, xc
    if mean_dist:
        m = pythaquickie(y, x, y.mean(dim=0), x.mean(dim=0))
    if based:
        db  = pythaquickie(y, x, yb, xb)
        d += db
        if estimated_base_distance < 1:
            db = mmnorm16(db)
            db = (estimated_base_distance - db).abs()
        else:
            db = 1 - mmnorm16(db)
        if sqrt_base:
            db = db.sqrt()
        d, db = d.mul(db), None
    if mean_dist:
        m = 1 - mmnorm16(m)
        d, m = d.mul(m), None
    if f_index > 0 and n_index > 0:
        p = mmnorm16(p, f_index)
        n = mmnorm16(n, n_index)
        p, n = p.sub(n), None
        d, p = apply_influence(d, p, influence, influence_method), None
    elif f_index > 0:
        p = mmnorm16(p)
        d, p = apply_influence(d, p, influence, influence_method), None
    elif n_index > 0:
        n = 1 - mmnorm16(n)
        d, n = apply_influence(d, n, influence, influence_method), None

    d = d.sort(dim=0, descending=True).indices
    y = torch.gather(y, dim=0, index=d)
    x = torch.gather(x, dim=0, index=d)
    if get_d_mean:
        d = d[-1].to(dtype=torch.float32).mean().item()
    else:
        d = None
    return y, x, d

@torch.no_grad()
def reorganize_1d(t, f_index, n_index, influence_method, based, estimated_base_distance, sqrt_base, mean_dist, influence):
    if based:
        t, b = t[:-1], t[-1]
    if n_index > 0:
        t, c = t[:-n_index], t[-n_index:]
        n = torch.zeros_like(t)
    if f_index > 0:
        p = torch.zeros_like(t)

    d = torch.zeros_like(t)
    for i in range(t.shape[0]):
        d[i] += (t[i] - t[torch.arange(t.shape[0]) != i]).pow(2).sum(dim=0)
    for i in range(f_index):
        p += (t - t[i]).pow(2)
    for i in range(n_index):
        n += (t - c[i]).pow(2)
    if n_index > 0:
        del c
    if mean_dist:
        m = (t - t.mean(dim=0)).pow(2)
    if based:
        db = (t - b).pow(2)
        d += db
        if estimated_base_distance < 1:
            db = mmnorm16(db)
            db = (estimated_base_distance - db).abs()
        else:
            db = 1 - mmnorm16(db)
        if sqrt_base:
            db = db.sqrt()
        d, db = d.mul(db), None
    if mean_dist:
        m = 1 - mmnorm16(m)
        d, m = d.mul(m), None
    if f_index > 0 and n_index > 0:
        p = mmnorm16(p, f_index)
        n = mmnorm16(n, n_index)
        p, n = p.sub(n), None
        d, p = apply_influence(d, p, influence, influence_method), None
    elif f_index > 0:
        p = mmnorm16(p)
        d, p = apply_influence(d, p, influence, influence_method), None
    elif n_index > 0:
        n = 1 - mmnorm16(n)
        d, n = apply_influence(d, n, influence, influence_method), None
    return torch.gather(t, dim=0, index=d.sort(dim=0, descending=True).indices)

def finalize(y, x, m, s):
    r = torch.atan2(y, x)
    if m is not None:
        r = r.mul(m)
    return r.view(s)

def linreg(t, positive_models, negative_models, influence_strength, influence_method, use_base_distances, target_base_distance, sqrt_base_distances, use_distances_to_mean, regression, extrapolation_step, loaded_list, projection, always_remag,**k):
    favs = 0
    nega = 0
    degree = regressions[regression]
    based  = loaded_list[-1][1] and use_base_distances
    if positive_models > 0:
        favs = sum([p[1] for p in loaded_list[:positive_models]])
    if negative_models > 0:
        if based:
            nega = sum([p[1] for p in loaded_list[-negative_models:-1]])
        else:
            nega = sum([p[1] for p in loaded_list[-negative_models:]])
    if (t.shape[0] - based - nega) < 3:
        return t[0], None
    if "2D" in projection:
        shape = t[0].shape
        tmax  = max(t.max().item(), t.min().abs().item())
        t = t.view(t.shape[0], -1).to(dtype=torch.float32)
        m = None
        norms = None
        remag = tmax >= pi or always_remag
        if remag:
            norms = t.norm(dim=1, keepdim=True)
        y, x, t  = toyx(t, norms)
        y, x, d = reorganize_2d(y, x, f_index=favs, n_index=nega, influence_method=influence_method, based=based, estimated_base_distance=target_base_distance, sqrt_base=sqrt_base_distances, mean_dist=use_distances_to_mean, influence=influence_strength, get_d_mean=(remag and "scores" in projection))
        if "2D_remag" in projection:
            if remag:
                if based:
                    norms = norms[:-1]
                if nega > 0:
                    norms = norms[:-nega]
                if "mean" in projection:
                    m = norms.mean()
                else:
                    m = get_mag(d, norms)
        if degree == -1:
            if remag and projection == "2D":
                m = pytmag(y[-1], x[-1])
            return finalize(y[-1], x[-1], m, shape), None
        x = unified_regression(t=x, degree=degree, step=extrapolation_step)
        y = unified_regression(t=y, degree=degree, step=extrapolation_step)
        if remag and projection == "2D":
            m = pytmag(y, x)
        return finalize(y, x, m, shape), None
    else:
        t = t.to(dtype=torch.float32)
        t = reorganize_1d(t, f_index=favs, n_index=nega, influence_method=influence_method, based=based, estimated_base_distance=target_base_distance, sqrt_base=sqrt_base_distances, mean_dist=use_distances_to_mean, influence=influence_strength)
        if degree == -1:
            return t[-1], None
        t = unified_regression(t=t, degree=degree, step=extrapolation_step)
        return t, None

regressions = {
    "best score (no extrapolation)": -1,
    "log scale": 0,
    "linear regression": 1,
    "2nd degree polynomial regression": 2,
    "3rd degree polynomial regression": 3,
}
rnames  = [r for r in regressions]
widgets = {
            "required": {
                "positive_models": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "negative_models": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "influence_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 1/20, "round": 1/100}),
                "influence_method": (["interpolate","multiply"],),
                "use_base_distances" : ("BOOLEAN", {"default": True}),
                "target_base_distance": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 1/100, "round": 1/1000}),
                "sqrt_base_distances" : ("BOOLEAN", {"default": True}),
                "use_distances_to_mean" : ("BOOLEAN", {"default": True}),
                "regression": (rnames, {"default": rnames[2]},),
                "extrapolation_step": ("FLOAT", {"default": 1.0, "min": 0, "max": 1000, "step": 1/4, "round": 1e-4}),
                "projection" : (["1D","2D","2D_remag_with_top_scores","2D_remag_with_mean"],),
                "always_remag" : ("BOOLEAN", {"default": True}),
                }
            }

function_properties = {
    "function": linreg,
    "widgets": widgets,
    }

if __name__ == '__main__':
    pass