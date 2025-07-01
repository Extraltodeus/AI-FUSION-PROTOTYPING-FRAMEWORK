import torch
from random import randint
from tqdm   import tqdm
from math import log

eps16  = torch.finfo(torch.float16).eps
eps32  = torch.finfo(torch.float32).eps
hasnan = lambda x: x.isnan().any() or x.isinf().any()
mmnorm = lambda x: (x - x.min(dim=0).values) / (x.max(dim=0).values - x.min(dim=0).values).add(eps16)
mmsiat = lambda x: (x - x.min()) / (x.max() - x.min()).add(torch.finfo(x.dtype).eps)
misiat = lambda x: x / x.max().add(torch.finfo(x.dtype).eps)
nimask = lambda x: x.isinf() | x.isnan()
rlen   = lambda x: range(len(x))
fylter = lambda f, x: type(x)(filter(f, x))
stninf = lambda a, b, c, d: (c > d) & ~nimask(d)
btninf = lambda a, b, c, d: (c < d) & ~nimask(d)
stbsni = lambda a, b, c, d: (c > d) & (b.abs() > a.abs()) & ~nimask(d)
stssni = lambda a, b, c, d: (c > d) & (b.abs() < a.abs()) & ~nimask(d)
pytmag = lambda y, x: (y.pow(2) + x.pow(2)).sqrt()
atnyxm = lambda y, x: torch.atan2(y, x).mul(pytmag(y, x))

update_rules = {
    "progressive": lambda a, b, c, d: ~nimask(d),
    "lower_error": stninf,
    "higher_error": btninf,
    "lower_error_and_bigger_slope" : stbsni,
    "lower_error_and_smaller_slope": stssni,
    }

def smtreplace(b1, b2, d1, d2, i, rule):
    m = update_rules[rule](b1, b2, d1, d2)
    if rule == "progressive":
        p  = 1 / (i + 2)
        d3 = torch.zeros_like(d2)
        d3[m] = misiat(d2[m])
        b1[m] = b1[m] * (1 - p * (1 - d3[m])) + b2[m] * p * (1 - d3[m])
        d1[m] = d1[m] * (1 - p) + d2[m] * p * d3[m] # The intent is to let it be useful or something.
        return [b1, d1]
    b1[m] = b2[m]
    d1[m] = d2[m]
    return [b1, d1]

def beta_delta(y, X, pos_amp=0):
    beta  = torch.linalg.pinv(X) @ y
    trend = X @ beta
    if pos_amp == 0:
        delta = (trend - y).pow(2).sum(dim=0)
    else:
        delta = (trend - y).pow(2)
        delta = delta.sum(dim=0).mul(delta[-pos_amp:].sum(dim=0))
    return [beta, delta.unsqueeze(0).repeat(beta.shape[0], 1)]

def get_X(t, degree):
    if degree <= 0:
        x = torch.arange(1, t.size(0) + 1, device=t.device).float().unsqueeze(1) # avoid log(0)
        X = torch.cat([torch.ones_like(x), torch.log(x)], dim=1)
        if degree < 0:
            X = X.flip(dims=(0,))
            X[...,1:] = (X[...,1:].max() - X[...,1:])
    else:
        x = torch.arange(t.size(0), device=t.device).float().unsqueeze(1)
        X = torch.cat([x**i for i in range(degree + 1)], dim=1)
    return X

def get_next_x(tsize, device, dtype, degree, step):
    if degree < 0 and step < 1:
        next_x = torch.tensor([[1.0, log(tsize) - torch.log(torch.tensor([tsize * (1 - step)], device=device))]], device=device)
    elif degree <= 0:
        next_x = torch.tensor([[1.0, torch.log(torch.tensor([tsize * step], device=device))]], device=device)
    else:
        next_x = torch.tensor([[(tsize * step) ** i for i in range(degree + 1)]], device=device, dtype=dtype)
    return next_x

@torch.no_grad()
def toyx(t, n):
    if n is not None:
        t = t.div(n)
        y, x = t.sin().mul(n), t.cos().mul(n)
    else:
        y, x = t.sin(), t.cos()
    return y, x, None

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

def apply_influence(d, p):
    if (p.max(dim=0).values + p.min(dim=0).values.abs()).sum() == 0:
        return d
    return slerp_score(d, d.mul(p), 0.5)

@torch.no_grad()
def reorganize(t, f_index, n_index, based, sqrt_base, mean_dist, estimated_base_distance=1):
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
            db = mmnorm(db)
            db = (estimated_base_distance - db).abs()
        else:
            db = 1 - mmnorm(db)
        if sqrt_base:
            db = db.sqrt()
        d, db = d.mul(db), None
    if mean_dist:
        m = 1 - mmnorm(m)
        d, m = d.mul(m), None
    if f_index > 0 and n_index > 0:
        p = mmnorm(p)
        n = mmnorm(n)
        p, n = p.sub(n), None
        d, p = apply_influence(d, p), None
    elif f_index > 0:
        p = mmnorm(p)
        d, p = apply_influence(d, p), None
    elif n_index > 0:
        n = 1 - mmnorm(n)
        d, n = apply_influence(d, n), None

    t = torch.gather(t, dim=0, index=d.sort(dim=0, descending=True).indices)
    return t

def make_extra_infos(delta_a, layer_key, save_delta, is_cropped, s, original_shape=None, need_dim=True):
    e_inf = {"layer_key": layer_key}
    if need_dim:
        delta_a = delta_a[0]
    if save_delta != "disabled":
        if save_delta == "float16":
            delta_a = delta_a.to(dtype=torch.float16)
        elif save_delta == "uint8":
            delta_a = mmsiat(delta_a).mul(256).to(dtype=torch.uint8)
        e_inf['delta'] = delta_a.to(device="cpu").view(s)
        if is_cropped:
            e_inf['delta'] = e_inf['delta'].unsqueeze(0)
            e_inf['original_shape'] = original_shape
    return e_inf

@torch.no_grad()
def linreg_score(t, positive_models, negative_models, last_is_base, base_sqrt, use_mean_difference, regression_scale, save_delta, loaded_list, layer_key, is_cropped, original_shape, **k):
    posi, nega = 0, 0
    if positive_models > 0:
        posi = sum([p[1] for p in loaded_list[:positive_models]])
    based = loaded_list[-1][1] and last_is_base
    if negative_models > 0:
        if based:
            nega = sum([p[1] for p in loaded_list[-negative_models:-1]])
        else:
            nega = sum([p[1] for p in loaded_list[-negative_models:]])
    if hasnan(t) or not t[0].is_floating_point() or (t.shape[0] - based - nega) < 3 or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        if is_cropped:
            return torch.zeros_like(t[0]).to(dtype=torch.bool), None
        else:
            return torch.tensor([0]).to(dtype=torch.bool), None
    s = t[0].shape
    t = t.view(t.shape[0], -1).to(dtype=torch.float32)
    t = reorganize(t, posi, nega, based, sqrt_base=base_sqrt, mean_dist=use_mean_difference)
    result_a, delta_a = beta_delta(t, get_X(t, regressions[regression_scale]))
    result_a = [result_a[i].view(s) for i in range(result_a.shape[0])]
    if not is_cropped:
        result_a = torch.stack(result_a)
    return result_a, make_extra_infos(delta_a.div(t.size(0)), layer_key, save_delta, is_cropped, s, original_shape)

@torch.no_grad()
def linreg_shuffle(t, n_shuffles, shuffle_order, positive_models, sticky_positive, last_is_base, regression_scale, update_rule, save_delta, seed,
                   loaded_list, layer_key, is_cropped, negative_models, original_shape, return_result=False, **k):

    posi, nega = 0, 0

    if positive_models > 0:
        posi = sum([p[1] for p in loaded_list[:positive_models]])

    based = loaded_list[-1][1] and last_is_base

    if negative_models > 0:
        if based:
            nega = sum([p[1] for p in loaded_list[-negative_models:-1]])
        else:
            nega = sum([p[1] for p in loaded_list[-negative_models:]])

    if not return_result and \
        hasnan(t) or not t[0].is_floating_point() or (t.shape[0] - based - nega) < 3 or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        if is_cropped:
            return torch.zeros_like(t[0]).to(dtype=torch.bool), None
        else:
            return torch.tensor([0]).to(dtype=torch.bool), None

    posia  = sticky_positive * posi
    degree = regressions[regression_scale]

    s = t[0].shape
    t = t.view(t.shape[0], -1).to(dtype=torch.float32)
    X = get_X(t, degree)
    result_a, delta_a = beta_delta(t.flip(dims=(0,)), X, pos_amp=posia)
    tnum = t.size(0)

    if n_shuffles == 0:
        result_a = [result_a[i].view(s) for i in range(result_a.shape[0])]
        if not is_cropped:
            result_a = torch.stack(result_a)
        return result_a, make_extra_infos(delta_a.div(tnum), layer_key, save_delta, is_cropped, s, original_shape)

    torch.manual_seed(seed)

    if based:
        t, b = t[:-1], t[-1].unsqueeze(0)
        loaded_list = loaded_list[:-1]

    if posi > 0:
        t, a = t[posi:], t[:posi]
        loaded_list = loaded_list[posi:]

    if nega > 0:
        t, n = t[:-nega], t[-nega:]

    for i in range(n_shuffles):
        shuffled = [t[torch.randperm(t.size(0))]]
        if nega > 0:
            if nega > 1:
                n = n[torch.randperm(n.size(0))]
            shuffled = [n] + shuffled
        if based:
            shuffled = [b] + shuffled
        if len(shuffled) == 3 and ((i % 2 == 0 and shuffle_order == "alternate") or shuffle_order == "nbmp"):
            shuffled[0], shuffled[1] = shuffled[1], shuffled[0]
        if posi > 0:
            if posi > 1:
                a = a[torch.randperm(a.size(0))]
            shuffled = shuffled + [a]

        result_b, delta_b = beta_delta(torch.cat(shuffled), X, pos_amp=posia)
        result_a, delta_a = smtreplace(result_a, result_b, delta_a, delta_b, i, update_rule)

    if return_result:
        return result_a, delta_a

    result_a = [result_a[i].view(s) for i in range(result_a.shape[0])]

    if not is_cropped:
        result_a = torch.stack(result_a)

    return result_a, make_extra_infos(delta_a.div(tnum), layer_key, save_delta, is_cropped, s, original_shape)

def custom_save_function(extras_infos, result, model_list, output_name, function_arguments, save_path, metadata, save_function, recombine_split, **k):
    save_delta = function_arguments.get("save_delta", None) != "disabled"
    delta_keyword  = "delta" if function_arguments['hidden'] != "multi" else "trend_index"
    done_keys  = []
    for i in tqdm(rlen(extras_infos), desc="Finalizing"):
        if extras_infos[i] is not None:
            if isinstance(extras_infos[i], list):
                key = extras_infos[i][0]["layer_key"]
                done_keys.append(key)
                if save_delta:
                    if extras_infos[i][0].get('delta', None) is None:
                        continue
                    delta_key = f"{delta_keyword}.{key}"
                    dimz = extras_infos[i][0]["delta"].shape[0]
                    original_shape = extras_infos[i][0]["original_shape"]
                    batch = len(extras_infos[i])
                    rez = [[] for _ in range(dimz)]
                    dimrez = []
                    for o in range(dimz):
                        rez[o] = []
                        for b in range(batch):
                            rez[o].append([extras_infos[i][b]["delta"][o]])
                        dimrez.append(recombine_split(rez[o], original_shape)[0])
                    if len(dimrez) > 1:
                        delta, dimrez = torch.stack(dimrez), None
                    else:
                        delta, dimrez = dimrez[0], None
            else:
                key = extras_infos[i]['layer_key']
                done_keys.append(key)
                if save_delta:
                    if extras_infos[i].get('delta', None) is None:
                        continue
                    delta, delta_key, extras_infos[i] = extras_infos[i]['delta'], f"{delta_keyword}.{key}", None
            if save_delta:
                result[delta_key] = delta.cpu()
                done_keys.append(delta_key)

    rkeys = [k for k in result]
    for i in tqdm(rlen(rkeys), desc="Preparing save"):
        k = rkeys[i]
        if result[k].dtype == torch.bool or k not in done_keys:
            result.pop(k)

    if function_arguments["hidden"] == "scored":
        nummodels = len(model_list) - function_arguments["last_is_base"] - function_arguments["negative_models"]
        result["num_models_used"] = torch.tensor([nummodels], dtype=torch.float32)
    elif function_arguments["hidden"] == "multi":
        result["num_models_used"] = torch.tensor([function_arguments['preparation_result']['num_models']], dtype=torch.float32)
    else:
        result["num_models_used"] = torch.tensor([len(model_list)], dtype=torch.float32)
    result["scale_degree"] = torch.tensor([regressions[function_arguments['regression_scale']]], dtype=torch.float32)

    return result

def pythaquickie(y1,x1,y2,x2):
    return ((x1 - x2).pow(2) + (y1 - y2).pow(2)).sqrt()

@torch.no_grad()
def reorganize_2d(y, x, f_index, n_index, influence_method, based, estimated_base_distance, sqrt_base, mean_dist, influence):
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
            db = mmnorm(db)
            db = (estimated_base_distance - db).abs()
        else:
            db = 1 - mmnorm(db)
        if sqrt_base:
            db = db.sqrt()
        d, db = d.mul(db), None
    if mean_dist:
        m = 1 - mmnorm(m)
        d, m = d.mul(m), None
    if f_index > 0 and n_index > 0:
        p = mmnorm(p, f_index)
        n = mmnorm(n, n_index)
        p, n = p.sub(n), None
        d, p = apply_influence(d, p, influence, influence_method), None
    elif f_index > 0:
        p = mmnorm(p)
        d, p = apply_influence(d, p, influence, influence_method), None
    elif n_index > 0:
        n = 1 - mmnorm(n)
        d, n = apply_influence(d, n, influence, influence_method), None

    d = d.sort(dim=0, descending=True).indices
    y = torch.gather(y, dim=0, index=d)
    x = torch.gather(x, dim=0, index=d)
    return y, x

@torch.no_grad()
def linreg_score_2d(t, positive_models, negative_models,
                 influence_strength, influence_method, 
                 last_is_base, sqrt_base_distances,
                 use_distances_to_mean, regression_scale,
                 save_delta, original_shape, is_cropped,
                 loaded_list, layer_key, **k):
    posi, nega = 0, 0
    if positive_models > 0:
        posi = sum([p[1] for p in loaded_list[:positive_models]])
    based = loaded_list[-1][1] and last_is_base

    if negative_models > 0:
        if based:
            nega = sum([p[1] for p in loaded_list[-negative_models:-1]])
        else:
            nega = sum([p[1] for p in loaded_list[-negative_models:]])
    if hasnan(t) or not t[0].is_floating_point() or (t.shape[0] - based - nega) < 3 or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        if is_cropped:
            return torch.zeros_like(t[0]).to(dtype=torch.bool), None
        else:
            return torch.tensor([0]).to(dtype=torch.bool), None

    s = t[0].shape
    tnum = t.size(0)
    t = t.view(t.shape[0], -1).to(dtype=torch.float32)
    norms = t.norm(dim=1, keepdim=True)
    y, x, t  = toyx(t, norms)
    y, x = reorganize_2d(y, x, f_index=posi, n_index=nega, influence_method=influence_method, based=based,
                            estimated_base_distance=1, sqrt_base=sqrt_base_distances,
                            mean_dist=use_distances_to_mean, influence=influence_strength)
    x_axis = get_X(y, regressions[regression_scale])
    y, yd = beta_delta(y, x_axis)
    x, xd = beta_delta(x, x_axis)
    delta_a, yd, xd = yd + xd, None, None
    result_a, y, x = [y[i].view(s) for i in range(y.shape[0])] + [x[i].view(s) for i in range(x.shape[0])], None, None
    # result_a, y, x = [torch.atan2(y[i], x[i]).view(s) for i in range(y.shape[0])], None, None
    if not is_cropped:
        result_a = torch.stack(result_a)
    return result_a, make_extra_infos(delta_a.div(tnum * 2), layer_key, save_delta, is_cropped, s, original_shape)

def generate_model_2d(first_layer, trend_scale, apply_to, get_layer, regression_scale, slope_topk, slope_topk_value,
                   delta_error_topk, delta_error_topk_value, preparation_result, layer_key, **k):
    degree = preparation_result['degree'] if regression_scale == "auto" else regressions[regression_scale]

    trend = get_layer(1)
    if trend is None:
        return first_layer, None

    s = first_layer.shape
    device = first_layer.device
    first_layer = first_layer.view(1, -1).to(torch.float32)
    mag = first_layer.norm()
    trend = trend.view(trend.shape[0], -1)

    if trend.dtype != torch.float32:
        trend = trend.to(dtype=torch.float32)

    tsi = trend.shape[0]
    y, x, trend = trend[:tsi // 2], trend[tsi // 2:], None
    yb, xb, y, x = y[:1], x[:1], y[1:], x[1:]

    mask, smask, dmask = [None for _ in range(3)]
    smult, dmult = None, None
    if slope_topk in ["smallest", "biggest"] and 0 < slope_topk_value < 1:
        smask = topk_mask(atnyxm(y, x).abs(), slope_topk_value, biggest=slope_topk=="biggest")
    elif slope_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
        smult = mmsiat(atnyxm(y, x).abs())
        if slope_topk == "proportional_big_is_less":
            smult = 1 - smult
        if 0 < slope_topk_value < 1:
            smult = smult * slope_topk_value + 1 - slope_topk_value
    if (delta_error_topk in ["smallest", "biggest"] and 0 < delta_error_topk_value < 1)\
        or delta_error_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
        delta = get_layer(1, override_key=f"delta.{layer_key}")
        if delta is not None:
            delta = delta.view(-1).unsqueeze(0)
            if delta_error_topk in ["smallest", "biggest"]:
                dmask = topk_mask(delta, delta_error_topk_value, biggest=delta_error_topk=="biggest")
            elif delta_error_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
                if delta.dtype == torch.uint8:
                    delta = delta.to(torch.float16)
                dmult = mmsiat(delta)
                if delta_error_topk == "proportional_big_is_less":
                    dmult = 1 - dmult
                if 0 < delta_error_topk_value < 1:
                    dmult = dmult * delta_error_topk_value + 1 - delta_error_topk_value

    masks = sum([d is None for d in [dmask, smask]])
    if masks == 0:
        mask, dmask, smask = dmask & smask, None, None
    elif dmask is not None:
        mask, dmask = dmask, None
    elif smask is not None:
        mask, smask = smask, None

    if masks < 2:
        maskend = first_layer[~mask]

    if apply_to == "first model":
        yb, xb, first_layer = toyx(first_layer, mag)

    if dmult is not None:
        y, x = y * dmult, x * dmult
    if smult is not None:
        y, x = y * smult, x * smult
    if trend_scale < 0:
        y, x = -y, -x

    y, x = torch.cat([yb, y]), torch.cat([xb, x])

    nx = max(1, preparation_result['x_axis'])
    next_x = get_next_x(nx, device, torch.float32, degree=degree, step=abs(trend_scale))
    y = (next_x @ y)
    x = (next_x @ x)
    r = torch.atan2(y, x).mul(mag)
    if mask is not None:
        r[~mask] = maskend
    return r.view(s), None

def topk_mask(t, v, biggest):
    d = torch.zeros_like(t, dtype=torch.bool).view(-1)
    k = int(t.numel() * v)
    d[t.view(-1).topk(k, largest=biggest).indices] = True
    return d.view(t.shape)

def generate_model(first_layer, trend_scale, apply_to, get_layer, regression_scale, slope_topk, slope_topk_value,
                   delta_error_topk, delta_error_topk_value, preparation_result, layer_key, total_models, **k):
    degree = preparation_result['degree'] if regression_scale == "auto" else regressions[regression_scale]

    trend = get_layer(total_models - 1)
    if trend is None:
        return first_layer, None

    s = first_layer.shape
    trend = trend.view(trend.shape[0], -1)

    if trend.dtype != torch.float32:
        trend = trend.to(dtype=torch.float32)

    base, trend = trend[:1], trend[1:]

    mask, smask, dmask = [None for _ in range(3)]
    smult, dmult = None, None
    if slope_topk in ["smallest", "biggest"] and 0 < slope_topk_value < 1:
        smask = topk_mask(trend.abs(), slope_topk_value, biggest=slope_topk=="biggest")
    elif slope_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
        smult = mmsiat(trend.abs())
        if slope_topk == "proportional_big_is_less":
            smult = 1 - smult
        if 0 < slope_topk_value < 1:
            smult = smult * slope_topk_value + 1 - slope_topk_value
    if (delta_error_topk in ["smallest", "biggest"] and 0 < delta_error_topk_value < 1)\
        or delta_error_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
        delta = get_layer(1, override_key=f"delta.{layer_key}")
        if delta is not None:
            delta = delta.view(-1).unsqueeze(0)
            if delta_error_topk in ["smallest", "biggest"]:
                dmask = topk_mask(delta, delta_error_topk_value, biggest=delta_error_topk=="biggest")
            elif delta_error_topk in ["proportional_big_is_less", "proportional_big_is_more"]:
                if delta.dtype == torch.uint8:
                    delta = delta.to(torch.float16)
                dmult = mmsiat(delta)
                if delta_error_topk == "proportional_big_is_less":
                    dmult = 1 - dmult
                if 0 < delta_error_topk_value < 1:
                    dmult = dmult * delta_error_topk_value + 1 - delta_error_topk_value

    masks = sum([d is None for d in [dmask, smask]])
    if masks < 2 or apply_to == "first model":
        first_layer = first_layer.view(1, -1).to(dtype=torch.float32)
    if masks == 0:
        mask, dmask, smask = dmask & smask, None, None
    elif dmask is not None:
        mask, dmask = dmask, None
    elif smask is not None:
        mask, smask = smask, None
    if dmult is not None:
        trend = trend * dmult
    if smult is not None:
        trend = trend * smult
    if trend_scale < 0:
        trend = -trend
    if apply_to == "first model":
        trend = torch.cat([first_layer, trend])
    else:
        trend = torch.cat([base, trend])
    nx = max(1, preparation_result['x_axis'])
    next_x = get_next_x(nx, first_layer.device, torch.float32, degree=degree, step=abs(trend_scale)) # * ((nx - 1) / nx)
    r = (next_x @ trend)
    if mask is not None:
        r[~mask] = first_layer[~mask]
    return r.view(s), None

def generate_model_multi(first_layer, apply_to, get_layer, regression_scale, preparation_result, layer_key, trend_scale1, trend_scale2, trend_scale3, **k):
    scales = [trend_scale1, trend_scale2, trend_scale3]
    degree = preparation_result['degree'] if regression_scale == "auto" else regressions[regression_scale]

    trend = get_layer(1)
    if trend is None:
        return first_layer, None

    s = first_layer.shape
    trend = trend.view(trend.shape[0], -1)

    if trend.dtype != torch.float32:
        trend = trend.to(dtype=torch.float32)

    base, trend = trend[:1], trend[1:]

    trend_index = get_layer(1, override_key=f"trend_index.{layer_key}")
    if trend_index is None:
        return first_layer, None

    trend_index = trend_index.view(1, -1).repeat(trend.shape[0], 1)
    for i in range(trend_index.max() + 1):
        if scales[i] < 0:
            mask = trend_index == i
            trend[mask] = -trend[mask]

    if apply_to == "first model":
        first_layer = first_layer.view(1, -1).to(dtype=torch.float32)
        trend = torch.cat([first_layer, trend])
    else:
        trend = torch.cat([base, trend])

    results = []
    for i in range(trend_index.max() + 1):
        next_x = get_next_x(max(1, preparation_result['x_axis'][i]), first_layer.device, torch.float32, degree=degree, step=abs(scales[i]))
        results.append(next_x @ trend)
    r = torch.zeros_like(results[0])
    trend_index = trend_index[0].unsqueeze(0)
    for i in range(trend_index.max() + 1):
        mask = trend_index == i
        r[mask] = results[i][mask]
    return r.view(s), None

def prepare_create(models_found, load_function, **a):
    x = load_function(models_found[-1][1], "num_models_used", device="cpu")
    if x is not None:
        if x.numel() > 1:
            x = [i.item() for i in x.view(-1)]
        else:
            x = x.item()
    else:
        x = 10
    d = load_function(models_found[-1][1], "scale_degree", device="cpu")
    d = 0 if d is None else d.item()
    return {"x_axis": x, "degree": int(d)}

def multi_trend(t, n_shuffles, seed, update_rule, regression_scale, loaded_list, last_is_base, preparation_result, original_shape, is_cropped, layer_key, positive_models1, positive_models2, negative_models1, negative_models2, negative_models3=0, positive_models3=0, **kwargs):
    if hasnan(t) or not t[0].is_floating_point() or (t.shape[0]) < 3 or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        if is_cropped:
            return torch.zeros_like(t[0]).to(dtype=torch.bool), None
        else:
            return torch.tensor([0]).to(dtype=torch.bool), None

    s = t[0].shape
    t = t.view(t.shape[0], -1).to(dtype=torch.float32)
    pos = [positive_models1, positive_models2, positive_models3]
    neg = [negative_models1, negative_models2, negative_models3]
    based = last_is_base and loaded_list[-1][1]
    if based:
        t, b = t[:-1], t[-1].unsqueeze(0)
        loaded_list = loaded_list[:-1]
    prepared = len(preparation_result['mdlist'])
    names   = [n[0] for n in loaded_list if n[1]]
    ordered = [[] for _ in range(prepared)]
    loadeds = [[] for _ in range(prepared)]
    diffs   = [None for _ in range(prepared)]
    for i in range(prepared):
        good = []
        for k in preparation_result['mdlist'][i]:
            for j in range(len(names)):
                if k == names[j]:
                    good.append(k)
                    ordered[i].append(t[j])
                    break
            loadeds[i].append([k, k in good])
        if len(ordered[i]) < 3:
            if is_cropped:
                return torch.zeros_like(t[0]).to(dtype=torch.bool).view(s), None
            else:
                return torch.tensor([0]).to(dtype=torch.bool), None
        ordered[i] = torch.stack(ordered[i])
        if based:
            ordered[i] = torch.cat([ordered[i], b])
            loadeds[i].append(["", True])
        ordered[i], diffs[i] = linreg_shuffle(ordered[i], return_result=True, n_shuffles=n_shuffles,
                                              last_is_base=based, regression_scale=regression_scale,
                                              update_rule=update_rule, sticky_positive=False, seed=seed, layer_key=layer_key,
                                              loaded_list=loadeds[i],  positive_models=pos[i],
                                              shuffle_order="bnmp",    negative_models=neg[i], save_delta=False, is_cropped=False,
                                              original_shape=original_shape, **kwargs)
    if None in diffs:
        if is_cropped:
            return torch.zeros_like(t[0]).to(dtype=torch.bool).view(s), None
        else:
            return torch.tensor([0]).to(dtype=torch.bool), None

    for h in rlen(diffs):
        diffs[h] = diffs[h][0]

    index = torch.zeros_like(diffs[0], dtype=torch.uint8)
    res_beta = torch.zeros_like(ordered[0])
    infmask  = torch.zeros_like(diffs[0])
    for h in rlen(diffs):
        diffs[h] += infmask
        tkm = topk_mask(diffs[h], v=1/len(ordered), biggest=False) & ~nimask(diffs[h])
        infmask[tkm] += torch.inf
        for d in range(res_beta.shape[0]):
            res_beta[d][tkm] = ordered[h][d][tkm]
        index[tkm] = h

    res_beta = [res_beta[i].view(s) for i in range(res_beta.shape[0])]

    if not is_cropped:
        res_beta = torch.stack(res_beta)

    return res_beta, make_extra_infos(index, layer_key=layer_key, is_cropped=is_cropped, s=s, original_shape=original_shape, save_delta=True, need_dim=False)

def format_name(n):
    return n.split("/")[-1].split("\\")[-1].replace(".safetensors", "")

def prepare_multi(model_list1, model_list2, **a):
    models_lists = [model_list1, model_list2]
    for k, v in a.items():
        if "model_list" in k and v:
            models_lists.append(v)
    nmu = []
    for i in range(len(models_lists)):
        models_lists[i] = [format_name(n) for n in models_lists[i]]
        nmu.append(len(models_lists[i]))
        # models_lists[i] = models_lists[i][::-1]
    return {"mdlist": models_lists, "num_models": nmu}

regressions = {
    "invert_log": -1,
    "log": 0,
    "linear": 1,
    "2nd degree polynomial": 2,
    "3rd degree polynomial": 3,
}

rnames = [r for r in regressions]
widgets1 = {
            "required": {
                "n_shuffles": ("INT", {"default": 6, "min": 0, "max": 1e3}),
                "positive_models": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "sticky_positive": ("BOOLEAN", {"default": False, "tooltip":"multiply the delta by the delta relative to the positive models. Increasing it's importance."}),
                "negative_models": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "last_is_base": ("BOOLEAN", {"default": True}),
                "shuffle_order": (["bnmp", "nbmp", "alternate"], {"tooltip":"n: negative\nb: base\nm: middle\np: positive."}),
                "regression_scale": (rnames, {"default": rnames[2]},),
                "update_rule": ([u for u in update_rules],),
                "save_delta": (["disabled","float16","float32","uint8"],),
                "seed": ("INT", {"default": randint(0, 1e9), "min": 0, "max": 1e9}),
                }
            }
widgets2 = {
            "required": {
                "positive_models": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "negative_models": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "last_is_base": ("BOOLEAN", {"default": True}),
                "base_sqrt": ("BOOLEAN", {"default": False}),
                "use_mean_difference": ("BOOLEAN", {"default": True}),
                "regression_scale": (rnames, {"default": rnames[2]},),
                "save_delta": (["disabled","float16","float32","uint8"],),
                }
            }
widgets3 = {
            "required": {
                "trend_scale": ("FLOAT", {"default": 1.0, "min": -1000, "max": 1000, "step": 1e-2, "round": 1e-4}),
                "apply_to": (["trend base", "first model"],),
                "regression_scale": (["auto"] + rnames,),
                "slope_topk": (["disabled","smallest","biggest","proportional_big_is_less","proportional_big_is_more"], {"default": "disabled"},),
                "slope_topk_value": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 1e-2, "round": 1e-4}),
                "delta_error_topk": (["disabled","smallest","biggest","proportional_big_is_less","proportional_big_is_more"], {"default": "disabled"},),
                "delta_error_topk_value": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 1e-2, "round": 1e-4}),
                }
            }
widgets4 = {
            "required": {
                "n_shuffles": ("INT", {"default": 6, "min": 3, "max": 1e3}),
                "last_is_base": ("BOOLEAN", {"default": True}),
                "regression_scale": (rnames, {"default": rnames[2]},),
                "update_rule": ([u for u in update_rules],),
                "seed": ("INT", {"default": randint(0, 1e9), "min": 0, "max": 1e9}),
                "model_list1": ("MDLIST",),
                "positive_models1": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "negative_models1": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "model_list2": ("MDLIST",),
                "positive_models2": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "negative_models2": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                },
            "optional": {
                "model_list3": ("MDLIST",),
                "positive_models3": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                "negative_models3": ("INT", {"default": 3, "min": 0, "max": 1e3}),
                }
            }
widgets5 = {
            "required": {
                "trend_scale1": ("FLOAT", {"default": 1.0, "min": -1000, "max": 1000, "step": 1e-2, "round": 1e-4}),
                "trend_scale2": ("FLOAT", {"default": 1.0, "min": -1000, "max": 1000, "step": 1e-2, "round": 1e-4}),
                "trend_scale3": ("FLOAT", {"default": 1.0, "min": -1000, "max": 1000, "step": 1e-2, "round": 1e-4}),
                "apply_to": (["trend base", "first model"],),
                "regression_scale": (["auto"] + rnames,),
                }
            }
widgets6 = {
            "required": {
                "positive_models": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "negative_models": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "influence_strength": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 1/20, "round": 1/100}),
                "influence_method": (["interpolate","multiply"],),
                "last_is_base" : ("BOOLEAN", {"default": True}),
                "sqrt_base_distances" : ("BOOLEAN", {"default": True}),
                "use_distances_to_mean" : ("BOOLEAN", {"default": True}),
                "regression_scale": (rnames, {"default": rnames[2]},),
                "save_delta": (["disabled","float16","float32","uint8"],),
                }
            }

V_NAME = 9

function_properties = [
    {
    "function": linreg_shuffle,
    "widgets" : widgets1,
    "custom_save_function": custom_save_function,
    "do_not_add_skipped": True,
    "name": f"Trend model v{V_NAME} - create standard shuffled weights",
    },
    {
    "function": linreg_score,
    "widgets" : widgets2,
    "custom_save_function": custom_save_function,
    "do_not_add_skipped": True,
    "name": f"Trend model v{V_NAME} - create standard scored weights",
    "hidden": "scored"
    },
    {
    "function": generate_model,
    "widgets" : widgets3,
    "preparation_function": prepare_create,
    "name": f"Trend model v{V_NAME} - use standard weights",
    "is_low_memory": True,
    },
    {
    "function": multi_trend,
    "widgets" : widgets4,
    "custom_save_function": custom_save_function,
    "preparation_function": prepare_multi,
    "do_not_add_skipped": True,
    "name": f"Trend model v{V_NAME} - create multi trend weights",
    "hidden": "multi"
    },
    {
    "function": generate_model_multi,
    "widgets" : widgets5,
    "preparation_function": prepare_create,
    "name": f"Trend model v{V_NAME} - use multi trend weights",
    "is_low_memory": True,
    },
    {
    "function": linreg_score_2d,
    "widgets" : widgets6,
    "custom_save_function": custom_save_function,
    "do_not_add_skipped": True,
    "name": f"Trend model v{V_NAME} - create 2D scored weights",
    "hidden": "scored"
    },
    {
    "function": generate_model_2d,
    "widgets" : widgets3,
    "preparation_function": prepare_create,
    "name": f"Trend model v{V_NAME} - use 2D weights",
    "is_low_memory": True,
    },
]

if __name__ == '__main__':
    regscale = "linear"
    regscale = "2nd degree polynomial"
    upr = "lower_error"
    upr = "progressive"
    t = torch.torch.rand(9,6)
    for j in range(t.shape[0]):
        i = j + 0.5
        t[j] = t[j] * 0 + i ** 2
    t = t.flip(dims=(0,))
    print(t)
    r, e = linreg_shuffle(t, n_shuffles=32, shuffle_order="bnmp", positive_models=1, sticky_positive=False, last_is_base=False, regression_scale=regscale, update_rule=upr, save_delta=False, seed=42,
                    loaded_list=[["", True] for _ in range(t.shape[0])], layer_key="fds", is_cropped=False, negative_models=0, original_shape=None, return_result=False)
    # print(r)
    next_x = get_next_x(t.shape[0], t.device, torch.float32, degree=2, step=1)
    result = next_x @ r
    print(result)