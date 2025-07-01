import torch
from math import floor

eps = torch.finfo(torch.float32).eps
eps16 = torch.finfo(torch.float16).eps
maxnorm = lambda x: x / x.max(dim=0).values.add(eps)

def cli_graph(d, b, s, n, width, height, key):
    pos = [[floor(x * width / max(d)), floor((1 - y / max(b)) * height), floor(height * (1 - z / max(s)))] for x, y, z in zip(d, b, s)]
    mname = max([len(m) for m in n])
    margin = floor(width / 5)
    whole_graph = ["\n"]
    whole_graph.append(f"X: Pairwise score / Y: Distance from base\nkey: {key}")
    height = height + 6 - min(12, max(6, len(s)))
    for i in range(height):
        line = " " * width
        for p in range(len(pos)):
            if pos[p][1] - 1 == i:
                line = f"{line[:pos[p][0]]}{p + 1}{line[pos[p][0] + len(str(p + 1)):]}"
        sp = (width + margin) - len(line)
        line += sp *" " + "|"
        for p in range(len(pos)):
            if pos[p][2] == i:
                line += f' {p + 1}'
        line = f" |{line}"
        if len(line) > (width + margin * 2):
            line = line[:width + margin * 2]
        whole_graph.append(line)
    whole_graph.append(f' |{"-" * (width + margin)}')
    for i in range(len(s)):
        line = f'{(i + 1):2.0f}: {n[i]} {" " * (mname - len(n[i]))}: X: {pos[i][0]:3.0f} / Y: {height - pos[i][1]:3.0f} / End proportion: {round(100 * s[i] / sum(s), 4):06.4f}%'
        whole_graph.append(line)
    whole_graph = "\n".join(whole_graph)
    print(whole_graph)

@torch.no_grad()
def pairwise_distances(t, pairwise_power, power_after_sum, variance, variance_power,
                        fix_madness, madness_power, base_power_pairwise, base_pw_individual_scale,
                        sine_final_score, sine_final_pi_pow,
                        hide_graph):
    t = t.to(torch.float32).mul(8192)
    d = torch.zeros_like(t[:-1]).to(torch.float32)
    b = torch.zeros_like(t[:-1]).to(torch.float32)
    for i in range(d.shape[0]):
        d[i] = (t[i] - t[:-1]).abs().pow(pairwise_power).sum(dim=0).pow(power_after_sum)
        b[i] = (t[i] - t[-1] ).abs().pow(base_power_pairwise)

    d += b
    d = d.max(dim=0, keepdim=True).values - d
    if base_pw_individual_scale:
        for i in range(b.shape[0]):
            b[i] = (b[i] - b[i].min()) / (b[i].max() - b[i].min() + eps16)

    if variance:
        d *= (t[:-1] - t[:-1].mean(dim=0)).abs().pow(variance_power)

    if fix_madness != "disabled":
        dm = (t[:-1] - t[:-1].mean(dim=[i for i in range(1, t.ndim)], keepdim=True)).abs()
        dm = (dm - dm.mean(dim=0)).abs()
        if fix_madness == "std":
            dm = dm.std(dim=[i for i in range(1, t.ndim)], keepdim=True)
            dm = 1 / dm
        elif fix_madness == "matrix_level":
            dm = (1 / (dm + eps16)).sqrt()
        elif fix_madness == "madder":
            pass
        if madness_power > 1:
            dm = dm.pow(madness_power)
        if fix_madness != "madder":
            dm = dm.add(1 / (t.shape[0] - 1))
        d *= dm
        del dm

    s = d * b
    if sine_final_score:
        s = s.div(s.max(dim=0).values.add(eps16))
        if sine_final_pi_pow > 0:
            s = s.pow(sine_final_pi_pow * torch.pi)
        s = s.mul(torch.pi).sin().clamp(min=0, max=1)

    h = t[0].clone()
    dm = s.max(dim=0).values
    for i in range(d.shape[0]):
        mask = s[i] == dm
        h[mask] = t[i][mask]
    s[t[:-1].sign() != h.sign()] = 0
    r = s.mul(t[:-1]).sum(dim=0)
    ssum = s.sum(dim=0)
    z = ssum <= eps
    r[z] = h[z]
    r[~z] = r[~z].div(ssum[~z])
    s = [k.mean().item() for k in s]
    if hide_graph:
        d = [1]
        b = [1]
    else:
        d = [k.mean().item() for k in d]
        b = [k.mean().item() for k in b]
    t[0] = t[0].div(8192)
    return d, b, s, r.div(8192)

@torch.no_grad()
def check_layer(t, pairwise_power, power_after_sum,
                variance, variance_power,
                fix_madness, madness_power, 
                base_power_pairwise, base_pw_individual_scale,
                sine_final_score, sine_final_pi_pow,
                graph_width, graph_height, hide_graph, **kwargs):
    loaded_list = kwargs['loaded_list']
    if not loaded_list[-1][1] or (t.shape[0] - 1) < 3 or all(torch.equal(t[0], tensor) for tensor in t[1:]):
        return t[0], None
    loaded_list = [m[0].split("\\")[-1].split("/")[-1].replace(".safetensors", "") for m in loaded_list if m[1]]
    layer_key = kwargs['layer_key']
    d, b, s, r = pairwise_distances(t,
                                pairwise_power, power_after_sum,
                                variance, variance_power,
                                fix_madness, madness_power, 
                                base_power_pairwise, base_pw_individual_scale,
                                sine_final_score, sine_final_pi_pow,
                                hide_graph
                                )
    if any([max(k) == 0 for k in [d,b,s]]):
        return t[0], None
    if not hide_graph:
        cli_graph(d, b, s, loaded_list, graph_width, graph_height, layer_key)
    ss = sum(s)
    s = [p / ss for p in s]
    return r, s

widgets = {
            "required": {
                "pairwise_power":  ("INT", {"default": 1, "min": 1, "max": 1000}),
                "power_after_sum":  ("INT", {"default": 2, "min": 1, "max": 1000}),
                "variance" : ("BOOLEAN", {"default": False}),
                "variance_power":   ("INT", {"default": 1, "min": 1, "max": 1000}),
                "fix_madness" : (["disabled", "matrix_level", "std", "madder"], {"default": "matrix_level"},),
                "madness_power":  ("INT", {"default": 2, "min": 1, "max": 1000}),
                "base_power_pairwise":   ("INT", {"default": 2, "min": 1, "max": 1000}),
                "base_pw_individual_scale": ("BOOLEAN", {"default": True}),
                "sine_final_score" : ("BOOLEAN", {"default": True}),
                "sine_final_pi_pow":  ("INT", {"default": 2, "min": 0, "max": 1000}),

                "graph_width":  ("INT", {"default": 80, "min": 10,  "max": 1000}),
                "graph_height": ("INT", {"default": 20,  "min": 10,  "max": 1000}),
                "hide_progress_bar" : ("BOOLEAN", {"default": True}),
                "hide_graph" : ("BOOLEAN", {"default": False}),
                }
            }

function_properties = {
    "function": check_layer,
    "widgets": widgets,
    }
