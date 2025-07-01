import torch

linormat = lambda x, y, z: x.div(torch.linalg.norm(x)).mul(z) if y else x.div(torch.linalg.matrix_norm(x, keepdim=True)).mul(z)

@torch.no_grad()
def merging_function(first_layer, sign_filter, set_to_same_norm, get_layer, total_models, **kwargs):
    r = first_layer.clone()
    sign_last = sign_filter == "same_as_first_model_minus_last"

    if set_to_same_norm:
        eps_b = torch.finfo(r.dtype).eps
        r = r.to(dtype=torch.float32)
        use_linalg = False
        if r.ndim > 1:
            n = torch.linalg.matrix_norm(r, keepdim=True)
        if r.ndim <= 1 or n.mean() <= eps_b or n.isnan().any() or n.isinf().any():
            use_linalg = True
            n = torch.linalg.norm(r)

    if sign_last:
        last = get_layer(total_models - 1)
        sign_last = last is not None
        if sign_last and set_to_same_norm:
            last = last.to(torch.float32)
            last = linormat(last, use_linalg, n)
        diffsign = (first_layer - last).sign()
    total_changed = 0
    for i in range(1, total_models - sign_last):
        t = get_layer(i)
        if t is not None:
            if set_to_same_norm:
                t = linormat(t.to(torch.float32), use_linalg, n)
            mabmask = t.abs() > r.abs()
            if sign_filter != "none":
                if sign_last:
                    mabmask = mabmask & ((t - last).sign() == diffsign)
                elif sign_filter in ["same_as_first_model", "same_as_first_model_minus_last"]: #in case last is not loaded/doesn't have the layer
                    mabmask = mabmask & (t.sign() == first_layer.sign())
            r[mabmask] = t[mabmask]
            total_changed += mabmask.sum().item()
    # print(f"\n{round(100 * total_changed / r.numel(), 2):5.2f}%")
    return r, None

widgets = {
            "required": {
                "sign_filter" : (["none","same_as_first_model", "same_as_first_model_minus_last"], {"default": "none"}),
                "set_to_same_norm" : ("BOOLEAN", {"default": False}),
                }
            }

function_properties = {
    "function": merging_function,
    "widgets": widgets,
    'is_low_memory': True,
    }
