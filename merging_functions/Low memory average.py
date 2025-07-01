import torch

@torch.no_grad()
def megaAverage(first_layer, optional_weights, get_layer, total_models, **kwargs):
    if optional_weights != "":
        weights = [float(o.replace("\n","")) for o in optional_weights.split(",") if o != ""]
    else:
        weights = [1 for _ in range(total_models)]
    was_loaded = [weights[0]]
    result = first_layer.to(dtype=torch.float32).mul(weights[0])
    for i in range(1, total_models):
        t = get_layer(i)
        if t is not None:
            result += t.to(dtype=torch.float32).mul(weights[i])
            was_loaded.append(weights[i])
    result = result.div(sum(was_loaded))
    return result, None

widgets = {
            "required": {
                "optional_weights": ("STRING", {"multiline": True, "default": "2,2.5,7,1.1", "tooltip": "Comma separated.\nIf used, must contain at least as many entries as merged models.\nThe resulting proportions will be determined by the sum, so you can use any kind of value.\nLine returns will be ignored."}),
                }
            }

function_properties = {
    "function": megaAverage,
    "widgets": widgets,
    'is_low_memory': True,
    }
