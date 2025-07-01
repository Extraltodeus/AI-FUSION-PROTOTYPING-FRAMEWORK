import torch

r4 = lambda x: round(x, 4)
infos = {
    'dtype': lambda x: x.dtype,
    'shape': lambda x: x.shape,
    'numel': lambda x: x.numel(),
    'zeros': lambda x: f"{round((100 * (x == 0).sum() / x.numel()).item(), 2):05.2f}%",
    'min': lambda x: x.min().item(),
    'max': lambda x: x.max().item(),
    'std': lambda x: x.std().item(),
    'norm': lambda x: x.norm().item(),
    'mean': lambda x: x.mean().item(),
}
pl = lambda x, y: print(f" {y} / {x.shape} / {x.numel()} / {x.dtype}")
@torch.no_grad()
def check_layer(first_layer, get_layer, layer_key, total_models, **kwargs):
    pl(first_layer, layer_key)
    return first_layer, None

widgets = {
            "required": {
                }
            }

function_properties = {
    "function": check_layer,
    "widgets" : widgets,
    "do_not_save"  : True,
    "is_low_memory": True,
    "hide_progress_bar": True,
    }
