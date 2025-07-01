import torch
import folder_paths
import safetensors.torch
from safetensors import safe_open
import comfy.model_management as model_management
import importlib.util
from pathlib import Path
import os
from .coordination import merge_models
from comfy.cli_args import args
from .setup_node import DefaultSettingsNode, general_settings, get_settings
import json

ensunique = lambda x: [str(k) for k in {m: 0 for m in x}]
current_dir   = os.path.dirname(os.path.realpath(__file__))
functions_dir = os.path.join(current_dir,"merging_functions")
old_functions_dir = os.path.join(current_dir,"merging_functions","old")

MENU_CATEGORY = "Experimental Model Merging"
DEPRECATED_MENU_CATEGORY = f"{MENU_CATEGORY}/Deprecated"
def get_modules(functions_dir):
    modules = {}
    for fname in os.listdir(functions_dir):
        if fname.endswith('.py'):
            try:
                nom_module = os.path.splitext(fname)[0]
                chemin = os.path.join(functions_dir, fname)
                spec = importlib.util.spec_from_file_location(nom_module, chemin)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "function_properties"):
                    m = getattr(module, "function_properties")
                    if isinstance(m, list):
                        for i in range(len(m)):
                            nom_module = m[i]['name']
                            modules[nom_module] = m[i]
                    else:
                        nom_module = m.get('name', nom_module)
                        modules[nom_module] = m
            except Exception as e:
                print(e)
    return modules

modules = get_modules(functions_dir)
deprecated_modules = get_modules(old_functions_dir)

def insert_after(d, k, nk, nv):
    return {**dict(list(d.items())[:i+1]), nk: nv, **dict(list(d.items())[i+1:])} if (i := list(d).index(k)) >= 0 else d

def get_model(m, print_missing=False):
    name = m.split("\\")[-1].split("/")[-1].replace(".safetensors","")
    if Path(m).is_file():
        return [name, m]
    elif Path(os.path.join(general_settings["input_folder"], m)).is_file():
        return [name, os.path.join(general_settings["input_folder"], m)]
    else:
        for sub in general_settings["input_subfolders"]:
            file_path = os.path.join(general_settings["input_folder"], sub, m)
            if Path(file_path).is_file():
                return [name, file_path]
    if print_missing:
        print(f" Missing checkpoint: {m}")
    return [name, None]

class ModelSelector:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = [f for f in folder_paths.get_filename_list("checkpoints")]
        checkpoints = sorted(checkpoints, key=str.lower)
        checkpoints.insert(0,"None")
        checkpoints_inputs = {f"checkpoint_{X+1}": (checkpoints, {"default": "None"}) for X in range(s.HOW_MANY)}
        required_inputs = {
                "checkpoint_1": (checkpoints, ),
            }
        checkpoints_inputs.update(required_inputs)
        if s.HOW_MANY > 6:
            checkpoints_inputs.update({"ensure_uniques": ("BOOLEAN", {"default": False})})
        node_inputs = {
            "required": checkpoints_inputs,
            "optional": {
                "sync_str": ("STRING", {"forceInput": True}),
                "add_before": ("MDLIST",),
                "add_after": ("MDLIST",),
            }}
        return node_inputs

    HOW_MANY = 24
    FUNCTION = "exec"
    CATEGORY = f"{MENU_CATEGORY}/Model selection"
    RETURN_TYPES = ("MDLIST",)
    RETURN_NAMES = ("Models_list",)

    def exec(self, add_before=[], add_after=[], ensure_uniques=False, **kwargs):
        models_paths = []
        for key, value in kwargs.items():
            if key.startswith("checkpoint_") and value != "None":
                models_paths.append(value)
        models_paths = add_before + models_paths + add_after
        if ensure_uniques:
            models_paths = ensunique(models_paths)
        return (models_paths,)

class ModelSelectFromInputPath:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        checkpoints = [f for f in folder_paths.get_filename_list("checkpoints")]
        checkpoints = sorted(checkpoints, key=str.lower)
        checkpoints.insert(0,"None")
        checkpoints_inputs = {f"checkpoint_{X+1}": ("STRING", {"forceInput": True}) for X in range(s.HOW_MANY)}
        required_inputs = {
                "checkpoint_1": ("STRING", {"forceInput": True}),
            }
        optional_inputs = {
            "sync_str": ("STRING", {"forceInput": True}),
            "add_before": ("MDLIST",),
            "add_after": ("MDLIST",),
            }
        checkpoints_inputs.update(optional_inputs)
        node_inputs = {"required": required_inputs, "optional": checkpoints_inputs}
        return node_inputs

    HOW_MANY = 12
    FUNCTION = "exec"
    CATEGORY = f"{MENU_CATEGORY}/Model selection"
    RETURN_TYPES = ("MDLIST",)
    RETURN_NAMES = ("Models_list",)

    def exec(self, add_before=[], add_after=[], **kwargs):
        models_paths = []
        for key, value in kwargs.items():
            if key.startswith("checkpoint_") and value != "None" and value != "":
                models_paths.append(value)
        models_paths = add_before + models_paths + add_after
        return (models_paths,)

class ModelSelectFromString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "models_list" : ("STRING", {"default": "","multiline": True}),
                "ensure_uniques": ("BOOLEAN", {"default": True}),
                },
            "optional": {
                "add_before": ("MDLIST",),
                "add_after": ("MDLIST",),
            }
            }

    HOW_MANY = 24
    FUNCTION = "exec"
    CATEGORY = f"{MENU_CATEGORY}/Model selection"
    RETURN_TYPES = ("MDLIST",)
    RETURN_NAMES = ("Models_list",)

    def exec(self, models_list, ensure_uniques, add_before=[], add_after=[], **kwargs):
        models_paths = []
        models_list = models_list.split("\n")
        for m in models_list:
            if m != "":
                models_paths.append(m)
        models_paths = add_before + models_paths + add_after
        if ensure_uniques:
            models_paths = ensunique(models_paths)
        return (models_paths,)

class ModelListJoin:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        optional = {f"model_list_{X+3}": ("MDLIST", {"forceInput": True}) for X in range(4)}
        return {
            "required": {
                "model_list_1": ("MDLIST",),
                "model_list_2": ("MDLIST",),
                "ensure_uniques": ("BOOLEAN", {"default": True}),
                },
            "optional": optional
            }

    FUNCTION = "exec"
    CATEGORY = f"{MENU_CATEGORY}/Model selection"
    RETURN_TYPES = ("MDLIST",)
    RETURN_NAMES = ("Models_list",)

    def exec(self, model_list_1, model_list_2, ensure_uniques, **a):
        model_list = model_list_1 + model_list_2
        for k, v in a.items():
            if "model_list" in k and v:
                model_list += v
        if ensure_uniques:
            model_list = ensunique(model_list)
        return (model_list,)

class ModelListCheckNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_list": ("MDLIST",),
                }
            }

    FUNCTION = "exec"
    OUTPUT_NODE = True
    CATEGORY = f"{MENU_CATEGORY}/Model selection"
    RETURN_TYPES = ()

    def exec(self, model_list):
        modellist = [get_model(m) for m in model_list]
        maxxlen = max([len(m[0]) for m in modellist])
        total = sum([m[1] is not None for m in modellist])
        for m in modellist:
            spa = maxxlen - len(m[0])
            print(f"{m[0]}:{spa * ' '} {'found!' if m[1] is not None else 'missing !!!'}")
        print(f"Models found: {total}")
        return {}

class ExtraFiltersNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "numel_min": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "numel_max": ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "ndim_min":  ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "do_not_save" : ("BOOLEAN", {"default": False}),
                "do_not_add_skipped_keys" : ("BOOLEAN", {"default": False}),
                }
            }

    FUNCTION = "exec"
    CATEGORY = f"{MENU_CATEGORY}"
    RETURN_TYPES = ("MFILT",)
    RETURN_NAMES = ("Extra_filters",)

    def exec(self, numel_min, numel_max, ndim_min, do_not_save, do_not_add_skipped_keys):
        mfilt = {
            "numel_min": numel_min,
            "numel_max": numel_max,
            "ndim_min" : ndim_min,
            "do_not_save": do_not_save,
            "do_not_add_skipped_keys": do_not_add_skipped_keys,
        }
        return (mfilt,)

class MergingFunctionNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return s.WIDGET_INPUTS

    FUNCTION = "exec"
    WIDGET_INPUTS = {}
    MERGING_PROPERTIES = {}
    CATEGORY = f"{MENU_CATEGORY}/Functions"
    RETURN_TYPES = ("MFUNC",)
    RETURN_NAMES = ("Merge function",)

    def exec(self, **kwargs):
        merge_parameters = {
            'function': self.MERGING_PROPERTIES['function'],
            'arguments': kwargs,
            'do_not_save':   self.MERGING_PROPERTIES.get('do_not_save', False),
            'is_low_memory': self.MERGING_PROPERTIES.get('is_low_memory', False),
            'do_not_add_skipped': self.MERGING_PROPERTIES.get('do_not_add_skipped', False),
            'post_merge_function' : self.MERGING_PROPERTIES.get('post_merge_function',  None),
            'preparation_function': self.MERGING_PROPERTIES.get('preparation_function', None),
            'custom_save_function': self.MERGING_PROPERTIES.get('custom_save_function', None),
            'hide_progress_bar': self.MERGING_PROPERTIES.get('hide_progress_bar', False),
            }
        merge_parameters["arguments"]["hidden"] = self.MERGING_PROPERTIES.get('hidden', None)
        return (merge_parameters,)

class MergerNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        general_settings = get_settings()
        devices = ["default_device", "all_devices","cpu"] + [str(i) for i in range(torch.cuda.device_count())]
        widgets = {
            "required": {
                "function": ("MFUNC",),
                "model_list":  ("MDLIST",),
                "output_dir":  ("STRING", {"default": general_settings["output_folder"], "tooltip": "in which folder the file will be saved."}),
                "output_name": ("STRING", {"default": general_settings["default_name"], "tooltip": ".safetensors will be added automatically."}),
                "skip_VAE" :  ("BOOLEAN", {"default": general_settings["skip_VAE"], "tooltip": "Will skip layers containing first_stage_model in their name."}),
                "layer_name_filter": ("STRING", {"default": ""}),
                "append_version_to_name" : ("BOOLEAN", {"default": general_settings["append_version_to_name"]}),
                "version" : ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "version_suffix" : ("BOOLEAN", {"default": general_settings["append_version_suffix"]}),
                "suffix" : ("INT", {"default": 0, "min": 0, "max": 1000000000}),
                "dtype_save": (["float16","same_as_first_model","function_output"], {"default": "float16"}),
                "use_device": (devices,),
                "if_nan_detected": (["Cancel merge", "Use first model"], {"default": general_settings["if_nan_detected"]}),
                },
            "optional": {
                "extra_filters": ("MFILT",),
            },
            "hidden":   {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
            }
        if s.VERSION >= 2:
            widgets["required"] = insert_after(widgets["required"],"layer_name_filter", "layer_name_filter_not", ("STRING", {"default": ""}))
        if s.VERSION >= 3:
            widgets["required"].pop("suffix")
            widgets["required"] = insert_after(widgets["required"],"version_suffix", "suffix_seed", ("INT", {"default": 0, "min": 0, "max": 1e9, "control_after_generate": ["increment"]}))
        return widgets

    FUNCTION = "exec"
    CATEGORY = MENU_CATEGORY
    OUTPUT_NODE = True
    VERSION = 2
    RETURN_TYPES = ("STRING", "MDLIST")
    RETURN_NAMES = ("Result path", "Result as model list",)

    def exec(self, model_list, function, output_dir, output_name, skip_VAE, layer_name_filter, append_version_to_name, version, version_suffix, dtype_save, use_device, if_nan_detected, suffix=None, suffix_seed=None, layer_name_filter_not="", extra_filters=None, prompt=None, extra_pnginfo=None,**kwargs):
        suffix = suffix_seed if suffix_seed is not None else suffix
        model_list = [get_model(m) for m in model_list]
        maxxlen = max([len(m[0]) for m in model_list])
        for m in model_list:
            spa = maxxlen - len(m[0])
            print(f"{m[0]}:{spa * ' '} {'found!' if m[1] is not None else 'missing !!!'}")
        mnum = len([m for m in model_list if m[1] is not None])
        print(f"Models to merge: {mnum}")
        metadata = None
        if prompt is not None and not args.disable_metadata:
            metadata = {}
            prompt_info = json.dumps(prompt)
            metadata["prompt"] = prompt_info
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata[x] = json.dumps(extra_pnginfo[x])

        result_path = merge_models(model_list, function, output_dir, output_name, skip_VAE, layer_name_filter, append_version_to_name, version, version_suffix, suffix, dtype_save, use_device, if_nan_detected, layer_name_filter_not, extra_filters, metadata=metadata)
        as_list = [result_path]
        return (result_path, as_list,)

# Nodes creation:
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS["Merger"]    = type(f"Merger",    (MergerNode,), {"CATEGORY": f"{MENU_CATEGORY}/Deprecated", "VERSION": 1})
NODE_CLASS_MAPPINGS["Merger V2"] = type(f"Merger V2", (MergerNode,), {"CATEGORY": f"{MENU_CATEGORY}/Deprecated", "VERSION": 2})
NODE_CLASS_MAPPINGS["Merger V3"] = type(f"Merger V3", (MergerNode,), {"VERSION": 3})
NODE_CLASS_MAPPINGS["Model select from menu"] = ModelSelector
mny = [1, 4]
for hmny in mny:
    NODE_CLASS_MAPPINGS[f"Model select from menu ({hmny})"] = type(f"Model_Selector_{hmny}", (ModelSelector,), {"HOW_MANY": hmny})
NODE_CLASS_MAPPINGS["Model select from input paths"] = ModelSelectFromInputPath
NODE_CLASS_MAPPINGS["Model select from list string"] = ModelSelectFromString
NODE_CLASS_MAPPINGS["Model list join"] = ModelListJoin
NODE_CLASS_MAPPINGS["Check if models exist"] = ModelListCheckNode
NODE_CLASS_MAPPINGS["Extra Filters"] = ExtraFiltersNode
NODE_CLASS_MAPPINGS["Default Settings"] = DefaultSettingsNode
for m in modules:
    NODE_CLASS_MAPPINGS[m] = type(m, (MergingFunctionNode,), {"WIDGET_INPUTS": modules[m]["widgets"],
                                                              "MERGING_PROPERTIES": modules[m],
                                                              "CATEGORY": f"{MENU_CATEGORY}/Functions{modules[m].get('SUBMENU', '')}",
                                                              })
for m in deprecated_modules:
    NODE_CLASS_MAPPINGS[m] = type(m, (MergingFunctionNode,), {"WIDGET_INPUTS": deprecated_modules[m]["widgets"],
                                                              "MERGING_PROPERTIES": deprecated_modules[m],
                                                              "CATEGORY": f"{DEPRECATED_MENU_CATEGORY}/Functions{deprecated_modules[m].get('SUBMENU', '')}",
                                                              })

NODE_CLASS_MAPPINGS = dict(sorted(NODE_CLASS_MAPPINGS.items(), key=lambda x: x[0]))