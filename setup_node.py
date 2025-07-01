import os
import json
import folder_paths
current_dir = os.path.dirname(os.path.realpath(__file__))
settings_file  = os.path.join(current_dir,"settings.json")
default_folder = folder_paths.get_folder_paths('checkpoints')[0]

def save_dict(d, path):
    with open(path, 'w') as f:
        json.dump(d, f)

def load_dict(path):
    with open(path, 'r') as f:
        return json.load(f)

def generate_new_settings():
    settings = {
        "default_name": "MODEL_NAME",
        "output_folder": default_folder,
        "input_folder":  default_folder,
        "input_subfolders": [],
        "skip_VAE": True,
        "append_version_to_name": True,
        "append_version_suffix": True,
        "if_nan_detected": "Cancel merge",
        "OOM_trigger_multiplier": 1.0
    }
    save_dict(settings, settings_file)

if not os.path.exists(settings_file):
    generate_new_settings()

def get_settings():
    return load_dict(settings_file)

general_settings = load_dict(settings_file)

class DefaultSettingsNode:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "default_name" :  ("STRING", {"default": general_settings["default_name"]}),
                "output_folder": ("STRING", {"default": general_settings["output_folder"]}),
                "input_folder":   ("STRING", {"default": general_settings["input_folder"]}),
                "input_subfolders": ("STRING", {"default": ",".join(general_settings["input_subfolders"])}),
                "skip_VAE" : ("BOOLEAN", {"default": general_settings["skip_VAE"]}),
                "append_version_to_name" : ("BOOLEAN", {"default": general_settings["append_version_to_name"]}),
                "append_version_suffix" : ("BOOLEAN", {"default": general_settings["append_version_suffix"]}),
                "if_nan_detected": (["Cancel merge", "Use first model"], {"default": general_settings["if_nan_detected"]}),
                "OOM_trigger_multiplier": ("FLOAT", {"default": 1, "min": 1/20, "max": 10, "step": 1/20, "round": 1/100, "tooltip": "If you're getting an OOM try to lower this value. The effect will be applied after restart."}),
                }
            }

    FUNCTION = "exec"
    OUTPUT_NODE = True
    CATEGORY = "Experimental Model Merging"
    RETURN_TYPES = ()

    def exec(self, default_name, output_folder, input_folder, input_subfolders, skip_VAE, append_version_to_name, append_version_suffix, if_nan_detected, OOM_trigger_multiplier):
        global general_settings
        settings = {
            "default_name": default_name,
            "output_folder": output_folder,
            "input_folder": input_folder,
            "input_subfolders": [i for i in input_subfolders.split(",") if i != ""],
            "skip_VAE": skip_VAE,
            "append_version_to_name": append_version_to_name,
            "append_version_suffix": append_version_suffix,
            "if_nan_detected": if_nan_detected,
            "OOM_trigger_multiplier": OOM_trigger_multiplier,
        }
        general_settings = settings
        save_dict(settings, settings_file)
        print("OOM TRIGGER WILL ONLY BE APPLIED AFTER RESTART")
        return {}