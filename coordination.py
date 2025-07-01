import torch
import safetensors.torch
from safetensors import safe_open
from copy import deepcopy as dc
import folder_paths
import comfy.model_management as model_management
from .setup_node import get_settings
import time
import threading
from queue import Queue
from tqdm import tqdm
from math import prod
import ast
import os
import gc

default_device = model_management.get_torch_device()

CROP_NUMEL_TRIGGER = 120000000 * get_settings()["OOM_trigger_multiplier"]

def get_lazy_tensors(path, key, device):
    try:
        with safe_open(path, framework="pt", device=device) as f:
            if key in f.keys():
                return f.get_tensor(key)
    except Exception as e:
        print(f"\n{e}\nPath: {path}")
    return None

def low_mem_get_layer_wrap(models_list, key, device):
    def low_mem_get_layer(model_id, override_key=None, override_device=None):
        if models_list[model_id][1] is None:
            return None
        return get_lazy_tensors(
            path   = models_list[model_id][1],
            key    = key if override_key is None else override_key,
            device = device if override_device is None else override_device
        )
    return low_mem_get_layer

def check_filter_ok(filters, path, key, device):
    if all([filters['numel_min']  == 0, filters['numel_max']  == 0, filters['ndim_min']  == 0]):
        return True
    try:
        with safe_open(path, framework="pt", device=device) as f:
            layer = f.get_tensor(key)
        lndim  = layer.ndim
        lnumel = layer.numel()
        return all([
            (filters['numel_min'] == 0 or lnumel >= filters['numel_min']),
            (filters['numel_max'] == 0 or lnumel <= filters['numel_max']),
            (filters['ndim_min']  == 0 or lndim  >= filters['ndim_min'] ),
                    ])
    except Exception as e:
        print(f" Error when getting layer at\n {path} \nFor key {key}\n")
        print(e)
    return None

def name_filter_check(j, k):
    if isinstance(j, list):
        return all([n in k for n in j])
    return j in k

def split_tensors_in_many(batch, num_splits):
    shape = list(batch[0].shape)
    with torch.no_grad():
        batch = [tensor.view(tensor.size(0), -1) for tensor in batch]
        split_tensors = []
        for tensor in batch:
            total_size = tensor.size(1)
            base_size = total_size // num_splits
            remainder = total_size % num_splits
            split_sizes = [base_size + (1 if i < remainder else 0) for i in range(num_splits)]
            split_tensors.append(torch.split(tensor, split_sizes, dim=1))
        split_tensors = [torch.stack(parts) for parts in list(zip(*split_tensors))]
    return split_tensors, shape

def hell(split_tensors, original_shape):
    dimz = len(split_tensors[0])
    rez = [[] for _ in range(dimz)]
    batch = len(split_tensors)
    dimrez = []
    for o in range(dimz):
        rez[o] = []
        for b in range(batch):
            rez[o].append([split_tensors[b][o]])
        dimrez.append(combine_many_tensors(rez[o], original_shape)[0])
    dimrez = torch.stack(dimrez)
    return [dimrez]

def combine_many_tensors(split_tensors, original_shape):
    if len(split_tensors[0]) > 1:
        return hell(split_tensors, original_shape)
    split_tensors = [torch.cat(parts, dim=1) for parts in zip(*split_tensors)]
    split_tensors = [tensor.view(original_shape) for tensor in split_tensors]
    return split_tensors

def load_layers(models_list, key, load_device, device_memory):
    loaded_layers = None
    load_report = []
    to_device = load_device
    for m in models_list:
        if m[1] is not None:
            layer = get_lazy_tensors(m[1], key=key, device=load_device)
            if layer is not None:
                load_report.append([m[0], True])
                if loaded_layers is None:
                    split_parts = max(1, (layer.numel() * len(models_list))  // int(CROP_NUMEL_TRIGGER * device_memory / 12))
                    loaded_layers = layer.unsqueeze(0).to(device=to_device)
                    if split_parts > 1:
                        to_device = "cpu"
                        loaded_layers = loaded_layers.to(device="cpu")
                else:
                    loaded_layers = torch.cat([loaded_layers, layer.unsqueeze(0).to(device=to_device)])
            else:
                load_report.append([m[0], False])
    return loaded_layers, load_report

def dtype_before_save(processed, dtp, dtype_save):
    match dtype_save:
        case "same_as_first_model":
            processed = processed.to(dtype=dtp)
        case "float16":
            processed = processed.to(dtype=torch.float16)
        case "function_output":
            pass
    return processed

def crop_split_merge(tensors, func, func_args, device, device_memory, layer_key, loaded_list):
    split_parts = max(1, tensors.numel() // int(CROP_NUMEL_TRIGGER * device_memory / 12))
    if split_parts > 1:
        extras_infos = []
        tensors, original_shape = split_tensors_in_many(tensors, split_parts)
        for y in range(len(tensors)):
            result, extra_infos = func(tensors[y].to(device=device), layer_key=layer_key, is_cropped=True, original_shape=original_shape, loaded_list=loaded_list, **func_args)
            if isinstance(result, list):
                tensors[y] = [r.cpu() for r in result]
            else:
                tensors[y] = [result.cpu()]
            if extra_infos is not None:
                extras_infos.append(extra_infos)
        if len(extras_infos) == 0:
            extras_infos = None
        tensors = combine_many_tensors(tensors, original_shape)
        result = tensors[-1]
        return result, extras_infos
    else:
        result, extra_infos = func(tensors, layer_key=layer_key, is_cropped=False, original_shape=None, loaded_list=loaded_list, **func_args)
    return result, extra_infos

# MAY HAPPEN TO BE REMOVED
def multi_gpu_merge(model_list, keys_to_do, merging_function, dtype_save, if_nan_detected):
    num_gpus = torch.cuda.device_count()
    max_memory = []
    proc_count = []
    torch.cuda.empty_cache()
    gc.collect()
    for i in range(num_gpus):
        free, total = torch.cuda.mem_get_info(device=i)
        mpc = torch.cuda.get_device_properties(i).multi_processor_count
        max_memory.append(round(free / 1024**3, 2))
        proc_count.append(mpc)
    dispatch = []
    for i in range(num_gpus):
        dispatch += [i] * proc_count[i]

    result = [{} for _ in range(num_gpus)]
    result_lock = threading.Lock()

    queues = [Queue() for _ in range(num_gpus)]
    extras_infos = []
    crashed = {}
    total_models = len(model_list)

    def gpu_worker(queue, gpu_id):
        while True:
            if not queue.empty():
                try:
                    key = queue.get()
                    if key is None or any(crashed) or model_management.processing_interrupted():
                        break

                    if merging_function['is_low_memory']:
                        first_tensor = get_lazy_tensors(model_list[0][1], key, gpu_id)
                        dtp = first_tensor.dtype
                        backup_layer = None if if_nan_detected == "Cancel merge" else first_tensor.clone().to(device="cpu")
                        processed, extra_infos = merging_function['function'](first_layer=first_tensor, get_layer=low_mem_get_layer_wrap(model_list, key, gpu_id), total_models=total_models, layer_key=key, **merging_function['arguments'])
                    else:
                        tensors, loaded_list = load_layers(model_list, key, gpu_id, max_memory[gpu_id])
                        dtp = tensors[0].dtype
                        backup_layer = None if if_nan_detected == "Cancel merge" else tensors[0].clone().to(device="cpu")
                        processed, extra_infos = crop_split_merge(tensors, func=merging_function['function'], func_args=merging_function['arguments'], device=gpu_id, device_memory=max_memory[gpu_id], layer_key=key, loaded_list=loaded_list)

                    if processed.isnan().any() or processed.isinf().any():
                        msg = f"\nNaN values detected for result with key:\n{key}\n"
                        if if_nan_detected == "Cancel merge":
                            raise Exception(msg)
                        else:
                            print(msg)
                            result[gpu_id][key] = backup_layer
                    else:
                        processed = dtype_before_save(processed, dtp, dtype_save)
                        result[gpu_id][key] = processed.clone().to(device="cpu")
                    if extra_infos is not None:
                        with result_lock:
                            extras_infos.append(extra_infos)

                    progress_bars[gpu_id].update(1)
                    queue.task_done()
                except Exception as e:
                    crashed[key] = [gpu_id, e]
                    import traceback
                    traceback.print_exc()
                    break
            else:
                time.sleep(1)
                if queue.empty():
                    break

    threads = [threading.Thread(target=gpu_worker, args=(q, i)) for i, q in enumerate(queues)]

    for t in threads:
        t.start()

    queues_length = [0] * num_gpus
    for i, k in enumerate(keys_to_do):
        queues_length[dispatch[i % len(dispatch)]] += 1
        queues[dispatch[i % len(dispatch)]].put(k)

    progress_bars = [tqdm(total=queues_length[i], position=i, desc=f"Device {i}") for i in range(num_gpus)]

    while True:
        if all(q.empty() for q in queues):
            for q in queues:
                q.join()
            break
        elif any(crashed) or model_management.processing_interrupted():
            break
        else:
            time.sleep(0.5)
    for q in queues:
        q.put(None)
    for t in threads:
        if t.is_alive():
            t.join()
    for p in progress_bars:
        p.close()
    if any(crashed):
        del result
        torch.cuda.empty_cache()
        crashed_key = list(crashed)[0]
        print(f"\nCrashed key:\n\
              {crashed_key}\n\
              On device {crashed[crashed_key][0]}\n")
        raise crashed[crashed_key][1]
    if model_management.processing_interrupted():
        return None, None
    for i in range(1, num_gpus):
        result[0].update(result[i])
    result = result[0]
    return result, extras_infos

def single_device_merge(model_list, keys_to_do, merging_function, dtype_save, use_device, if_nan_detected):
    result = {}
    extras_infos = []
    if use_device == "cpu" or (use_device == "default_device" and str(default_device) == "cpu"):
        gpu_id = "cpu"
        max_memory = 1e6
    else:
        if use_device.isdigit():
            gpu_id = int(use_device)
        else:
            gpu_id = int(str(model_management.get_torch_device()).replace("cuda:", ""))
        torch.cuda.empty_cache()
        gc.collect()
        free, total = torch.cuda.mem_get_info(device=gpu_id)
        max_memory = round(free / 1024**3, 2)
    total_models = len(model_list)
    backup_layer = None
    hide_bar = merging_function['arguments'].get('hide_progress_bar', False) or merging_function.get('hide_progress_bar', False)
    progress_bar = tqdm(total=len(keys_to_do), position=0, desc="Merging models...", leave=True, disable=hide_bar)
    for key in keys_to_do:
        try:
            if model_management.processing_interrupted():
                return None, None
            if merging_function['is_low_memory']:
                first_tensor = get_lazy_tensors(model_list[0][1], key, gpu_id)
                dtp = first_tensor.dtype
                if if_nan_detected == "Use first model":
                    backup_layer = first_tensor.clone().to(device="cpu")
                processed, extra_infos = merging_function['function'](first_layer=first_tensor, get_layer=low_mem_get_layer_wrap(model_list, key, gpu_id), total_models=total_models, layer_key=key, **merging_function['arguments'])
            else:
                tensors, loaded_list = load_layers(model_list, key, gpu_id, max_memory)
                dtp = tensors[0].dtype
                if if_nan_detected == "Use first model":
                    backup_layer = tensors[0].clone().to(device="cpu")
                processed, extra_infos = crop_split_merge(tensors, func=merging_function['function'], func_args=merging_function['arguments'], device=gpu_id, device_memory=max_memory, layer_key=key, loaded_list=loaded_list)

            if processed.isnan().any() or processed.isinf().any():
                msg = f"\nNaN values detected for result with key:\n{key}\n"
                if if_nan_detected == "Cancel merge":
                    raise Exception(msg)
                else:
                    print(msg)
                    result[key] = backup_layer
            else:
                processed = dtype_before_save(processed, dtp, dtype_save)
                result[key] = processed.clone().to(device="cpu")
            if not hide_bar:
                progress_bar.update(1)
            if extra_infos is not None:
                extras_infos.append(extra_infos)
        except Exception as e:
            del result
            torch.cuda.empty_cache()
            gc.collect()
            print(f"\nCrashed key:\n\
                    {key}\n")
            import traceback
            traceback.print_exc()
            raise e
    return result, extras_infos

def merge_models(model_list, merging_function, output_dir, output_name, skip_VAE, layer_name_filter,
                 append_version_to_name, version, version_suffix, suffix, dtype_save, use_device, if_nan_detected, layer_name_filter_not, extra_filters=None, metadata=None):
    model_management.cleanup_models()
    do_not_save = merging_function['do_not_save'] or (extra_filters is not None and extra_filters['do_not_save'])
    no_skipped  = merging_function['do_not_add_skipped'] or  (extra_filters is not None and extra_filters['do_not_add_skipped_keys'])
    vs_str = f"_v{version}{f'.{suffix:02.0f}' * version_suffix}" * append_version_to_name
    output_name = f"{output_name}{vs_str}.safetensors"
    save_path = os.path.join(output_dir, output_name)
    if not do_not_save:
        if not os.path.exists(save_path):
            if metadata is not None:
                safetensors.torch.save_file({'placeholder': torch.rand(4, 4)}, save_path, metadata=metadata)
            else:
                open(save_path, 'w').close()
        print(f"save path will be {save_path}")

    # custom preparation
    if merging_function['preparation_function'] is not None:
        merging_function['arguments']['preparation_result'] = merging_function['preparation_function'](models_found=model_list, load_function=get_lazy_tensors, **merging_function['arguments'])

    # get keys, set filters
    with safe_open(model_list[0][1], framework="pt", device="cpu") as f:
        model_keys = f.keys()
    model_keys = sorted(model_keys, key=str.lower)

    layer_filters = []
    layer_filter_not = []
    if "[" in layer_name_filter:
        layer_filters = ast.literal_eval(layer_name_filter)
    elif layer_name_filter != "":
        layer_filters = layer_name_filter.split(",")
    if "[" in layer_name_filter_not:
        layer_filter_not = ast.literal_eval(layer_name_filter_not)
    elif layer_name_filter_not != "":
        layer_filter_not = layer_name_filter_not.split(",")
    keys_to_do   = []
    keys_to_skip = []
    # "model_ema.decay" and "model_ema.num_updates" often throw a NaN from SD1.5 checkpoints so... sneaky skip.
    for k in tqdm(model_keys, desc=f"Checking keys to merge...", disable=extra_filters is None):
        if (len(layer_filters) == 0 or any([name_filter_check(j, k) for j in layer_filters])) \
            and (len(layer_filter_not) == 0 or not any([name_filter_check(j, k) for j in layer_filter_not])) \
                and ((skip_VAE and not any([j in k for j in ["first_stage_model", "model_ema.decay", "model_ema.num_updates"]])) or not skip_VAE)\
                    and (extra_filters is None or check_filter_ok(extra_filters, model_list[0][1], k, device="cpu")):
            keys_to_do.append(k)
        else:
            keys_to_skip.append(k)
    if len(keys_to_skip) > 0 and (extra_filters is not None or len(layer_filters) > 0):
        print(f"Keys to do: {len(keys_to_do)}\nKeys to skip: {len(keys_to_skip)}\n")

    # actually merge
    if use_device != "all_devices" or torch.cuda.device_count() <= 1:
        result, extras_infos = single_device_merge(model_list, keys_to_do, merging_function, dtype_save, use_device, if_nan_detected)
    else:
        result, extras_infos = multi_gpu_merge(model_list, keys_to_do, merging_function, dtype_save, if_nan_detected)
    if result is None:
        return ""

    # custom save function
    custom_save_function = merging_function.get("custom_save_function", None)
    if custom_save_function is not None:
        try:
            result = custom_save_function(extras_infos=extras_infos, result=result, model_list=model_list, recombine_split=combine_many_tensors,
                                          output_name=output_name, function_arguments=merging_function['arguments'],
                                          save_path=save_path, metadata=metadata, save_function=safetensors.torch.save_file)
        except Exception as e:
            print(e)

    # save result
    if not do_not_save and result is not None:
        if len(keys_to_skip) > 0 and not no_skipped:
            for x in tqdm(range(len(keys_to_skip)), desc=f"Adding skipped keys"):
                k = keys_to_skip[x]
                result[k] = get_lazy_tensors(model_list[0][1], k, device=0).to(device="cpu")

        print(f" Done!\nSaving in progress...")
        if metadata is not None:
            safetensors.torch.save_file(result, save_path, metadata=metadata)
        else:
            safetensors.torch.save_file(result, save_path)

    del result
    torch.cuda.empty_cache()
    gc.collect()

    # post merge
    post_merge_function = merging_function.get("post_merge_function", None)
    if post_merge_function is not None:
        try:
            post_merge_function(extras_infos=extras_infos, model_list=model_list, output_name=output_name, function_arguments=merging_function['arguments'])
        except Exception as e:
            print(e)

    if not do_not_save and not model_management.processing_interrupted():
        print(f"Model saved as {output_name}")

    return save_path