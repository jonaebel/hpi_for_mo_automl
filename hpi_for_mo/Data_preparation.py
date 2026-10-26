import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write.json import read as cs_read_json

Objective = namedtuple('Objective', ['name'])


@dataclass
class RunGroup:
    """Lightweight replacement for DeepCAVE's Group, backed by raw SMAC JSON files."""
    configspace: ConfigurationSpace
    configs: Dict[str, dict]       # config_id_str -> {hp: value, ...}
    data: List[dict]               # runhistory entries
    objective_names: List[str]

    def get_config(self, config_id):
        """Return HP values dict for a config ID."""
        return self.configs[str(int(config_id))]

    def encode_config(self, config, specific=True):
        """Encode a config (dict or Configuration) to a numeric vector.

        Categorical HPs -> index in hp.choices, numerical HPs -> float value.
        """
        encoded = []
        for hp in self.configspace.get_hyperparameters():
            name = hp.name
            try:
                val = config[name]
            except (KeyError, TypeError):
                encoded.append(np.nan)
                continue
            if val is None:
                encoded.append(np.nan)
            elif hasattr(hp, 'choices'):
                choices = list(hp.choices)
                encoded.append(float(choices.index(val)) if val in choices else np.nan)
            else:
                encoded.append(float(val))
        return encoded

    def get_objectives(self):
        """Return Objective objects with a .name attribute."""
        return [Objective(name=n) for n in self.objective_names]


def _is_smac_run(path):
    """Check if a directory looks like a SMAC run."""
    return (os.path.isdir(path)
            and os.path.exists(os.path.join(path, 'runhistory.json'))
            and os.path.exists(os.path.join(path, 'configspace.json')))


def _read_single_run(run_path):
    """Read a single SMAC run directory and return (cs, configs, data, objective_names)."""
    with open(os.path.join(run_path, 'configspace.json'), 'r') as f:
        cs_raw = json.load(f)

    # SMAC 2.4 writes 'default', ConfigSpace 0.6.1 expects 'default_value'
    for hp in cs_raw.get("hyperparameters", []):
        if "default" in hp and "default_value" not in hp:
            hp["default_value"] = hp.pop("default")

    cs = cs_read_json(json.dumps(cs_raw))

    with open(os.path.join(run_path, 'runhistory.json'), 'r') as f:
        rh = json.load(f)

    configs = rh.get('configs', {})
    raw_data = rh.get('data', [])

    # Parse data entries — handle both flat-dict and [key, value]-pair formats
    data = []
    for entry in raw_data:
        if isinstance(entry, list) and len(entry) == 2:
            key, value = entry
            item = {**key, **value}
        elif isinstance(entry, dict):
            item = entry
        else:
            continue

        status = item.get('status', None)
        if isinstance(status, dict):
            status = 1 if 'SUCCESS' in status.get('__enum__', '') else 0
        elif isinstance(status, str):
            status = 1 if 'SUCCESS' in status else 0
        item['status'] = status
        data.append(item)

    # Read objective names from scenario.json
    objective_names = []
    scenario_path = os.path.join(run_path, 'scenario.json')
    if os.path.exists(scenario_path):
        with open(scenario_path, 'r') as f:
            scenario = json.load(f)
        objective_names = scenario.get('objectives', [])
        if not objective_names:
            obj_field = scenario.get('objective', None)
            if isinstance(obj_field, str):
                objective_names = [obj_field]
            elif isinstance(obj_field, list):
                objective_names = obj_field

    return cs, configs, data, objective_names


def read_runs(path):
    """
    Read SMAC runs from a given path and return them as a RunGroup.
    :param path: path to the smac runs
    :return: the RunGroup of smac runs
    """
    all_configs: Dict[str, dict] = {}
    all_data: List[dict] = []
    configspace = None
    objective_names: List[str] = []
    config_id_offset = 0
    runs_found = 0

    # Collect run directories (same two-level scan as the original)
    run_dirs: List[str] = []
    if _is_smac_run(path):
        run_dirs.append(path)
    else:
        for folder in sorted(os.listdir(path)):
            subpath = os.path.join(path, folder)
            if _is_smac_run(subpath):
                run_dirs.append(subpath)
            elif os.path.isdir(subpath):
                for subfolder in sorted(os.listdir(subpath)):
                    sub_subpath = os.path.join(subpath, subfolder)
                    if _is_smac_run(sub_subpath):
                        run_dirs.append(sub_subpath)

    for run_dir in run_dirs:
        try:
            print('Reading', os.path.basename(run_dir))
            cs, configs, data, obj_names = _read_single_run(run_dir)

            if configspace is None:
                configspace = cs
            if not objective_names and obj_names:
                objective_names = obj_names

            max_id = max((int(k) for k in configs), default=0)
            for cid_str, cval in configs.items():
                all_configs[str(int(cid_str) + config_id_offset)] = cval
            for entry in data:
                entry['config_id'] = int(entry['config_id']) + config_id_offset
                all_data.append(entry)

            config_id_offset += max_id
            runs_found += 1
        except Exception as e:
            print(os.path.basename(run_dir), 'could not be read.')
            print(e)
            continue

    if runs_found == 0:
        print('Data could not be read.')
        exit()

    return RunGroup(
        configspace=configspace,
        configs=all_configs,
        data=all_data,
        objective_names=objective_names,
    )


norm_dict: dict = {}


def normalize(values, obj):
    """
    Normalize the input values with min-max normalization.
    :param values: the values to be normalized
    :param obj: the corresponding objective id
    :return: the normalized values
    """
    global norm_dict
    vmin, vmax = values.min(), values.max()
    norm_dict[obj] = (vmin, vmax)
    if vmax - vmin < 1e-12:
        return values * 0.0
    return (values - vmin) / (vmax - vmin)


def normalize_ablation(values, obj):
    """
    Normalize the input values by the same normalization of the input objective id.
    :param values: the values to be normalized
    :param obj: the corresponding objective id
    :return: the normalized values
    """
    global norm_dict
    vmin, vmax = norm_dict[obj]
    if vmax - vmin < 1e-12:
        normed = values * 0.0
    else:
        normed = (values - vmin) / (vmax - vmin)
    normed[normed < 0] = 0
    return normed


def encode_and_normalize(group, objectives):
    """
    Encodes the configurations and normalizes the objectives.
    :param group: the RunGroup
    :param objectives: the objectives to weight as list of Objective namedtuples
    :return: the dataframe containing the encoded data
    """
    hp_names = group.configspace.get_hyperparameter_names()
    obj_names = [o.name for o in objectives]

    # Map selected objective names to their index in the full cost vector
    all_obj_names = group.objective_names
    obj_indices = []
    for name in obj_names:
        if name in all_obj_names:
            obj_indices.append(all_obj_names.index(name))
        else:
            obj_indices.append(None)

    rows = []
    for entry in group.data:
        if entry.get('status') != 1:
            continue
        config_id = entry['config_id']
        config = group.get_config(config_id)
        encoded = group.encode_config(config, specific=True)

        cost = entry.get('cost', [])
        if not isinstance(cost, list):
            cost = [cost]

        row = dict(zip(hp_names, encoded))
        for obj_name, idx in zip(obj_names, obj_indices):
            if idx is not None and idx < len(cost):
                row[obj_name] = cost[idx]
            else:
                row[obj_name] = np.nan
        row['config_id'] = config_id
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=obj_names)

    global norm_dict
    norm_dict = dict()
    for i, o in enumerate(objectives):
        df[o.name + '_normed'] = normalize(df[o.name], i)

    return df
