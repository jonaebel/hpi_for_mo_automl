import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import copy

from sklearn.metrics import mean_squared_error

from .Data_preparation import normalize_ablation


def train_model(group, df, objective):
    """
    Train a Random Forest model as surrogate to predict the performance of a configuration. Where the input data are the
    configurations and the objective.
    :param group: the runs as group
    :param df: dataframe containing the encoded configurations
    :param objective: the name of the target column
    :return: the trained model
    """
    X = df[group.configspace.get_hyperparameter_names()].to_numpy()
    Y = df[objective].to_numpy()
    regr = RandomForestRegressor(max_depth=100, random_state=0)
    regr.fit(X, Y)
    print('MSE of the surrogate model:', mean_squared_error(Y, regr.predict(X)))
    return regr


def get_cfg(hp, cfg, hp_value):
    """
    Copies the input cfg and changes the given hyperparameter to the given value.
    :param hp: hyperparameter to change
    :param cfg: configuration
    :param hp_value: hyperparameter value
    :return: the changed cfg
    """
    cfg_copy = copy.copy(cfg)
    cfg_copy[hp] = hp_value
    return cfg_copy


def predict_w_weight(group, cfg, models, weighting):
    """
    Predicts the performance of the input configuration with the input list of models. The model results are normalized,
    weighted by the input wieghtings and summed.
    :param group: the runs as group
    :param cfg: configuration
    :param models: list of models
    :param weighting: tuple of weightings
    :return: the mean and variance of the normalized weighted sum of predictions
    """
    cfg_encoded = [group.encode_config(cfg, specific=True)]
    mean, var = 0, 0
    obj = 0
    for model, w in zip(models, weighting):
        all_predictions = np.stack([tree.predict(cfg_encoded) for tree in model.estimators_])
        mean += w * np.mean(normalize_ablation(all_predictions, obj), axis=0)
        var += w * np.var(normalize_ablation(all_predictions, obj), axis=0)
        obj += 1
    return mean, var


def get_hp_best_performance(hps, config, models, group, incumbent_config, res_previous, weighting):
    """
    Returns the hyperparameter with the minimum performance given a start configuration, a config to compare to, a
    list of hps to consider and a model to predict the performance.
    :param hps: list of hyperparameters to consider
    :param config: the configuration
    :param models: the list of trained surrogate models, one for each objective
    :param group: the runs as group
    :param incumbent_config: the configuration of the incumbent
    :param res_previous: the previous performance
    :param weighting: the weighting of
    :return: whether to continue the ablation, the name of the min hp, the new performance, the variance and the new
    configuration with the changed hyperparameter
    """
    min_hp = ''
    res_min = 1
    for hp in hps:
        if incumbent_config[hp] is not None and hp in config.keys():
            res, _ = predict_w_weight(group, get_cfg(hp, config, incumbent_config[hp]), models, weighting)
            if res < res_min:
                min_hp = hp
                res_min = res
        else:
            continue
    if (min_hp != '') & (res_min < res_previous):
        config[min_hp] = incumbent_config[min_hp]
        min_hp_performance, min_hp_var = predict_w_weight(group, config, models, weighting)
        return True, min_hp, min_hp_performance, min_hp_var, config
    else:
        return False, None, None, None, None


def calculate_ablation_path(group, df, objectives_normed, weighting, models):
    """
    Calculates the ablation path for an input weighting.
    :param group: the runs as group
    :param df: dataframe containing the encoded data
    :param objectives_normed: the normalized objective names as a list of strings
    :param weighting: the weightings as list of lists
    :param models: the list of trained surrogate models, one for each objective
    :return: the dataframe containing the importances for each hyperparameter for the weighting, the default performance
    """
    incumbent_cfg_id = np.argmin(sum(df[obj] * w for obj, w in zip(objectives_normed, weighting)))
    incumbent_cfg = group.get_config(df.iloc[incumbent_cfg_id]['config_id'])
    default_cfg = group.configspace.get_default_configuration()
    res_previous, var = predict_w_weight(group, default_cfg, models, weighting)

    cfg = copy.copy(default_cfg)
    df_abl = pd.DataFrame([])
    df_abl = pd.concat(
        [df_abl,
         pd.DataFrame({'hp_name': 'Default', 'ablation': 0, 'variance': 0, 'new_performance': res_previous})])
    hps = group.configspace.get_hyperparameter_names()
    for i in range(len(hps)):
        continue_ablation, min_hp, min_hp_performance, min_hp_var, cfg = get_hp_best_performance(hps, cfg, models, group,
                                                                                                 incumbent_cfg, res_previous,
                                                                                                 weighting)
        if not continue_ablation:
            break
        diff = res_previous - min_hp_performance
        res_previous = min_hp_performance
        df_abl = pd.concat(
            [df_abl, pd.DataFrame(
                {'hp_name': min_hp, 'ablation': diff, 'variance': min_hp_var, 'new_performance': res_previous,
                 'incumbent_cfg_id': incumbent_cfg_id})])
        hps.remove(min_hp)
    default = [model.predict([group.encode_config(default_cfg, specific=True)]) for model in models]
    return df_abl.reset_index(drop=True).reset_index(), default


def do_weighted_ablation_mo(group, df, objectives_normed, weightings):
    """
    Calculates the ablation path for diferent weightings of multiple objectives.
    :param group: the runs as group
    :param df: dataframe containing the encoded data
    :param objectives_normed: the normalized objective names as a list of strings
    :param weightings: the weightings as list of lists
    :return: the dataframe containing the importances for each hyperparameter per weighting, the default performance
    """
    df_all = pd.DataFrame([])
    models = []
    for obj in objectives_normed:
        models.append(train_model(group, df, obj.split('_normed')[0]))
    for w in weightings:
        df_res, default = calculate_ablation_path(group, df, objectives_normed, w, models)
        df_res = df_res.drop(columns=['index'])
        df_res['weight_for_' + objectives_normed[0]] = w[0]
        df_all = pd.concat([df_all, df_res])
    return df_all, default
