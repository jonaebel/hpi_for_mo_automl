import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def compute_weighted_fanova(X, y_objectives, weighting, hp_names, n_trees=100):
    """Compute fANOVA importances on weighted scalarized objective.

    Returns {hp_name: (importance, variance)}.
    """
    y = sum(w * obj for w, obj in zip(weighting, y_objectives))
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=0)
    rf.fit(X, y)

    # Partial-dependence variance decomposition
    var_total = np.var(rf.predict(X))
    importances = {}
    for i, hp in enumerate(hp_names):
        pd_vals = []
        for val in np.unique(X[:, i]):
            Xc = X.copy()
            Xc[:, i] = val
            pd_vals.append(np.mean(rf.predict(Xc)))
        imp = np.var(pd_vals) / var_total if var_total > 1e-12 else 0.0
        # Bootstrap variance estimate from tree disagreement
        tree_imps = []
        for tree in rf.estimators_:
            preds_full = tree.predict(X)
            vt = np.var(preds_full)
            if vt < 1e-12:
                tree_imps.append(0.0)
                continue
            pd_tree = []
            for val in np.unique(X[:, i]):
                Xc = X.copy()
                Xc[:, i] = val
                pd_tree.append(np.mean(tree.predict(Xc)))
            tree_imps.append(np.var(pd_tree) / vt)
        importances[hp] = (imp, np.var(tree_imps))
    return importances


def do_weighted_fanova_mo(group, df, objectives_normed, weightings):
    """
    Calculates weighted fAnova for multiple objectives.
    :param group: the RunGroup
    :param df: dataframe containing the encoded data
    :param objectives_normed: the normalized objective names as a list of strings
    :param weightings: the weightings as list of lists
    :return: the dataframe containing the importances for each hyperparameter per weighting
    """
    hp_names = group.configspace.get_hyperparameter_names()
    X = df[hp_names].to_numpy()
    y_objectives = [df[obj].to_numpy() for obj in objectives_normed]

    df_all = pd.DataFrame([])
    for w in weightings:
        importances = compute_weighted_fanova(X, y_objectives, w, hp_names)
        rows = [{'hp_name': hp, 'fanova': imp, 'variance': var}
                for hp, (imp, var) in importances.items()]
        df_res = pd.DataFrame(rows)
        df_res['weight_for_' + objectives_normed[0]] = w[0]
        df_all = pd.concat([df_all, df_res])
    return df_all
