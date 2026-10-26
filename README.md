# Hyperparameter Importance Analysis for Multi-Objective AutoML

> **Fork note:** This fork removes the runtime dependencies on DeepCAVE and SMAC.
> The library now reads SMAC output files (JSON) directly and uses scikit-learn's
> `RandomForestRegressor` for fANOVA instead of DeepCAVE's `FanovaForest`.
> This makes it compatible with any SMAC 2.x version (including >=2.4) and
> installable without pulling in heavy transitive dependencies (torch, lightning, etc.).
>
> **Install this fork:** `pip install git+https://github.com/YOUR_USERNAME/hpi_for_mo_automl.git@feat/updating-for-SMAC`
>
> **Quick usage:**
> ```python
> from hpi_for_mo.Data_preparation import read_runs, encode_and_normalize
> from hpi_for_mo.weighting import get_weightings
> from hpi_for_mo.fanova import do_weighted_fanova_mo
> from hpi_for_mo.ablation import do_weighted_ablation_mo
>
> group = read_runs("path/to/smac_output")
> objectives = group.get_objectives()
> df = encode_and_normalize(group, objectives)
> objectives_normed = [o.name + "_normed" for o in objectives]
> weightings = get_weightings(objectives_normed, df)
> df_fanova = do_weighted_fanova_mo(group, df, objectives_normed, weightings)
> df_ablation, default = do_weighted_ablation_mo(group, df, objectives_normed, weightings)
> ```

This is the implementation of the paper ["Hyperparameter Importance Analysis for Multi-Objective AutoML"](https://arxiv.org/abs/2405.07640), published at ECAI 2024.

## Abstract
Hyperparameter optimization plays a pivotal role in enhancing the predictive performance and generalization capabilities of ML models. However, in many applications, we do not only care about predictive performance but also about additional objectives such as inference time, memory, or energy consumption. In such multi-objective scenarios, determining the importance of hyperparameters poses a significant challenge due to the complex interplay between the conflicting objectives. In this paper, we propose the first method for assessing the importance of hyperparameters in multi-objective hyperparameter optimization. Our approach leverages surrogate-based hyperparameter importance measures, i.e., fANOVA and ablation paths, to provide insights into the impact of hyperparameters on the optimization objectives. Specifically, we compute the a-priori scalarization of the objectives and determine the importance of the hyperparameters for different objective trade-offs. Through extensive empirical evaluations on diverse benchmark datasets with three different objective pairs, each combined with accuracy, namely time, demographic parity loss, and energy consumption, we demonstrate the effectiveness and robustness of our proposed method. Our findings not only offer valuable guidance for hyperparameter tuning in multi-objective optimization tasks but also contribute to advancing the understanding of hyperparameter importance in complex optimization scenarios. 

## Installation
```
git clone https://github.com/automl/HPI_for_MO_AutoML.git
cd HPI_for_MO_AutoML
conda create -n hpi_for_mo python=3.10
conda activate hpi_for_mo

pip install -e .
```

## Run the experiments:
The experiment scripts require additional dependencies (smac, torch, etc.) that are **not** included in the library's requirements. Install them separately before running experiments.
```
python experiments/run_experiments.py --experiment_name time|fair_loss|energy
```
## Run the analysis and generate the plots:
To calculate MO-HPI on the results of the experiments, run the following command. The ```in_path``` should lead to the folder containing a multi-objective SMAC run, e.g., from the previous step. The ```out_path``` can be any path to where the plots will be saved. The ```objectives``` should be 1-accuracy and one of: time, fair_loss or energy. 

```
python -m hpi_for_mo.main --in_path path_to_run --out_path path_to_save_plots --objectives 1-accuracy time|fair_loss|energy
```

## Funding
The DFKI Niedersachsen (DFKI NI) is funded in the "zukunft.niedersachsen" by the Lower Saxony Ministry of Science and Culture and the Volkswagen Foundation (funding no. ZN3480). Marius Lindauer was supported by the German Federal Ministry of the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (GreenAutoML4FAS project no. 67KI32007A) and was funded by the European Union (ERC, “ixAutoML”, grant no.101041029).
