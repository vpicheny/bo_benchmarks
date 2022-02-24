from typing import Dict
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import os
from config import make_all_configs, make_config

configs = make_all_configs()
config = make_config(configs[1])

dir = "results_ll2"

tag="lunar_lander_6_dim_6_q_0.9"

all_subdir = glob(f"{dir}/{tag}/*/")
print(f"Found {len(all_subdir)} algorithms")
# Each subdir corresponds to a particular algorithm instance, so
# we'll group the results by subdir, using a dictionary to allow for varying lengths
all_Xs = dict()

for i, subdir in enumerate(all_subdir):
    all_files = glob(f"{subdir}*X.npy")
    exp_name = os.path.basename(subdir[:-1])
    print(f"    Processed results for {exp_name}")
    print(f"    Found {len(all_files)} files")

    if len(all_files) > 1:
        regret = np.load(file=all_files[0])
        for file in all_files[1:]:
            reg = np.load(file=file)
            regret = np.hstack([regret, reg])

        all_Xs[exp_name] = regret

all_Xs["TS"] = all_Xs.pop("quantile_rule_TS_init_50_budget_250_batch_25")
all_Xs["MES"] = all_Xs.pop("quantile_rule_MES_init_50_budget_250_batch_25")
all_Xs["GIBBON"] = all_Xs.pop("quantile_rule_GIBBON_init_50_budget_250_batch_25")
all_Xs["GPR"] = all_Xs.pop("GPR_rule_TS_init_50_budget_250_batch_25")
all_Xs["hetGP"] = all_Xs.pop("hetgp_rule_TS_init_50_budget_250_batch_25")
all_Xs["Q-GIBBON"] = all_Xs.pop("homquantile_rule_GIBBON_init_50_budget_250_batch_25")

all_Xs.pop("hetGP")
all_Xs.pop("MES")
all_Xs.pop("TS")
all_Xs.pop("GPR")
all_Xs.pop("GIBBON")
all_Qs = dict()

for name, X in all_Xs.items():
    print(name)
    final_Xs = X[1250, :]
    Xs = final_Xs.reshape(-1, 6)
    Qs = np.zeros(Xs.shape[0])
    for j in range(Xs.shape[0]):
        print(j)
        quantile = config.problem.quantile_fun(Xs[j:(j+1), :])
        print(quantile)
        Qs[j] = quantile
    all_Qs[name] = Qs

for name, X in all_Qs.items():
    print(name, np.mean(X) * 300, np.std(X) * 300)
