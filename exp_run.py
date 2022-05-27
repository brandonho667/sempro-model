import json
import sys
import experiment
import os

ks = [3, 5, 10]
sigma = [2, 3, 5]
bin_factors = [3, 5, 10]

exp_mse = dict()
exp_mae = dict()

exp_num = 0

for ld_ks in ks:
    for ld_sigma in sigma:
        for bf in bin_factors:
            exp_name = "ks{ks}sigma{sigma}bf{bf}_lr1e-5".format(
                ks=ld_ks, sigma=ld_sigma, bf=bf)
            with open("experiments/default.json") as json_file:
                template = json.load(json_file)
            template["experiment_name"] = exp_name
            template["hparams"]["lds_ks"] = ld_ks
            template["hparams"]["lds_sigma"] = ld_sigma
            template["hparams"]["bf"] = bf

            with open("experiments/{name}.json".format(
                    name=exp_name), 'w') as outfile:
                json.dump(template, outfile)
            exp = experiment.Experiment(
                name=exp_name, root_dir="experiments", stats_dir="stats")
            exp.train()
            exp.analyze_training()
            exp.analyze_error_dist()
            mae, mse = exp.test()
            exp_mse[exp_name] = mse
            exp_mae[exp_name] = mae
            exp_num += 1
            exp.clear_cache()
            del exp

with open("experiments/results.json", 'w') as outfile:
    json.dump({"mse": exp_mse, "mae": exp_mae}, outfile)

print("min mae: ", min(exp_mae, key=exp_mae.get))
print("min mse: ", min(exp_mse, key=exp_mse.get))
