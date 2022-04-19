import json
import sys
import experiment
import os

weight_decay = [0.01, 0.02, 0.03]
steps_per_decay = [18, 20, 22]
LR_gamma = [0.4, 0.5, 0.6]

exp_mse = [sys.maxsize]*len(weight_decay)*len(steps_per_decay)*len(LR_gamma)
exp_mae = [sys.maxsize]*len(weight_decay)*len(steps_per_decay)*len(LR_gamma)

exp_num = 0

for wd in weight_decay:
    for spd in steps_per_decay:
        for g in LR_gamma:
            with open("default.json") as json_file:
                template = json.load(json_file)
            template["experiment_name"] = "experiment_{num}".format(
                num=exp_num)
            template["hparams"]["weight_decay"] = wd
            template["hparams"]["steps_per_decay"] = spd
            template["hparams"]["LR_gamma"] = g
            with open('experiments/experiment_{num}.json'.format(num=exp_num), 'w') as outfile:
                json.dump(template, outfile)
            exp = experiment.Experiment(
                name="experiment_{num}".format(num=exp_num), root_dir="experiments", stats_dir="stats")
            exp.train()
            exp.analyze_training()
            mae, mse = exp.test()
            exp_mse[exp_num] = mse
            exp_mae[exp_num] = mae
            exp_num += 1
            exp.clear_cache()
            del exp

with open("experiments/results.json", 'w') as outfile:
    json.dump({"mse": mse, "mae": mae}, outfile)

print("min mae: {i}, {mae}".format(i=mae.index(min(mae)), mae=min(mae)))
print("min mse: {i}, {mse}".format(i=mse.index(min(mse)), mse=min(mse)))
