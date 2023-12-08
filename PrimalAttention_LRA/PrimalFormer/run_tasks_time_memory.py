from model_wrapper import ModelForSC, ModelForSCDual
from dataset import LRADataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
import os
import json
import pickle
import numpy as np
import argparse
import math
import itertools
import lra_config

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model", dest = "model", required = True)
parser.add_argument("--task", type = str, help = "task", dest = "task", required = True)
parser.add_argument("--skip_train", type = int, help = "skip_train", dest = "skip_train", default = 0)
# only for the primal attention
parser.add_argument("--low_rank", type = int, help = "low rank dimension", default = 20)
parser.add_argument("--eta", type = float, help = "eta to balance the loss_ksvd and loss_ce", default = 0.1)
parser.add_argument("--rank_multi", type = int, help = "low rank dimension * rank_multi", default = 10)
# whether to use data-independent weights for computing trace on KSVD loss
parser.add_argument("--trace_no_x", action='store_true', help='use data-independent weights for trace computatation in KSVD loss if true') 

args = parser.parse_args()

attn_type = args.model
task = args.task

checkpoint_dir = "./logs/"

print(lra_config.config[task]["extra_attn_config"].keys(), flush = True)

model_config = lra_config.config[task]["model"]
model_config.update(lra_config.config[task]["extra_attn_config"][attn_type])

model_config["mixed_precision"] = True
model_config["attn_type"] = attn_type
model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))

training_config = lra_config.config[task]["training"]
gpu_memory_config = lra_config.config[task]["gpu_memory"]

if model_config["attn_type"].startswith("primal"):
    model_config["low_rank"] = args.low_rank
    model_config["eta"] = args.eta
    model_config["rank_multi"] = args.rank_multi
    model_config["trace_no_x"] = args.trace_no_x

device_ids = list(range(torch.cuda.device_count()))
print(f"GPU list: {device_ids}")

print(json.dumps([model_config, training_config], indent = 4))

if task == "retrieval":
    model = ModelForSCDual(model_config)
else:
    model = ModelForSC(model_config)

print(model)
print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)

model = model.cuda()
model = nn.DataParallel(model, device_ids = device_ids)

ds_iter = {
    "train":enumerate(DataLoader(LRADataset(f"./datasets/{task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "dev":enumerate(DataLoader(LRADataset(f"./datasets/{task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
    "test":enumerate(DataLoader(LRADataset(f"./datasets/{task}.test.pickle", True), batch_size = training_config["batch_size"], drop_last = True)),
}

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = training_config["learning_rate"],
    betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
)

lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer = optimizer,
    max_lr = training_config["learning_rate"],
    pct_start = training_config["warmup"] / training_config["num_train_steps"],
    anneal_strategy = training_config["lr_decay"],
    total_steps = training_config["num_train_steps"]
)

amp_scaler = torch.cuda.amp.GradScaler() if model_config["mixed_precision"] else None

def step(component, step_idx):
    t0 = time.time()

    optimizer.zero_grad()

    _, batch = next(ds_iter[component])
    for key in batch:
        batch[key] = batch[key].cuda()

    if component == "train":
        outputs = {}

        partial_inputs_list = [{} for _ in range(accumu_steps)]
        for key in batch:
            for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                partial_inputs_list[idx][key] = inp

        for partial_inputs in partial_inputs_list:
            partial_outputs = model(**partial_inputs)
            for key in partial_outputs:
                partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                if key not in outputs:
                    outputs[key] = partial_outputs[key]
                else:
                    outputs[key] += partial_outputs[key]
            amp_scaler.scale(partial_outputs["loss"]).backward()

        amp_scaler.step(optimizer)
        amp_scaler.update()
        lr_scheduler.step()
    else:
        with torch.no_grad():
            outputs = {}

            partial_inputs_list = [{} for _ in range(accumu_steps)]
            for key in batch:
                for idx, inp in enumerate(torch.chunk(batch[key], accumu_steps, dim = 0)):
                    partial_inputs_list[idx][key] = inp

            for partial_inputs in partial_inputs_list:
                partial_outputs = model(**partial_inputs)
                for key in partial_outputs:
                    partial_outputs[key] = partial_outputs[key].mean() / accumu_steps
                    if key not in outputs:
                        outputs[key] = partial_outputs[key]
                    else:
                        outputs[key] += partial_outputs[key]

    t1 = time.time()

    batch_size = batch[list(batch.keys())[0]].size(0)
    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = outputs["loss"].data.item()
    accu = outputs["accu"].data.item()
    if "loss_ksvd" in outputs.keys():
        loss_ce = outputs["loss_ce"].data.item()
        loss_ksvd = outputs["loss_ksvd"].data.item()
    time_since_start = time.time() - init_t

    if "loss_ksvd" in outputs.keys():
        print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, loss_ce={loss_ce:.4f}, loss_ksvd={loss_ksvd:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)
    else:
        print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)

    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)
    if "loss_ksvd" in outputs.keys():
        summary[component]["loss_ce"].append(loss_ce)
        summary[component]["loss_ksvd"].append(loss_ksvd)

def print_summary(summary, save_if_improved, train_step_idx):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])
    if "loss_ksvd" in summary.keys():
        summary["loss_ce"] = np.mean(summary["loss_ce"])
        summary["loss_ksvd"] = np.mean(summary["loss_ksvd"])

    print()
    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]
        if save_if_improved:
            best_accu = summary["best_accu"]
            torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", ".model"))
            print(f"best_accu={best_accu}. Saved best model")
    
    # save the latest model
    torch.save({"model_state_dict":model.module.state_dict()}, log_f_path.replace(".log", "_last.model"))

    summary_round = {"train_step_idx":train_step_idx}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)
    log_f.write(json.dumps(summary_round, sort_keys = True) + "\n")
    log_f.flush()

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []
    if "loss_ksvd" in summary.keys():
        summary["loss_ce"] = []
        summary["loss_ksvd"] = []

init_t = time.time()

if model_config["attn_type"].startswith("primal"):
    if model_config["trace_no_x"]:
        log_f_path = os.path.join(checkpoint_dir, "{}_{}_eta{}_rank{}_multi{}_consTrace_output.log".format(task, attn_type, model_config["eta"], model_config["low_rank"], model_config["rank_multi"]))
    else:
        log_f_path = os.path.join(checkpoint_dir, "{}_{}_eta{}_rank{}_multi{}_output.log".format(task, attn_type, model_config["eta"], model_config["low_rank"], model_config["rank_multi"]))
else:
    log_f_path = os.path.join(checkpoint_dir, f"{task}_{attn_type}_output.log")
log_f = open(log_f_path, "a+")

if model_config["attn_type"].startswith("primal"):
    summary = {
        component:{"t":0, "loss":[], "loss_ce":[], "loss_ksvd":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test"]
    }
else:
    summary = {
        component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test"]
    }

accumu_steps = max(training_config["batch_size"] // len(device_ids) // gpu_memory_config[attn_type], 1)
print(f"accumu_steps={accumu_steps}")

if args.skip_train == 0:
    try:
        model.train()

        # record time
        init_total_t = time.time()

        for train_step_idx in range(training_config["num_train_steps"]):

            if train_step_idx >= 1000:
                break

            outputs = step("train", train_step_idx)

            if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
                print_summary(summary["train"], False, train_step_idx)
                model.eval()
                for dev_step_idx in range(training_config["num_eval_steps"]):
                    outputs = step("dev", dev_step_idx)
                print_summary(summary["dev"], True, train_step_idx)
                # also eval on the test set
                if (train_step_idx + 1) % (training_config["eval_frequency"] * 10) == 0:
                    for test_step_idx in range(training_config["num_eval_steps"]):
                        outputs = step("test", test_step_idx)
                    print_summary(summary["test"], False, train_step_idx)

                model.train()
    except KeyboardInterrupt as e:
        print(e)


print("total training time (s) of the first 1K training steps: {}\n".format(time.time()-init_total_t))
print("peak memory usage (MB) of the first 1K training steps: {}\n".format(torch.cuda.max_memory_allocated()))

ds_iter["test"] = enumerate(DataLoader(LRADataset(f"./datasets/{task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True))

checkpoint = torch.load(log_f_path.replace(".log", ".model"), map_location = "cpu")
model.module.load_state_dict(checkpoint["model_state_dict"])
model.eval()
try:
    for test_step_idx in itertools.count():
        outputs = step("test", test_step_idx)
except StopIteration:
    print("Please only check the Acc not best_Acc for the test set")
    print_summary(summary["test"], False, train_step_idx)