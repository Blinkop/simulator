import json
import argparse
from argparse import Namespace
from pathlib import Path

from typing import List

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from simulator import SasrecSimulator
from data_utils import get_dataset


def moving_average(x, n: int = 25):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-upr",
        "--users_per_round",
        help="number of synthetic users to sample per simulation round",
        type=int
    )
    parser.add_argument(
        "-si",
        "--simulation_iterations",
        help="number of simulation iterations",
        type=int
    )
    parser.add_argument(
        "-nm",
        "--n_multinomial",
        help="number of multinomial trials",
        type=int
    )
    parser.add_argument(
        "-g",
        "--gamma",
        help="discount factor",
        type=float
    )
    parser.add_argument(
        "-lf",
        "--log_file",
        help="path to a log_file",
        type=str
    )
    parser.add_argument(
        "-sf",
        "--sasrec_file",
        help="path to a sasrec model",
        type=str
    )
    parser.add_argument(
        "-pf",
        "--policy_file",
        help="path to a policy",
        type=str
    )
    parser.add_argument(
        "-d",
        "--device",
        help="which device to use (cpu/cuda)",
        type=str
    )
    parser.add_argument(
        "-rs",
        "--random_seed",
        help="random seed",
        type=int
    )
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="results folder",
        type=str
    )

    return parser.parse_args()


@torch.no_grad()
def sasrec_prediction(seq: List[int], model: torch.nn.Module, args: Namespace):
    seq_tensor = torch.LongTensor(seq).to(args.device)
    logits = model.score(seq_tensor).flatten().detach().cpu()[:-1]
    logits[seq] = logits.min()

    return logits.argmax().item()


def run_simulation(args: Namespace):
    sim = SasrecSimulator(
        n_multinomial=args.n_multinomial,
        sasrec_path=args.sasrec_file,
        device=args.device,
        seed=args.random_seed
    )

    log_df = pd.read_csv(args.log_file).rename(columns={'movieid' : 'itemid'})
    train_log_df, _, _, _, _, _ = get_dataset(
        path=log_df,
        splitting='temporal_full',
        q=0.8,
        test_size=log_df['userid'].nunique()
    )

    sim.initialize_from_log(train_log_df.rename(columns={'itemid' : 'movieid'}))

    sasrec = torch.load(args.policy_file).to(args.device)
    sasrec.eval()

    rewards = []
    batch_sequences = sim.start(batch_size=args.users_per_round)

    for i in tqdm(
        range(args.simulation_iterations),
        total=args.simulation_iterations
    ):
        actions = {}

        for u, seq in batch_sequences.items():
            actions[u] = sasrec_prediction(seq=seq, model=sasrec, args=args)

        batch_sequences, r = sim.step(actions=actions, gamma=args.gamma, count_initial_length=False)
        rewards.append(r)

    return sim.log_df, np.array(rewards)


def main():
    args = get_args()

    log_df, rewards = run_simulation(args=args)

    exp_folder = Path(f"experiments/{args.experiment_name}")
    exp_folder.mkdir(parents=True, exist_ok=True)

    with open(exp_folder / "arguments.json", 'w') as f:
        json.dump(vars(args), f, indent="\t")

    np.save(exp_folder / "rewards.npy", rewards)
    log_df.to_csv(exp_folder / "log.csv", index=False)

    sns.set_theme()
    plt.title("simulator rewards")
    plt.xlabel("simulator iteration")
    plt.ylabel("discounted reward")
    plt.plot(moving_average(rewards.mean(axis=1), n=25))
    plt.savefig(exp_folder / "rewards.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
