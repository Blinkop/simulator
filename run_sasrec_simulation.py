import json
import argparse
from argparse import Namespace
from pathlib import Path

from typing import List

from tqdm import tqdm

import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from simulator import SasrecSimulator


def moving_average(x, n: int = 25):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-nu",
        "--num_users",
        help="number of synthetic users",
        type=int
    )
    parser.add_argument(
        "-upr",
        "--users_per_round",
        help="number of synthetic users to sample per simulation round",
        type=int
    )
    parser.add_argument(
        "-ihl",
        "--init_history_length",
        help="minimum number of items at initial state",
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
        rewards_gamma=args.gamma,
        seed=args.random_seed
    )

    sim.initialize(
        num_users=args.num_users,
        init_history_len=args.init_history_length
    )

    sasrec = torch.load(args.policy_file, weights_only=False).to(args.device)
    sasrec.eval()

    rewards = []

    for _ in tqdm(
        range(args.simulation_iterations),
        total=args.simulation_iterations
    ):
        batch_sequences = sim.get_next_batch(batch_size=args.users_per_round)

        actions = {}

        for u, seq in batch_sequences.items():
            actions[u] = sasrec_prediction(seq=seq, model=sasrec, args=args)

        step_rewards = sim.step(actions=actions)
        rewards.append(step_rewards)

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
    plt.title(f"simulator rewards (avg: {rewards.mean()})")
    plt.xlabel("simulator iteration")
    plt.ylabel("discounted reward")
    plt.plot(rewards.mean(axis=1))
    plt.savefig(exp_folder / "rewards.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
