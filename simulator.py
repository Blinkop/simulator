import abc
from typing import Iterable, Union, Dict

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd


class AbstractSimulator(abc.ABC):

    INITIAL_SOURCE = 'initial'
    POLICY_SOURCE = 'policy'
    OTHER_SOURCE = 'other'

    def __init__(self, rewards_gamma: float = 1.0, seed: int = None):
        super().__init__()

        self._gamma = rewards_gamma
        self._seed = seed
        self._numpy_generator = np.random.default_rng(self._seed)

        self._history = {} # user : (item, source, rating)
        self._num_items = 0

        self._initialized = False

    @property
    def log_df(self):
        if not self._initialized:
            raise RuntimeError(f'Simulator is not initialized. Call initialize() first')

        max_seq_len = max([len(seq) for _, seq in self._history.items()])

        user_dfs = []
        for u, seq in self._history.items():
            user_df = pd.DataFrame(seq, columns=['movieid', 'source', 'rating'])
            user_df['userid'] = u
            user_df['timestamp'] = np.linspace(0, max_seq_len - 1, len(seq)).round().astype(int)

            user_dfs.append(user_df[['userid', 'movieid', 'rating', 'timestamp', 'source']])

        return pd.concat(user_dfs)
    
    @property
    def item_ids(self):
        if not self._initialized:
            raise RuntimeError(f'Simulator is not initialized. Call initialize() first')

        return np.array([*range(self._num_items)])

    def _compute_rewards(
        self,
        response: Dict[int, bool],
        gamma: float = 1.0
    ):
        discounted_rewards = [
            int(r) * (gamma ** len(self.get_policy_sequence(u)))
            for u, r in response.items()
        ]

        return np.array(discounted_rewards)

    def get_positive_sequence(self, user_id: int):
        user_history = list(zip(*self._history[user_id]))
        ratings_mask = np.array(user_history[-1]) > 0

        return np.array(user_history[0])[ratings_mask].tolist()
    
    def get_policy_sequence(self, user_id: int):
        user_history = list(zip(*self._history[user_id]))
        policy_mask = np.array(user_history[1]) == self.POLICY_SOURCE

        return np.array(user_history[0])[policy_mask].tolist()

    def initialize(self, num_users: int):
        if num_users <= 0:
            raise ValueError(f'number of users must be a positive number')
        
        self._num_users = num_users

        for u in range(self._num_users):
            self._history[u] = []

        self._initialized = True

    def initialize_from_log(self, df: pd.DataFrame):
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        df = df[['userid', 'movieid', 'rating', 'timestamp', 'source']]
        
        num_users = len(df['userid'].unique())

        if df['userid'].max() >= num_users:
            raise ValueError('user ids are not dense')
        
        if df['movieid'].max() >= self._num_items:
            raise ValueError('item ids are not dense')
        
        self.initialize(num_users=num_users)
        
        for row in df.itertuples(index=False, name=None):
            self._history[row[0]].append((row[1], row[4], row[2]))
    
    def get_next_batch(self, batch_size: int):
        self._current_users = self._numpy_generator.choice(
            self._num_users, size=batch_size, replace=False
        )

        return {u : self.get_positive_sequence(u) for u in self._current_users}
    
    @abc.abstractmethod
    def step(
        self,
        actions: Dict[int, int],
        gamma: float = 1.0
    ):
        raise NotImplementedError()
    

class AbstractSasrecSimulator(AbstractSimulator):
    def __init__(
        self,
        sasrec_path: str,
        device: str = 'cpu',
        rewards_gamma: float = 1.0,
        seed: int = None
    ):
        super().__init__(rewards_gamma=rewards_gamma, seed=seed)

        self._device = device

        self._sasrec = torch.load(
            sasrec_path, weights_only=False
        ).to(self._device)
        self._sasrec.eval()

        self._num_items = self._sasrec.item_num

    def initialize(
        self,
        num_users: int,
        init_history_len: int
    ):
        super().initialize(num_users=num_users)

        for u in self._history:
            random_items = self._numpy_generator.choice(
                self._num_items, size=init_history_len, replace=False
            ).tolist()

            for item in random_items:
                self._history[u].append((item, self.INITIAL_SOURCE, 1))

        items_not_in_log = (
            self.item_ids[~np.isin(
                self.item_ids,
                [item for _, seq in self._history.items() for item, _, _ in seq]
            )]
        )

        for iid in tqdm(items_not_in_log):
            rand_user = self._numpy_generator.choice(self._num_users, size=1).item()
            self._history[rand_user].append((iid, self.INITIAL_SOURCE, 1))


class SasrecSimulator(AbstractSasrecSimulator):
    def __init__(
        self,
        n_multinomial: int,
        sasrec_path: str,
        device: str,
        rewards_gamma: float = 1.0,
        seed: int = None
    ):
        super().__init__(
            sasrec_path=sasrec_path,
            device=device,
            rewards_gamma=rewards_gamma,
            seed=seed
        )

        self._n_multinomial = n_multinomial
    
    @torch.no_grad()
    def step(
        self,
        actions: Dict[int, int]
    ):
        response = {}

        for u in self._current_users:
            seq = self.get_positive_sequence(u)
            seq_tensor = torch.LongTensor(seq).to(self._device)
            logits = self._sasrec.score(seq_tensor).flatten().detach().cpu()[:-1]

            logits[seq] = logits.min()
            logits_softmax = torch.nn.functional.softmax(logits, dim=-1).numpy()
            logits_softmax[seq] = 0.0
            logits_softmax = logits_softmax.astype(np.float64) # to avoid p > 1
            logits_softmax = logits_softmax / logits_softmax.sum()

            user_choice = self._numpy_generator.choice(
                self._num_items,
                size=self._n_multinomial,
                replace=False,
                p=logits_softmax
            )

            response[u] = np.isin(actions[u], user_choice).item()

            self._history[u].append((actions[u], self.POLICY_SOURCE, int(response[u])))
            if not response[u]:
                self._history[u].append((user_choice[0], self.OTHER_SOURCE, 1))

        return self._compute_rewards(response=response, gamma=self._gamma)
