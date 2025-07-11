import abc
from typing import Iterable, Union, Dict

from tqdm import tqdm

import torch
import numpy as np
import pandas as pd


class AbstractSimulator(abc.ABC):
    def __init__(self, seed: int = None):
        super().__init__()

        self._seed = seed
        self._numpy_generator = np.random.default_rng(self._seed)

        self._initialized = False
        self._num_items = 0

    @property
    def log_df(self):
        if not self._initialized:
            raise RuntimeError(f'Simulator is not initialized. Call initialize() first')

        max_seq_len = max([len(seq) for _, seq in self._history.items()])

        return pd.concat([
            pd.DataFrame({
                'userid' : [u] * len(seq),
                'movieid' : seq,
                'timestamp' : np.linspace(0, max_seq_len - 1, len(seq)).round().astype(int)
            }) for u, seq in self._history.items()
        ])
    
    @property
    def item_ids(self):
        if not self._initialized:
            raise RuntimeError(f'Simulator is not initialized. Call initialize() first')

        return np.array([*range(self._num_items)])

    def _compute_rewards(
        self,
        response: Dict[int, np.ndarray],
        gamma: float = 1.0,
        count_init_lengths: bool = True
    ):
        rewards = np.array([np.mean(r.astype(int)) for _, r in response.items()])
        lengths = {u : len(self._history[u]) for u in response}

        if not count_init_lengths:
            lengths = {u : lengths[u] - self._initial_seq_lengths[u] for u in response}

        rewards = rewards * gamma ** np.array([l for _, l in lengths.items()])

        return rewards
    
    def _append_history(
        self,
        actions: Dict[int, Iterable[int]],
        response: Dict[int, np.ndarray]
    ):
        for u in self._current_users:
            self._history[u] += (
                np.array(actions[u])[response[u].astype(bool)].tolist()
            )

    def _fix_initial_lenghts(self):
        self._initial_seq_lengths = {u : len(seq) for u, seq in self._history.items()}

    def initialize(self, num_users: int, min_history_len: int):
        if num_users <= 0:
            raise ValueError(f'number of users must be a positive number')
        
        if min_history_len <= 0:
            raise ValueError(f'minimal history length must be a positive number')
        
        self._num_users = num_users
        self._history = {u : [] for u in range(self._num_users)}
        self._fix_initial_lenghts()

        self._initialized = True

    def initialize_from_log(self, df: pd.DataFrame):
        df = df.sort_values(['timestamp']).reset_index(drop=True)
        
        self._num_users = len(df['userid'].unique())

        if self._num_users <= df['userid'].max():
            raise ValueError('user ids are not dense')
        
        if self._num_items <= df['movieid'].max():
            raise ValueError('item ids are not dense')

        history = df.groupby('userid')['movieid'].apply(list)
        self._history = {u : seq for u, seq in history.items()}

        self._fix_initial_lenghts()

        self._initialized = True
    
    def get_history_length(self, user_ids: Iterable = None):
        if user_ids is None:
            user_ids = [u for u in self._history]

        return {u: len(self._history[u]) for u in user_ids}
    
    def start(self, batch_size: int):
        self._batch_size = batch_size

        self._current_users = self._numpy_generator.choice(
            self._num_users, size=self._batch_size, replace=False
        )

        return {u : self._history[u] for u in self._current_users}
    
    @abc.abstractmethod
    def step(
        self,
        actions: Dict[int, Union[int, Iterable[int]]],
        gamma: float = 1.0
    ):
        raise NotImplementedError()
    

class AbstractSasrecSimulator(AbstractSimulator):
    def __init__(
        self,
        sasrec_path: str,
        device: str = 'cpu',
        seed: int = None
    ):
        super().__init__(seed=seed)

        self._device = device

        self._sasrec = torch.load(sasrec_path).to(self._device)
        self._sasrec.eval()

        self._num_items = self._sasrec.item_num

    def initialize(
        self,
        num_users: int,
        min_history_len: int
    ):
        super().initialize(num_users=num_users, min_history_len=min_history_len)

        for u in range(self._num_users):
            self._history[u] += self._numpy_generator.choice(
                self._num_items, size=min_history_len, replace=False
            ).tolist()

        items_not_in_log = (
            self.item_ids[~np.isin(
                self.item_ids,
                [item for _, seq in self._history.items() for item in seq]
            )]
        )

        for iid in tqdm(items_not_in_log):
            rand_user = self._numpy_generator.choice(self._num_users, size=1).item()
            self._history[rand_user].append(iid)

        self._fix_initial_lenghts()


class SasrecSimulator(AbstractSasrecSimulator):
    def __init__(
        self,
        n_multinomial: int,
        sasrec_path: str,
        device: str,
        seed: int = None
    ):
        super().__init__(
            sasrec_path=sasrec_path,
            device=device,
            seed=seed
        )

        self._n_multinomial = n_multinomial
    
    @torch.no_grad()
    def step(
        self,
        actions: Dict[int, Union[int, Iterable[int]]],
        gamma: float = 1.0,
        count_initial_length: bool = True
    ):
        for u, a in actions.items():
            if np.isscalar(a):
                actions[u] = [a]

        response = {}
        for u in self._current_users:
            seq = self._history[u]
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

            response[u] = np.isin(actions[u], user_choice)

            if not response[u].any():
                self._history[u] = self._history[u] + [user_choice[0]]
            else:
                self._history[u] = self._history[u] + np.array(actions[u])[response[u]].tolist()

        rewards = self._compute_rewards(response, gamma, count_initial_length)

        return self.start(self._batch_size), rewards
