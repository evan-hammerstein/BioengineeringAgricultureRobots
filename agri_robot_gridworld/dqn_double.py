from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Tuple, Any
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class ReplayBuffer:
    """
    Stores (potentially n-step) transitions:
      (s, a, R, s2, done, mask2, n_steps)

    mask2 is the VALID ACTION MASK for the next state s2.
    n_steps is how many steps were aggregated into R (1 if standard 1-step DQN).
    """
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, R, s2, done, mask2, n_steps: int):
        self.buffer.append((s, a, R, s2, done, mask2, n_steps))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, R, s2, done, mask2, n_steps = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(R, dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
            np.stack(mask2).astype(np.bool_),
            np.array(n_steps, dtype=np.int64),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Standard MLP Q-network."""
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            #Linear observation of states to hidden
            nn.Linear(obs_dim, 256),
            #ReLu activation
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            #Linear hidden to action Q-values
            nn.Linear(256, num_actions),
        )
    # Forward pass of environment observation through Q-network
    # Produces Q-values for all actions
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN:
      Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
    """
    def __init__(self, obs_dim: int, num_actions: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)                 # [B, 1]
        a = self.advantage(f)             # [B, A]
        a = a - a.mean(dim=1, keepdim=True)
        return v + a


@dataclass
class DQNConfig:
    # Core
    gamma: float = 0.995
    lr: float = 3e-4

    # Replay + training cadence
    replay_size: int = 500_000
    batch_size: int = 64
    learning_starts: int = 10_000
    train_every: int = 4
    target_update_every: int = 2_000

    # If >0, use Polyak averaging for target network updates:
    #   target <- (1-tau)*target + tau*online
    # When tau>0, hard updates via target_update_every are disabled.
    tau: float = 0.0

    # Exploration schedule in *environment steps*
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 1_500_000

    # Stability
    grad_clip_norm: float = 10.0

    # Innovation B toggles
    dueling: bool = False   # Innovation B #2
    n_step: int = 3         # Innovation B #1 (set to 3 or 5)

    # Algorithm toggle
    # True: Double DQN target (online selects, target evaluates)
    # False: Single DQN target (target selects and evaluates via max)
    double_dqn: bool = True


class DoubleDQNAgent:
    """
    Double DQN:
      - Online net selects action: a* = argmax_a Q_online(s', a)
      - Target net evaluates it:     Q_target(s', a*)

    Optional Innovation Bs:
      - dueling=True -> dueling architecture
      - n_step>1 -> n-step returns
    """
    def __init__(self, obs_dim: int, num_actions: int, cfg: DQNConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_actions = num_actions

        Net = DuelingQNetwork if cfg.dueling else QNetwork
        self.q_online = Net(obs_dim, num_actions).to(self.device)
        self.q_target = Net(obs_dim, num_actions).to(self.device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q_online.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size)

        # n-step accumulator buffer (raw 1-step transitions)
        self._nstep_buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = deque()

        self.total_steps = 0  # counts env steps (train_step called each env step)

    def epsilon(self) -> float:
        t = min(self.total_steps / max(1, self.cfg.eps_decay_steps), 1.0)
        return float(self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start))

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: Optional[np.ndarray] = None, greedy: bool = False) -> int:
        if mask is None:
            mask = np.ones(self.num_actions, dtype=np.bool_)

        valid_actions = np.flatnonzero(mask)
        if valid_actions.size == 0:
            valid_actions = np.arange(self.num_actions)

        if (not greedy) and (random.random() < self.epsilon()):
            return int(np.random.choice(valid_actions))

        s = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q_online(s).squeeze(0)  # [A]

        q_masked = q.clone()
        invalid = torch.tensor(~mask, dtype=torch.bool, device=self.device)
        q_masked[invalid] = -1e9
        return int(torch.argmax(q_masked).item())

    def push(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool, mask2: np.ndarray):
        """
        This is called once per environment step.

        If n_step == 1: store directly.
        If n_step > 1: fold into n-step returns before storing.
        """
        # - keep a small buffer of recent 1-step transitions
        n = int(max(1, self.cfg.n_step))
        if n == 1:
            self.replay.push(s, int(a), float(r), s2, bool(done), mask2, 1)
            return
        
        self._nstep_buf.append((s, int(a), float(r), s2, bool(done), mask2))
        # - once we collect 3 steps, fold them into ONE replay entry
        # - if episode ends early, flush partial n-step returns so nothing is wasted
        def fold_and_push(k: int):
            R = 0.0
            gamma = float(self.cfg.gamma)
            done_k = False
            s0, a0, *_ = self._nstep_buf[0]
            sK = None
            maskK = None
            used = 0
            # Instead of learning from only the next reward r_t, we learn from the next k rewards:
            for i in range(k):
                si, ai, ri, s2i, di, mask2i = self._nstep_buf[i]
                R += (gamma ** i) * float(ri)
                used += 1
                sK = s2i
                maskK = mask2i
                if di:
                    done_k = True
                    break

            assert sK is not None and maskK is not None
            self.replay.push(s0, int(a0), float(R), sK, bool(done_k), maskK, int(used))

        if len(self._nstep_buf) >= n:
            fold_and_push(n)
            self._nstep_buf.popleft()

        if done:
            while len(self._nstep_buf) > 0:
                fold_and_push(len(self._nstep_buf))
                self._nstep_buf.popleft()

    def train_step(self) -> Optional[float]:
        self.total_steps += 1

        if len(self.replay) < self.cfg.learning_starts:
            return None
        if self.total_steps % self.cfg.train_every != 0:
            return None

        s, a, R, s2, done, mask2, n_steps = self.replay.sample(self.cfg.batch_size)

        s_t = torch.tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.tensor(a, dtype=torch.int64, device=self.device)
        R_t = torch.tensor(R, dtype=torch.float32, device=self.device)
        s2_t = torch.tensor(s2, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device)
        mask2_t = torch.tensor(mask2, dtype=torch.bool, device=self.device)
        n_t = torch.tensor(n_steps, dtype=torch.float32, device=self.device)

        # Defensive: if the env ever reports an all-false mask for a non-terminal next state,
        # masking would force Q(s',a) to -1e9 and blow up TD targets/loss. Treat as "no mask".
        # (The action-selection path already falls back to all-actions-valid in this case.)
        if mask2_t.ndim == 2:
            no_valid = ~mask2_t.any(dim=1)
            if bool(no_valid.any()):
                mask2_t = mask2_t.clone()
                mask2_t[no_valid] = True

        q_all = self.q_online(s_t)
        q_sa = q_all.gather(1, a_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if bool(self.cfg.double_dqn):
                # Double DQN selection (online) with mask
                next_q_online = self.q_online(s2_t).masked_fill(~mask2_t, -1e9)
                next_actions = torch.argmax(next_q_online, dim=1)

                # Double DQN evaluation (target)
                next_q_target = self.q_target(s2_t)
                next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Single DQN target: max_a Q_target(s', a) among valid actions
                next_q_target = self.q_target(s2_t).masked_fill(~mask2_t, -1e9)
                next_q = torch.max(next_q_target, dim=1).values

            gamma_n = torch.pow(torch.tensor(float(self.cfg.gamma), device=self.device), n_t)
            # Bellman target calculation, this is where the n-step return is used!
            y = R_t + gamma_n * (1.0 - done_t) * next_q

        loss = nn.functional.smooth_l1_loss(q_sa, y)

        self.optim.zero_grad()
        loss.backward()
        if float(self.cfg.grad_clip_norm) and float(self.cfg.grad_clip_norm) > 0.0:
            nn.utils.clip_grad_norm_(self.q_online.parameters(), float(self.cfg.grad_clip_norm))
        self.optim.step()

        tau = float(getattr(self.cfg, "tau", 0.0) or 0.0)
        if tau > 0.0:
            # Soft update each training step
            with torch.no_grad():
                for p_t, p_o in zip(self.q_target.parameters(), self.q_online.parameters()):
                    p_t.data.mul_(1.0 - tau).add_(tau * p_o.data)
        else:
            if self.total_steps % self.cfg.target_update_every == 0:
                self.q_target.load_state_dict(self.q_online.state_dict())

        return float(loss.item())

    def save(self, path: str):
        payload = {
            "state_dict": self.q_online.state_dict(),
            "dueling": bool(self.cfg.dueling),
            "double_dqn": bool(self.cfg.double_dqn),
            "num_actions": int(self.num_actions),
        }
        torch.save(payload, path)

    def load(self, path: str):
        obj: Any = torch.load(path, map_location=self.device)
        if isinstance(obj, dict) and "state_dict" in obj:
            self.q_online.load_state_dict(obj["state_dict"])
        else:
            self.q_online.load_state_dict(obj)

        self.q_target.load_state_dict(self.q_online.state_dict())
