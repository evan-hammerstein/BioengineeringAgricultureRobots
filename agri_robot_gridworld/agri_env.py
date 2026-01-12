from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config


class Terrain(IntEnum):
    TILLED = 0
    WALKWAY = 1
    WATER = 2
    HOLE = 3
    CHEM_TANK = 4
    DROPOFF = 5
    CHARGER = 6


class Action(IntEnum):
    # Movement
    MOVE_N = 0
    MOVE_E = 1
    MOVE_S = 2
    MOVE_W = 3

    # Spray weed
    SPRAY_N = 4
    SPRAY_E = 5
    SPRAY_S = 6
    SPRAY_W = 7

    # Water tile
    WATER_N = 8
    WATER_E = 9
    WATER_S = 10
    WATER_W = 11

    # Fertilize tile
    FERT_N = 12
    FERT_E = 13
    FERT_S = 14
    FERT_W = 15

    # Plant tile
    PLANT_N = 16
    PLANT_E = 17
    PLANT_S = 18
    PLANT_W = 19

    # Harvest tile
    HARVEST_N = 20
    HARVEST_E = 21
    HARVEST_S = 22
    HARVEST_W = 23

    # Refill tanks (adjacent to chem/water)
    REFILL_WEED = 24
    REFILL_FERT = 25
    REFILL_WATER = 26

    # Dropoff produce
    DROPOFF_PRODUCE = 27


DIR: Dict[Action, Tuple[int, int]] = {
    Action.MOVE_N: (0, -1),
    Action.MOVE_E: (1, 0),
    Action.MOVE_S: (0, 1),
    Action.MOVE_W: (-1, 0),

    Action.SPRAY_N: (0, -1),
    Action.SPRAY_E: (1, 0),
    Action.SPRAY_S: (0, 1),
    Action.SPRAY_W: (-1, 0),

    Action.WATER_N: (0, -1),
    Action.WATER_E: (1, 0),
    Action.WATER_S: (0, 1),
    Action.WATER_W: (-1, 0),

    Action.FERT_N: (0, -1),
    Action.FERT_E: (1, 0),
    Action.FERT_S: (0, 1),
    Action.FERT_W: (-1, 0),

    Action.PLANT_N: (0, -1),
    Action.PLANT_E: (1, 0),
    Action.PLANT_S: (0, 1),
    Action.PLANT_W: (-1, 0),

    Action.HARVEST_N: (0, -1),
    Action.HARVEST_E: (1, 0),
    Action.HARVEST_S: (0, 1),
    Action.HARVEST_W: (-1, 0),
}


class AgriRobotEnv(gym.Env):
    """
    Updated env features (important):
      1) Feature toggles (weeds, saturation, fertilizer, spraying, watering, holes, upkeep, etc.)
      2) Action mask includes return-to-charger feasibility using BFS distance.
      3) Optional "strict endgame return" mask to reduce missed-charger near the end of the day.
      4) Observation: fixed 3x3 local view + "memory map" of last-seen crop/weed/saturation/fertilized.
         (This is what you wanted: keep 3x3 current vision, but remember last seen elsewhere.)
      5) Reward: delta net worth (money + pending value in produce storage).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # ---- layout
        self.w = int(cfg.GRID_W)
        self.h = int(cfg.GRID_H)

        # Terrain grid
        self.terrain = np.zeros((self.h, self.w), dtype=np.int32)

        # Plant stage on tilled tiles: 0 empty, 1..5 stages
        self.plant_stage = np.zeros((self.h, self.w), dtype=np.int32)

        # Weed presence: 0/1
        self.weed = np.zeros((self.h, self.w), dtype=np.int8)

        # Fertilized: 0/1
        self.fertilized = np.zeros((self.h, self.w), dtype=np.int8)

        # Soil saturation: 0..1
        self.saturation = np.zeros((self.h, self.w), dtype=np.float32)

        # Dry-night counter for death logic
        self.dry_nights = np.zeros((self.h, self.w), dtype=np.int32)

        # Mature days counter (stage 5)
        self.mature_days = np.zeros((self.h, self.w), dtype=np.int32)

        # Positions
        self.pos_charger = (0, 0)
        self.pos_dropoff = (0, 0)
        self.pos_chem = (0, 0)

        # Robot state
        self.robot_x = 0
        self.robot_y = 0
        self.day = 1
        self.actions_left_today = int(cfg.BATTERY_ACTIONS_PER_DAY)

        # Tanks + storage
        self.weed_tank = int(cfg.WEED_TANK_MAX)
        self.fert_tank = int(cfg.FERT_TANK_MAX)
        self.water_tank = int(cfg.WATER_TANK_MAX)

        self.produce_count = 0
        self.produce_value_pending = 0.0  # harvested but not yet sold (pending $)

        self.money = 0.0

        # Refill constraints
        self.refilled_weed_today = False
        self.refilled_fert_today = False
        self.refilled_water_today = False

        # Charger miss -> disable the next day
        self.disabled_days_remaining = 0
        self.disabled_today = False

        # Debug counters
        self.missed_charger_days = 0
        self.disabled_days_total = 0
        self.missed_charger_today = False

        # Action mask settings
        self.invalid_action_penalty = -0.05
        self.deterministic_noop_penalty = -0.01
        self.stochastic_noop_penalty = -0.002

        # Reward shaping bookkeeping (set during action execution)
        self._last_trample_count = 0
        self._last_dropoff_count = 0

        # Return feasibility
        self.return_safety_margin = 1
        self.strict_endgame_return = True
        self.strict_endgame_when_remaining_leq = 4

        # Internal BFS distances to charger
        self._dist_to_charger = None
        self._dist_valid = False

        # Memory maps for last-seen world state (agent observation)
        # Unknown marked as -1 for plant_stage and saturation, 0 for weed/fert (optional).
        self.mem_plant_stage = -np.ones((self.h, self.w), dtype=np.int32)
        self.mem_saturation = -np.ones((self.h, self.w), dtype=np.float32)
        self.mem_weed = -np.ones((self.h, self.w), dtype=np.int8)
        self.mem_fertilized = -np.ones((self.h, self.w), dtype=np.int8)

        self._generate_world()

        obs_dim = self._get_obs().shape[0]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Action))

    def _flag(self, name: str, default: bool) -> bool:
        return bool(getattr(self.cfg, name, default))

    def _net_worth(self) -> float:
        return float(self.money + self.produce_value_pending)

    def _generate_world(self):
        cfg = self.cfg

        probs = np.array(
            [cfg.P_TILLED, cfg.P_WALKWAY, cfg.P_WATER, cfg.P_HOLE, cfg.P_CHEM_TANK, cfg.P_DROPOFF, cfg.P_CHARGER],
            dtype=np.float64
        )
        probs = probs / probs.sum()

        flat = self.rng.choice(np.arange(len(probs)), size=self.w * self.h, p=probs)
        self.terrain = flat.reshape((self.h, self.w)).astype(np.int32)

        # Force special tiles: place charger, then dropoff, then chem (adjacent layout)
        # We intentionally cluster these 3 for easier learning.
        # Pick a charger position on a non-blocked tile.
        cx, cy = self.w // 2, self.h // 2
        self.pos_charger = (cx, cy)
        self.terrain[cy, cx] = int(Terrain.CHARGER)

        # Adjacent positions for dropoff and chem (try east and south; fallback to any valid)
        cand = [(cx + 1, cy), (cx, cy + 1), (cx - 1, cy), (cx, cy - 1)]
        def inb(p): return 0 <= p[0] < self.w and 0 <= p[1] < self.h
        cand = [p for p in cand if inb(p)]

        if len(cand) >= 2:
            self.pos_dropoff = cand[0]
            self.pos_chem = cand[1]
        elif len(cand) == 1:
            self.pos_dropoff = cand[0]
            self.pos_chem = (cx, cy)
        else:
            self.pos_dropoff = (cx, cy)
            self.pos_chem = (cx, cy)

        dx, dy = self.pos_dropoff
        kx, ky = self.pos_chem
        self.terrain[dy, dx] = int(Terrain.DROPOFF)
        self.terrain[ky, kx] = int(Terrain.CHEM_TANK)

        # Reset state grids
        self.plant_stage.fill(0)
        self.weed.fill(0)
        self.fertilized.fill(0)
        self.saturation.fill(0.5)
        self.dry_nights.fill(0)
        self.mature_days.fill(0)

        # Reset memory maps
        self.mem_plant_stage.fill(-1)
        self.mem_saturation.fill(-1)
        self.mem_weed.fill(-1)
        self.mem_fertilized.fill(-1)

        # Reset robot at charger
        self.robot_x, self.robot_y = self.pos_charger
        self.day = 1
        self.actions_left_today = int(self.cfg.BATTERY_ACTIONS_PER_DAY)

        self.weed_tank = int(self.cfg.WEED_TANK_MAX)
        self.fert_tank = int(self.cfg.FERT_TANK_MAX)
        self.water_tank = int(self.cfg.WATER_TANK_MAX)

        self.produce_count = 0
        self.produce_value_pending = 0.0
        self.money = 0.0

        self.refilled_weed_today = False
        self.refilled_fert_today = False
        self.refilled_water_today = False

        self.disabled_days_remaining = 0
        self.disabled_today = False

        # Debug counters
        self.missed_charger_days = 0
        self.disabled_days_total = 0
        self.missed_charger_today = False

        self._dist_valid = False

        # Initialize memory with starting view
        self._update_memory_from_view()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._generate_world()

        obs = self._get_obs()
        info = {
            "final_money": float(self.money),
            "final_net_worth": float(self._net_worth()),
            "action_mask": self.valid_action_mask().astype(np.bool_),
            "missed_charger_days": int(self.missed_charger_days),
            "missed_charger_today": bool(self.missed_charger_today),
        }
        return obs, info

    # -----------------------
    # Geometry / neighbors
    # -----------------------
    def _in_bounds(self, x: int, y: int) -> bool:
        return (0 <= x < self.w) and (0 <= y < self.h)

    def _passable(self, x: int, y: int) -> bool:
        t = Terrain(self.terrain[y, x])
        if t in (Terrain.HOLE, Terrain.WATER, Terrain.DROPOFF, Terrain.CHEM_TANK):
            return False
        return True

    def _is_adjacent_to(self, terr: Terrain) -> bool:
        x, y = self.robot_x, self.robot_y
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if self._in_bounds(nx, ny) and Terrain(self.terrain[ny, nx]) == terr:
                return True
        return False

    def _dir_target(self, a: Action) -> Optional[Tuple[int, int]]:
        dx, dy = DIR.get(a, (0, 0))
        x, y = self.robot_x + dx, self.robot_y + dy
        if not self._in_bounds(x, y):
            return None
        return (x, y)

    # -----------------------
    # Distance to charger (BFS)
    # -----------------------
    def _compute_dist_to_charger(self):
        w, h = self.w, self.h
        dist = np.full((h, w), np.inf, dtype=np.float32)
        cx, cy = self.pos_charger
        dist[cy, cx] = 0.0

        q = [(cx, cy)]
        head = 0
        while head < len(q):
            x, y = q[head]
            head += 1
            d = dist[y, x]
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if not self._in_bounds(nx, ny):
                    continue
                if (nx, ny) != (cx, cy) and (not self._passable(nx, ny)):
                    continue
                nd = d + 1.0
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    q.append((nx, ny))

        self._dist_to_charger = dist
        self._dist_valid = True

    def _current_dist_to_charger(self) -> float:
        if not self._dist_valid:
            self._compute_dist_to_charger()
        return float(self._dist_to_charger[self.robot_y, self.robot_x])

    def _dist_after_action_assuming_success(self, a: Action) -> float:
        if not self._dist_valid:
            self._compute_dist_to_charger()

        x, y = self.robot_x, self.robot_y
        if a in (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W):
            dx, dy = DIR[a]
            nx, ny = x + dx, y + dy
            if not self._in_bounds(nx, ny):
                return float("inf")
            if not self._passable(nx, ny):
                return float("inf")
            return float(self._dist_to_charger[ny, nx])
        else:
            return float(self._dist_to_charger[y, x])

    # -----------------------
    # Action mask
    # -----------------------
    def valid_action_mask(self) -> np.ndarray:
        n = self.action_space.n
        mask = np.zeros(n, dtype=np.bool_)

        # If disabled or no actions left, allow everything (step() ends day anyway)
        if self.disabled_today or self.actions_left_today <= 0:
            mask[:] = True
            return mask

        x, y = self.robot_x, self.robot_y
        remaining_after = max(0, int(self.actions_left_today) - 1)

        # With stochastic move fails, require extra buffer moves near end-of-day.
        safety = max(0, int(self.return_safety_margin))

        def feasible(a: Action) -> bool:
            d2 = self._dist_after_action_assuming_success(a)
            return np.isfinite(d2) and (d2 + safety <= float(remaining_after))

        # Strict endgame logic: when close to end-of-day and not on charger, force
        # only moves that reduce distance to charger.
        strict_now = False
        if self.strict_endgame_return and remaining_after <= int(self.strict_endgame_when_remaining_leq):
            if (x, y) != self.pos_charger:
                strict_now = True

        cur_d = self._current_dist_to_charger()

        def is_distance_reducing_move(a: Action) -> bool:
            if a not in (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W):
                return False
            d2 = self._dist_after_action_assuming_success(a)
            return np.isfinite(d2) and (d2 < cur_d - 1e-6)

        # Moves
        for a in (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W):
            dx, dy = DIR[a]
            nx, ny = x + dx, y + dy
            if not (self._in_bounds(nx, ny) and self._passable(nx, ny)):
                continue
            if not feasible(a):
                continue
            if strict_now:
                if not is_distance_reducing_move(a) and (nx, ny) != self.pos_charger:
                    continue
            mask[int(a)] = True

        allow_non_move = not strict_now or (x, y) == self.pos_charger

        def tgt(a: Action) -> Optional[Tuple[int, int]]:
            return self._dir_target(a)

        # Spray
        if allow_non_move and self._flag("ENABLE_SPRAYING", True) and self.weed_tank > 0:
            for a in (Action.SPRAY_N, Action.SPRAY_E, Action.SPRAY_S, Action.SPRAY_W):
                if not feasible(a):
                    continue
                p = tgt(a)
                if p is None:
                    continue
                tx, ty = p
                if Terrain(self.terrain[ty, tx]) == Terrain.TILLED:
                    # Only allow spraying if there is a weed present (reduces pointless spraying)
                    if self._flag("ENABLE_WEEDS", True) and self.weed[ty, tx] == 1:
                        mask[int(a)] = True

        # Water
        if allow_non_move and self._flag("ENABLE_WATERING", True) and self.water_tank > 0:
            for a in (Action.WATER_N, Action.WATER_E, Action.WATER_S, Action.WATER_W):
                if not feasible(a):
                    continue
                p = tgt(a)
                if p is None:
                    continue
                tx, ty = p
                if Terrain(self.terrain[ty, tx]) == Terrain.TILLED:
                    mask[int(a)] = True

        # Fert
        if allow_non_move and self._flag("ENABLE_FERTILIZER", True) and self.fert_tank > 0:
            for a in (Action.FERT_N, Action.FERT_E, Action.FERT_S, Action.FERT_W):
                if not feasible(a):
                    continue
                p = tgt(a)
                if p is None:
                    continue
                tx, ty = p
                if Terrain(self.terrain[ty, tx]) == Terrain.TILLED:
                    mask[int(a)] = True

        # Plant
        if allow_non_move:
            for a in (Action.PLANT_N, Action.PLANT_E, Action.PLANT_S, Action.PLANT_W):
                if not feasible(a):
                    continue
                p = tgt(a)
                if p is None:
                    continue
                tx, ty = p
                if Terrain(self.terrain[ty, tx]) == Terrain.TILLED and self.plant_stage[ty, tx] == 0:
                    mask[int(a)] = True

        # Harvest (maturity-gated)
        if allow_non_move and self.produce_count < self.cfg.PRODUCE_STORAGE_MAX:
            for a in (Action.HARVEST_N, Action.HARVEST_E, Action.HARVEST_S, Action.HARVEST_W):
                if not feasible(a):
                    continue
                p = tgt(a)
                if p is None:
                    continue
                tx, ty = p
                if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
                    continue
                stage = int(self.plant_stage[ty, tx])
                if stage <= 0:
                    continue
                if self._flag("ENABLE_MATURITY_GATE", True):
                    min_stage = int(getattr(self.cfg, "MIN_HARVEST_STAGE", 3))
                    if stage < min_stage:
                        continue
                mask[int(a)] = True

        # Refills
        if allow_non_move:
            near_chem = self._is_adjacent_to(Terrain.CHEM_TANK)
            near_water = self._is_adjacent_to(Terrain.WATER)

            if self._flag("ENABLE_SPRAYING", True) and near_chem and (not self.refilled_weed_today) and feasible(Action.REFILL_WEED):
                if self.cfg.ALLOW_PARTIAL_REFILL_WEED or self.weed_tank == 0:
                    mask[int(Action.REFILL_WEED)] = True

            if self._flag("ENABLE_FERTILIZER", True) and near_chem and (not self.refilled_fert_today) and feasible(Action.REFILL_FERT):
                if self.cfg.ALLOW_PARTIAL_REFILL_FERT or self.fert_tank == 0:
                    mask[int(Action.REFILL_FERT)] = True

            if self._flag("ENABLE_WATERING", True) and near_water and (not self.refilled_water_today) and feasible(Action.REFILL_WATER):
                if self.cfg.ALLOW_PARTIAL_REFILL_WATER or self.water_tank == 0:
                    mask[int(Action.REFILL_WATER)] = True

        # Dropoff
        if allow_non_move and feasible(Action.DROPOFF_PRODUCE):
            if self._is_adjacent_to(Terrain.DROPOFF) and self.produce_count > 0:
                mask[int(Action.DROPOFF_PRODUCE)] = True

        return mask

    # -----------------------
    # Observation (3x3 view + memory)
    # -----------------------
    def _update_memory_from_view(self):
        cx, cy = self.robot_x, self.robot_y
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                x, y = cx + dx, cy + dy
                if not self._in_bounds(x, y):
                    continue

                if Terrain(self.terrain[y, x]) == Terrain.TILLED:
                    self.mem_plant_stage[y, x] = int(self.plant_stage[y, x])
                    self.mem_weed[y, x] = int(self.weed[y, x])
                    self.mem_fertilized[y, x] = int(self.fertilized[y, x])
                    self.mem_saturation[y, x] = float(self.saturation[y, x])
                else:
                    # For non-tilled tiles, we don't care about plant metrics; keep them as-is.
                    pass

    def _get_local_view(self):
        cx, cy = self.robot_x, self.robot_y
        terr = np.full((3, 3), -1, dtype=np.int32)
        pstage = np.full((3, 3), -1, dtype=np.int32)
        weed = np.full((3, 3), -1, dtype=np.int32)
        fert = np.full((3, 3), -1, dtype=np.int32)
        sat = np.full((3, 3), -1.0, dtype=np.float32)

        for j, dy in enumerate((-1, 0, 1)):
            for i, dx in enumerate((-1, 0, 1)):
                x, y = cx + dx, cy + dy
                if not self._in_bounds(x, y):
                    continue
                terr[j, i] = int(self.terrain[y, x])

                if Terrain(self.terrain[y, x]) == Terrain.TILLED:
                    pstage[j, i] = int(self.plant_stage[y, x])
                    weed[j, i] = int(self.weed[y, x])
                    fert[j, i] = int(self.fertilized[y, x])
                    sat[j, i] = float(self.saturation[y, x])
                else:
                    pstage[j, i] = 0
                    weed[j, i] = 0
                    fert[j, i] = 0
                    sat[j, i] = 0.0

        return terr, pstage, weed, fert, sat

    def _get_obs(self) -> np.ndarray:
        terr, pstage, weed, fert, sat = self._get_local_view()

        # Normalize / encode
        terr_norm = terr.astype(np.float32) / float(len(Terrain) - 1)
        pstage_norm = pstage.astype(np.float32) / 5.0
        weed_norm = weed.astype(np.float32)
        fert_norm = fert.astype(np.float32)
        sat_norm = sat.astype(np.float32)

        # Robot scalars
        rx = float(self.robot_x) / max(1.0, float(self.w - 1))
        ry = float(self.robot_y) / max(1.0, float(self.h - 1))
        day_norm = float(self.day) / max(1.0, float(self.cfg.NUM_DAYS))
        act_norm = float(self.actions_left_today) / max(1.0, float(self.cfg.BATTERY_ACTIONS_PER_DAY))

        tanks = np.array([
            float(self.weed_tank) / max(1.0, float(self.cfg.WEED_TANK_MAX)),
            float(self.fert_tank) / max(1.0, float(self.cfg.FERT_TANK_MAX)),
            float(self.water_tank) / max(1.0, float(self.cfg.WATER_TANK_MAX)),
        ], dtype=np.float32)

        storage = np.array([
            float(self.produce_count) / max(1.0, float(self.cfg.PRODUCE_STORAGE_MAX)),
        ], dtype=np.float32)

        # Memory map (last-seen), normalized
        mem_p = np.clip(self.mem_plant_stage.astype(np.float32), -1.0, 5.0) / 5.0
        mem_w = np.clip(self.mem_weed.astype(np.float32), -1.0, 1.0)
        mem_f = np.clip(self.mem_fertilized.astype(np.float32), -1.0, 1.0)
        mem_s = np.clip(self.mem_saturation.astype(np.float32), -1.0, 1.0)

        obs = np.concatenate([
            terr_norm.flatten(),
            pstage_norm.flatten(),
            weed_norm.flatten(),
            fert_norm.flatten(),
            sat_norm.flatten(),
            np.array([rx, ry, day_norm, act_norm], dtype=np.float32),
            tanks,
            storage,
            mem_p.flatten(),
            mem_w.flatten(),
            mem_f.flatten(),
            mem_s.flatten(),
        ]).astype(np.float32)

        return obs

    # -----------------------
    # Step + action execution
    # -----------------------
    def step(self, action: int):
        cfg = self.cfg

        # Episode termination condition: after day > NUM_DAYS
        if self.day > cfg.NUM_DAYS:
            obs = self._get_obs()
            info = {
                "final_money": float(self.money),
                "final_net_worth": float(self._net_worth()),
                "action_mask": self.valid_action_mask().astype(np.bool_),
                "missed_charger_days": int(self.missed_charger_days),
                "missed_charger_today": bool(self.missed_charger_today),
            }
            return obs, 0.0, True, False, info

        # Disabled day: no actions, immediately end day
        if self.disabled_today or self.actions_left_today <= 0:
            before = self._net_worth()
            self._end_day()
            after = self._net_worth()
            obs = self._get_obs()
            info = {
                "disabled_today": True,
                "missed_charger_today": bool(self.missed_charger_today),
                "final_money": float(self.money),
                "final_net_worth": float(after),
                "action_mask": self.valid_action_mask().astype(np.bool_),
                "missed_charger_days": int(self.missed_charger_days),
            }
            reward = float(after - before)
            return obs, reward, (self.day > cfg.NUM_DAYS), False, info

        a = Action(int(action))

        before_worth = self._net_worth()

        # Reset per-step shaping side channels
        self._last_trample_count = 0
        self._last_dropoff_count = 0

        # Action mask check
        mask = self.valid_action_mask()
        invalid_action = (mask[int(a)] == 0)

        deterministic_noop = False
        stochastic_noop = False
        outcome = "invalid_noop"

        # Execute action, but if invalid -> noop
        if invalid_action:
            deterministic_noop = True
        else:
            outcome = self._apply_action(a)
            if outcome == "invalid_noop":
                deterministic_noop = True
            elif outcome == "stochastic_noop":
                stochastic_noop = True

        # Battery always decreases for a chosen action
        self.actions_left_today -= 1

        # End of day?
        if self.actions_left_today <= 0:
            self._end_day()

        # Update memory based on new position/view
        self._update_memory_from_view()

        after_worth = self._net_worth()
        reward = float(after_worth - before_worth)

        # Apply penalties (separate buckets)
        if invalid_action:
            reward += float(self.invalid_action_penalty)
        elif deterministic_noop:
            reward += float(self.deterministic_noop_penalty)
        elif stochastic_noop:
            reward += float(self.stochastic_noop_penalty)

        # -----------------------
        # Reward shaping (optional)
        # -----------------------
        # Small per-step cost to reduce wandering
        if self._flag("ENABLE_STEP_PENALTY", False):
            reward += float(getattr(cfg, "STEP_PENALTY", 0.0))

        # Immediate planting bonus (helps delayed credit)
        if self._flag("ENABLE_PLANT_BONUS", False):
            if (a in (Action.PLANT_N, Action.PLANT_E, Action.PLANT_S, Action.PLANT_W)) and (not invalid_action) and (not deterministic_noop):
                # Outcome is "success" only when a plant was actually placed
                if outcome == "success":
                    reward += float(getattr(cfg, "PLANT_BONUS", 0.0))

        # Trample penalty when movement destroys an existing plant/weed on tilled
        if self._flag("ENABLE_TRAMPLE_PENALTY", False) and self._last_trample_count > 0:
            reward += float(getattr(cfg, "TRAMPLE_PENALTY", 0.0)) * float(self._last_trample_count)

        # Dropoff bonus (dropoff itself doesn't change net worth)
        if self._flag("ENABLE_DROPOFF_BONUS", False) and self._last_dropoff_count > 0:
            reward += float(getattr(cfg, "DROPOFF_BONUS_PER_ITEM", 0.0)) * float(self._last_dropoff_count)

        obs = self._get_obs()
        terminated = self.day > cfg.NUM_DAYS
        info = {
            "final_money": float(self.money),
            "final_net_worth": float(self._net_worth()),
            "action_mask": self.valid_action_mask().astype(np.bool_),
            "invalid_action": bool(invalid_action),
            "deterministic_noop": bool(deterministic_noop),
            "stochastic_noop": bool(stochastic_noop),
            "missed_charger_days": int(self.missed_charger_days),
            "missed_charger_today": bool(self.missed_charger_today),
            "disabled_days_total": int(self.disabled_days_total),
            "dist_to_charger": float(self._current_dist_to_charger()),
        }
        return obs, reward, terminated, False, info

    # -----------------------
    # Action implementations
    # -----------------------
    def _apply_action(self, a: Action) -> str:
        if a in (Action.MOVE_N, Action.MOVE_E, Action.MOVE_S, Action.MOVE_W):
            return self._do_move(a)

        if a in (Action.SPRAY_N, Action.SPRAY_E, Action.SPRAY_S, Action.SPRAY_W):
            return self._do_spray(a)

        if a in (Action.WATER_N, Action.WATER_E, Action.WATER_S, Action.WATER_W):
            return self._do_water(a)

        if a in (Action.FERT_N, Action.FERT_E, Action.FERT_S, Action.FERT_W):
            return self._do_fert(a)

        if a in (Action.PLANT_N, Action.PLANT_E, Action.PLANT_S, Action.PLANT_W):
            return self._do_plant(a)

        if a in (Action.HARVEST_N, Action.HARVEST_E, Action.HARVEST_S, Action.HARVEST_W):
            return self._do_harvest(a)

        if a == Action.REFILL_WEED:
            return self._do_refill_weed()

        if a == Action.REFILL_FERT:
            return self._do_refill_fert()

        if a == Action.REFILL_WATER:
            return self._do_refill_water()

        if a == Action.DROPOFF_PRODUCE:
            return self._do_dropoff()

        return "invalid_noop"

    def _do_move(self, a: Action) -> str:
        dx, dy = DIR[a]
        nx, ny = self.robot_x + dx, self.robot_y + dy

        if not self._in_bounds(nx, ny):
            return "invalid_noop"
        if not self._passable(nx, ny):
            return "invalid_noop"

        if self._flag("ENABLE_MOVE_FAIL_FROM_LOAD", True):
            p_fail = float(self.cfg.MOVE_FAIL_PER_PRODUCE) * float(self.produce_count)
            if self.rng.random() < p_fail:
                return "stochastic_noop"

        # Moving into tilled can kill what's in it (trampling)
        if self._flag("ENABLE_TILLED_STEP_KILL", True) and Terrain(self.terrain[ny, nx]) == Terrain.TILLED:
            if self.rng.random() < float(self.cfg.TILLED_STEP_KILL_PROB):
                trampled = 0
                if int(self.plant_stage[ny, nx]) > 0:
                    trampled += 1
                if int(self.weed[ny, nx]) > 0:
                    trampled += 1

                if trampled > 0:
                    self._last_trample_count += trampled

                self.plant_stage[ny, nx] = 0
                self.weed[ny, nx] = 0
                self.fertilized[ny, nx] = 0
                self.dry_nights[ny, nx] = 0
                self.mature_days[ny, nx] = 0

        self.robot_x, self.robot_y = nx, ny
        return "success"

    def _do_spray(self, a: Action) -> str:
        if not self._flag("ENABLE_SPRAYING", True):
            return "invalid_noop"
        if self.weed_tank <= 0:
            return "invalid_noop"

        tgt = self._dir_target(a)
        if tgt is None:
            return "invalid_noop"
        tx, ty = tgt
        if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
            return "invalid_noop"

        # Only meaningful if weeds are enabled and present
        if not self._flag("ENABLE_WEEDS", True):
            return "invalid_noop"
        if self.weed[ty, tx] == 0:
            return "invalid_noop"

        self.weed_tank -= 1

        # Probabilistic kill
        if self.rng.random() < float(self.cfg.SPRAY_KILL_WEED_PROB):
            self.weed[ty, tx] = 0
        if self.plant_stage[ty, tx] > 0 and self.rng.random() < float(self.cfg.SPRAY_KILL_PLANT_PROB):
            self.plant_stage[ty, tx] = 0
            self.fertilized[ty, tx] = 0
            self.dry_nights[ty, tx] = 0
            self.mature_days[ty, tx] = 0

        return "success"

    def _do_water(self, a: Action) -> str:
        if not self._flag("ENABLE_WATERING", True):
            return "invalid_noop"
        if self.water_tank <= 0:
            return "invalid_noop"

        tgt = self._dir_target(a)
        if tgt is None:
            return "invalid_noop"
        tx, ty = tgt
        if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
            return "invalid_noop"

        self.water_tank -= 1
        self.saturation[ty, tx] = float(np.clip(self.saturation[ty, tx] + float(self.cfg.WATER_ACTION_BOOST), 0.0, 1.0))
        return "success"

    def _do_fert(self, a: Action) -> str:
        if not self._flag("ENABLE_FERTILIZER", True):
            return "invalid_noop"
        if self.fert_tank <= 0:
            return "invalid_noop"

        tgt = self._dir_target(a)
        if tgt is None:
            return "invalid_noop"
        tx, ty = tgt
        if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
            return "invalid_noop"

        self.fert_tank -= 1
        self.fertilized[ty, tx] = 1
        return "success"

    def _do_plant(self, a: Action) -> str:
        tgt = self._dir_target(a)
        if tgt is None:
            return "invalid_noop"
        tx, ty = tgt
        if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
            return "invalid_noop"
        if self.plant_stage[ty, tx] != 0:
            return "invalid_noop"

        # Pay seed cost immediately (can go negative)
        self.money -= float(self.cfg.SEED_COST)

        self.plant_stage[ty, tx] = 1
        self.fertilized[ty, tx] = 0
        self.dry_nights[ty, tx] = 0
        self.mature_days[ty, tx] = 0
        return "success"

    def _do_harvest(self, a: Action) -> str:
        cfg = self.cfg
        if self.produce_count >= cfg.PRODUCE_STORAGE_MAX:
            return "invalid_noop"

        tgt = self._dir_target(a)
        if tgt is None:
            return "invalid_noop"
        tx, ty = tgt
        if Terrain(self.terrain[ty, tx]) != Terrain.TILLED:
            return "invalid_noop"

        stage = int(self.plant_stage[ty, tx])
        if stage <= 0:
            return "invalid_noop"

        # Optional: maturity gate (prevents "plant then immediately harvest" degenerate behavior).
        if self._flag("ENABLE_MATURITY_GATE", True):
            min_stage = int(getattr(cfg, "MIN_HARVEST_STAGE", 3))
            if stage < min_stage:
                return "invalid_noop"

        self.produce_count += 1

        # Stage value: BASE_PRODUCE_VALUE * max(0, stage-2)
        self.produce_value_pending += float(cfg.BASE_PRODUCE_VALUE) * max(0, (float(stage) - 2))

        # reset tile
        self.plant_stage[ty, tx] = 0
        self.fertilized[ty, tx] = 0
        self.dry_nights[ty, tx] = 0
        self.mature_days[ty, tx] = 0
        return "success"

    def _do_dropoff(self) -> str:
        if not self._is_adjacent_to(Terrain.DROPOFF):
            return "invalid_noop"
        if self.produce_count <= 0:
            return "invalid_noop"

        # For shaping: remember how many items were dropped off
        self._last_dropoff_count = int(self.produce_count)

        # transfers pending -> money (net worth unchanged)
        self.money += float(self.produce_value_pending)
        self.produce_value_pending = 0.0
        self.produce_count = 0
        return "success"

    def _do_refill_weed(self) -> str:
        if not self._is_adjacent_to(Terrain.CHEM_TANK):
            return "invalid_noop"
        if self.refilled_weed_today:
            return "invalid_noop"
        if (not self.cfg.ALLOW_PARTIAL_REFILL_WEED) and (self.weed_tank > 0):
            return "invalid_noop"
        self.weed_tank = int(self.cfg.WEED_TANK_MAX)
        self.refilled_weed_today = True
        return "success"

    def _do_refill_fert(self) -> str:
        if not self._is_adjacent_to(Terrain.CHEM_TANK):
            return "invalid_noop"
        if self.refilled_fert_today:
            return "invalid_noop"
        if (not self.cfg.ALLOW_PARTIAL_REFILL_FERT) and (self.fert_tank > 0):
            return "invalid_noop"
        self.fert_tank = int(self.cfg.FERT_TANK_MAX)
        self.refilled_fert_today = True
        return "success"

    def _do_refill_water(self) -> str:
        if not self._is_adjacent_to(Terrain.WATER):
            return "invalid_noop"
        if self.refilled_water_today:
            return "invalid_noop"
        if (not self.cfg.ALLOW_PARTIAL_REFILL_WATER) and (self.water_tank > 0):
            return "invalid_noop"
        self.water_tank = int(self.cfg.WATER_TANK_MAX)
        self.refilled_water_today = True
        return "success"

    # -----------------------
    # Nightly dynamics
    # -----------------------
    def _update_holes(self):
        cfg = self.cfg
        for y in range(self.h):
            for x in range(self.w):
                t = Terrain(self.terrain[y, x])
                if t == Terrain.WALKWAY:
                    if self.rng.random() < float(cfg.HOLE_FORM_PROB_FROM_WALKWAY):
                        self.terrain[y, x] = int(Terrain.HOLE)
                elif t == Terrain.HOLE:
                    if self.rng.random() < float(cfg.HOLE_FILL_PROB):
                        self.terrain[y, x] = int(Terrain.WALKWAY)

    def _apply_water_pool_boosts(self):
        cfg = self.cfg
        # Boost saturation around water tiles
        for y in range(self.h):
            for x in range(self.w):
                if Terrain(self.terrain[y, x]) != Terrain.WATER:
                    continue
                for dx, dy, boost in [
                    (0, -1, cfg.WATER_BOOST_1_STEP),
                    (1, 0, cfg.WATER_BOOST_1_STEP),
                    (0, 1, cfg.WATER_BOOST_1_STEP),
                    (-1, 0, cfg.WATER_BOOST_1_STEP),
                ]:
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny) and Terrain(self.terrain[ny, nx]) == Terrain.TILLED:
                        self.saturation[ny, nx] = float(np.clip(self.saturation[ny, nx] + float(boost), 0.0, 1.0))

                # 2 steps
                for dx, dy, boost in [
                    (0, -2, cfg.WATER_BOOST_2_STEP),
                    (2, 0, cfg.WATER_BOOST_2_STEP),
                    (0, 2, cfg.WATER_BOOST_2_STEP),
                    (-2, 0, cfg.WATER_BOOST_2_STEP),
                ]:
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny) and Terrain(self.terrain[ny, nx]) == Terrain.TILLED:
                        self.saturation[ny, nx] = float(np.clip(self.saturation[ny, nx] + float(boost), 0.0, 1.0))

    def _weed_growth(self):
        cfg = self.cfg
        for y in range(self.h):
            for x in range(self.w):
                if Terrain(self.terrain[y, x]) != Terrain.TILLED:
                    continue

                sat = float(self.saturation[y, x])
                noise = float(self.rng.normal(loc=sat, scale=float(cfg.WEED_SAT_GAUSS_SIGMA)))
                p_extra = float(cfg.WEED_SAT_GAUSS_SCALE) * max(0.0, noise)

                if self.plant_stage[y, x] <= 0:
                    p = float(cfg.WEED_SPAWN_EMPTY_BASE) + p_extra
                    if self.rng.random() < p:
                        self.weed[y, x] = 1
                else:
                    p = float(cfg.WEED_SPAWN_WITH_PLANT_BASE) + p_extra
                    if self.rng.random() < p:
                        self.weed[y, x] = 1

    def _plant_growth_and_death(self):
        cfg = self.cfg
        sat_enabled = self._flag("ENABLE_SATURATION", True)
        if (not sat_enabled) and bool(getattr(cfg, "SATURATION_ALWAYS_OK_WHEN_DISABLED", True)):
            sat_enabled = False

        for y in range(self.h):
            for x in range(self.w):
                if Terrain(self.terrain[y, x]) != Terrain.TILLED:
                    continue

                stage = int(self.plant_stage[y, x])
                if stage <= 0:
                    continue

                # saturation gate
                if sat_enabled and float(self.saturation[y, x]) < float(cfg.MIN_SATURATION_FOR_GROWTH):
                    self.dry_nights[y, x] += 1
                    if self.dry_nights[y, x] >= int(cfg.DRY_NIGHTS_TO_DIE):
                        self.plant_stage[y, x] = 0
                        self.weed[y, x] = 0
                        self.fertilized[y, x] = 0
                        self.dry_nights[y, x] = 0
                        self.mature_days[y, x] = 0
                    continue
                else:
                    self.dry_nights[y, x] = 0

                if stage >= 5:
                    self.mature_days[y, x] += 1
                    if self.mature_days[y, x] >= int(cfg.MATURE_DAYS_BEFORE_DEATH):
                        self.plant_stage[y, x] = 0
                        self.weed[y, x] = 0
                        self.fertilized[y, x] = 0
                        self.dry_nights[y, x] = 0
                        self.mature_days[y, x] = 0
                    continue

                base_p = float(cfg.GROWTH_PROB_BY_STAGE[stage - 1]) if cfg.GROWTH_PROB_BY_STAGE is not None else 0.30

                if self._flag("ENABLE_FERTILIZER", True) and self.fertilized[y, x] == 1:
                    base_p += float(cfg.FERT_GROWTH_BONUS)

                if self._flag("ENABLE_WEEDS", True) and self.weed[y, x] == 1:
                    base_p -= float(cfg.WEED_GROWTH_PENALTY)

                base_p = float(np.clip(base_p, 0.0, 1.0))

                if self.rng.random() < base_p:
                    self.plant_stage[y, x] += 1

    def _end_day(self) -> float:
        cfg = self.cfg
        before_money = float(self.money)

        # Charger rule
        on_charger = (self.robot_x, self.robot_y) == self.pos_charger
        self.missed_charger_today = (not on_charger)
        if getattr(cfg, "DISABLE_NEXT_DAY_IF_NOT_CHARGED", True) and (not on_charger):
            self.disabled_days_remaining = 1
            self.missed_charger_days += 1
            # Optional: immediate penalty to reduce delayed-credit issues for the charger constraint.
            if self._flag("ENABLE_CHARGER_MISS_PENALTY", True):
                self.money -= float(getattr(cfg, "CHARGER_MISS_PENALTY", 5.0))

        # Holes
        if self._flag("ENABLE_HOLES", True):
            self._update_holes()

        # Saturation + diffusion
        if self._flag("ENABLE_SATURATION", True):
            self.saturation = np.clip(self.saturation - float(cfg.SATURATION_DECAY_PER_NIGHT), 0.0, 1.0).astype(np.float32)
            self._apply_water_pool_boosts()

        # Weeds
        if self._flag("ENABLE_WEEDS", True):
            self._weed_growth()
        else:
            self.weed.fill(0)

        # Plants
        self._plant_growth_and_death()

        # Upkeep
        if self._flag("ENABLE_UPKEEP_COSTS", True):
            if self._flag("ENABLE_SPRAYING", True) and self.weed_tank <= 0:
                self.money -= float(cfg.UPKEEP_COST_IF_WEED_EMPTY)
                self.weed_tank = int(cfg.WEED_TANK_MAX)

            if self._flag("ENABLE_FERTILIZER", True) and self.fert_tank <= 0:
                self.money -= float(cfg.UPKEEP_COST_IF_FERT_EMPTY)
                self.fert_tank = int(cfg.FERT_TANK_MAX)

        # Reset for next day
        self.day += 1
        self.actions_left_today = int(cfg.BATTERY_ACTIONS_PER_DAY)

        self.refilled_weed_today = False
        self.refilled_fert_today = False
        self.refilled_water_today = False

        # Water tank refills every morning (per your original spec)
        self.water_tank = int(cfg.WATER_TANK_MAX)

        if self.disabled_days_remaining > 0:
            self.disabled_today = True
            self.disabled_days_remaining -= 1
            self.disabled_days_total += 1
        else:
            self.disabled_today = False

        # Start each day at charger (sim simplification)
        self.robot_x, self.robot_y = self.pos_charger

        # Distances may change due to holes
        self._dist_valid = False

        return float(self.money - before_money)
