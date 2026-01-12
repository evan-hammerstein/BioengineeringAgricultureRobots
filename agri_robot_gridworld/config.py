from dataclasses import dataclass
from typing import Dict, Tuple, List
from pathlib import Path


@dataclass
class Config:
    # -------------------------
    # Rendering / grid
    # -------------------------
    GRID_W: int = 12
    GRID_H: int = 10
    TILE_SIZE: int = 48

    # Viewer-only: how much fog-of-war to show in main.py.
    # NOTE: This does NOT change the RL observation size (the env uses a fixed 3x3 local view + memory map).
    VISION_RADIUS: int = 6

    # -------------------------
    # Feature toggles (ablations)
    # -------------------------
    ENABLE_WEEDS: bool = True

    # Keep these off while debugging (you can turn them on later).
    ENABLE_SATURATION: bool = False
    ENABLE_FERTILIZER: bool = False
    ENABLE_WATERING: bool = False

    # Keep spraying on if weeds are on.
    ENABLE_SPRAYING: bool = True

    ENABLE_UPKEEP_COSTS: bool = False  # end-of-day costs for empty tanks (turn on later)
    ENABLE_HOLES: bool = True    # holes forming/filling on walkway (turn on later)

    ENABLE_MOVE_FAIL_FROM_LOAD: bool = True
    ENABLE_TILLED_STEP_KILL: bool = True

    # Optional: prevent degenerate "plant then immediately harvest" policies.
    ENABLE_MATURITY_GATE: bool = True
    MIN_HARVEST_STAGE: int = 3  # harvest allowed only if stage >= this (when maturity gate enabled)

    # Optional: immediate penalty for missing charger (reduces delayed-credit issues).
    ENABLE_CHARGER_MISS_PENALTY: bool = True
    CHARGER_MISS_PENALTY: float = 5.0

    # If saturation disabled, treat all tilled as always "wet enough"
    SATURATION_ALWAYS_OK_WHEN_DISABLED: bool = True

    # -------------------------
    # Simulation horizon
    # -------------------------
    NUM_DAYS: int = 30

    # -------------------------
    # Robot "battery": actions per day
    # -------------------------
    BATTERY_ACTIONS_PER_DAY: int = 60

    # If robot does NOT end day on charger => next day disabled (0 actions), day after resumes.
    DISABLE_NEXT_DAY_IF_NOT_CHARGED: bool = True

    # -------------------------
    # Robot storage + tanks
    # -------------------------
    WEED_TANK_MAX: int = 12
    FERT_TANK_MAX: int = 10
    WATER_TANK_MAX: int = 14
    PRODUCE_STORAGE_MAX: int = 20

    # Move failure increases with produce carried
    MOVE_FAIL_PER_PRODUCE: float = 0.01  # 1% per produce item

    # If False, tank refill only allowed when tank is empty
    ALLOW_PARTIAL_REFILL_WEED: bool = False
    ALLOW_PARTIAL_REFILL_FERT: bool = False
    ALLOW_PARTIAL_REFILL_WATER: bool = False

    # -------------------------
    # Terrain generation probabilities
    # (exactly one CHARGER, one DROPOFF, one CHEM_TANK will be forced)
    # -------------------------
    P_TILLED: float = 0.55
    P_WALKWAY: float = 0.28
    P_WATER: float = 0.05
    P_HOLE: float = 0.00
    P_CHEM_TANK: float = 0.02
    P_DROPOFF: float = 0.02
    P_CHARGER: float = 0.01

    # -------------------------
    # Hole dynamics (per night)
    # -------------------------
    HOLE_FORM_PROB_FROM_WALKWAY: float = 0.03
    HOLE_FILL_PROB: float = 0.10

    # -------------------------
    # Saturation dynamics
    # -------------------------
    SATURATION_MAX: float = 1.0
    SATURATION_MIN: float = 0.0
    SATURATION_DECAY_PER_NIGHT: float = 0.03

    # Water pool passive boosts
    WATER_BOOST_1_STEP: float = 0.20
    WATER_BOOST_2_STEP: float = 0.10

    # Water action boost (watering a tilled tile)
    WATER_ACTION_BOOST: float = 0.30

    # -------------------------
    # Moving on tilled land can kill plant/weed
    # Spec says "will kill any plant/weed". Keep = 1.0
    # -------------------------
    TILLED_STEP_KILL_PROB: float = 1.0

    # -------------------------
    # Plant / weed dynamics (nightly)
    # -------------------------
    # Plant stages: 1..5 (0 empty). Growth probability can depend on current stage.
    GROWTH_PROB_BY_STAGE: List[float] = None  # set in make_config()


    FERT_GROWTH_BONUS: float = 0.20
    WEED_GROWTH_PENALTY: float = 0.15

    MIN_SATURATION_FOR_GROWTH: float = 0.25

    DRY_NIGHTS_TO_DIE: int = 3
    MATURE_DAYS_BEFORE_DEATH: int = 2

    # Weed spawn base probs
    WEED_SPAWN_EMPTY_BASE: float = 0.04
    WEED_SPAWN_WITH_PLANT_BASE: float = 0.06

    # Additional weed spawn term: base + scale * Normal(mean=saturation, sigma=...)
    WEED_SAT_GAUSS_SIGMA: float = 0.10
    WEED_SAT_GAUSS_SCALE: float = 0.20

    # -------------------------
    # Actions (probabilistic effects)
    # -------------------------
    SPRAY_KILL_WEED_PROB: float = 0.85
    SPRAY_KILL_PLANT_PROB: float = 0.15

    # -------------------------
    # Economy
    # -------------------------
    BASE_PRODUCE_VALUE: float = 8.0

    # Seeds can put you "in debt" so this is a direct money cost at action time.
    SEED_COST: float = 0.001

    # -------------------------
    # Reward shaping (training helpers)
    # -------------------------
    # These do NOT change the underlying simulation state; they only add to the per-step reward.
    # Keep magnitudes small relative to harvest value.
    ENABLE_STEP_PENALTY: bool = True
    STEP_PENALTY: float = -0.001

    # At the end of the day, we want to be produces crops. You cant do that without planting!
    ENABLE_PLANT_BONUS: bool = True
    PLANT_BONUS: float = 1.0

    # Applies when the robot destroys (tramples) an existing plant/weed by moving onto a tilled tile.
    ENABLE_TRAMPLE_PENALTY: bool = True
    TRAMPLE_PENALTY: float = -0.20

    # Dropoff doesn't change net worth (pending -> money). This bonus helps credit assignment.
    ENABLE_DROPOFF_BONUS: bool = True
    DROPOFF_BONUS_PER_ITEM: float = 0.02

    # End-of-day upkeep costs IF tank is empty, conserve or refill!
    UPKEEP_COST_IF_WEED_EMPTY: float = 2.0
    UPKEEP_COST_IF_FERT_EMPTY: float = 2.0

    # -------------------------
    # Assets (Pygame viewer)
    # -------------------------
    TILE_SPRITES: Dict[str, str] = None
    # Optional overlay sprites for plant growth stages (1..5).
    # If a file is missing, the viewer falls back to drawing the stage number.
    PLANT_STAGE_SPRITES: Dict[int, str] = None
    # Rendered size relative to TILE_SIZE (e.g. 0.75 => 75% of tile).
    PLANT_STAGE_SPRITE_SCALE: float = 0.75
    ROBOT_SPRITE: str = "assets/robot/robot.png"
    FALLBACK_COLORS: Dict[str, Tuple[int, int, int]] = None


def make_config() -> Config:
    cfg = Config()

    ROOT = Path(__file__).resolve().parent
    ASSETS = ROOT / "assets"

    cfg.TILE_SPRITES = {
        "TILLED": str(ASSETS / "tiles" / "tilled.png"),
        "WALKWAY": str(ASSETS / "tiles" / "walkway.png"),
        "WATER": str(ASSETS / "tiles" / "water.png"),
        "HOLE": str(ASSETS / "tiles" / "hole.png"),
        "CHEM_TANK": str(ASSETS / "tiles" / "chem_tank.png"),
        "DROPOFF": str(ASSETS / "tiles" / "dropoff.png"),
        "CHARGER": str(ASSETS / "tiles" / "charger.png"),
        "UNKNOWN": str(ASSETS / "tiles" / "unknown.png"),
    }

    cfg.ROBOT_SPRITE = str(ASSETS / "robot" / "robot.png")

    # Plant stage overlays (tomato growth stages). Put your images here:
    #   assets/plants/tomato_stage1.png
    #   assets/plants/tomato_stage2.png
    #   assets/plants/tomato_stage3.png
    #   assets/plants/tomato_stage4.png
    #   assets/plants/tomato_stage5.png
    cfg.PLANT_STAGE_SPRITES = {
        1: str(ASSETS / "plants" / "tomato_stage1.png"),
        2: str(ASSETS / "plants" / "tomato_stage2.png"),
        3: str(ASSETS / "plants" / "tomato_stage3.png"),
        4: str(ASSETS / "plants" / "tomato_stage4.png"),
        5: str(ASSETS / "plants" / "tomato_stage5.png"),
    }

    cfg.FALLBACK_COLORS = {
        "TILLED": (139, 69, 19),
        "WALKWAY": (180, 180, 180),
        "WATER": (60, 120, 255),
        "HOLE": (40, 40, 40),
        "CHEM_TANK": (255, 200, 0),
        "DROPOFF": (170, 70, 200),
        "CHARGER": (0, 200, 100),
        "UNKNOWN": (30, 30, 30),
        "FOG": (0, 0, 0),
        "GRID": (20, 20, 20),
        "TEXT": (230, 230, 230),
        "WEED": (40, 200, 70),
        "PLANT": (255, 255, 255),
        "FERT": (255, 140, 0),
    }

    # Growth probs for stages 1..4; stage 5 handled separately (mature death timer)
    cfg.GROWTH_PROB_BY_STAGE = [0.85, 0.85, 0.85, 0.85, 0.0]
    return cfg
