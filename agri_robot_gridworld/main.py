import os
import argparse
import pygame
import numpy as np

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:
    imageio = None

from config import make_config
from sprites import SpriteBank
from agri_env import AgriRobotEnv, Terrain, Action

try:
    import torch
    from dqn_double import DoubleDQNAgent, DQNConfig
except Exception:
    torch = None


def visible_cells(cx: int, cy: int, radius: int, w: int, h: int):
    cells = set()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                cells.add((nx, ny))
    return cells


class Replay:
    def __init__(self, path: str):
        self.path = path
        self.data = np.load(path, allow_pickle=False)
        self.n = len(self.data["action"])
        self.i = 0
        self.playing = False

    def toggle(self):
        self.playing = not self.playing

    def step_forward(self):
        self.i = min(self.i + 1, self.n - 1)

    def step_back(self):
        self.i = max(self.i - 1, 0)

    def frame(self):
        i = self.i
        return {
            "action": int(self.data["action"][i]),
            "reward": float(self.data["reward"][i]),
            "robot_x": int(self.data["robot_x"][i]),
            "robot_y": int(self.data["robot_y"][i]),
            "day": int(self.data["day"][i]),
            "actions_left": int(self.data["actions_left"][i]),
            "money": float(self.data["money"][i]),
            "produce_count": int(self.data["produce_count"][i]),
            "pending": float(self.data["pending"][i]),
            "weed_tank": int(self.data["weed_tank"][i]),
            "fert_tank": int(self.data["fert_tank"][i]),
            "water_tank": int(self.data["water_tank"][i]),
            "disabled": int(self.data["disabled"][i]),
            "terrain": self.data["terrain"][i],
            "saturation": self.data["saturation"][i],
            "plant_stage": self.data["plant_stage"][i],
            "weed": self.data["weed"][i],
            "fertilized": self.data["fertilized"][i],
        }


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--replay",
        action="append",
        default=[],
        help=(
            "Replay NPZ path(s) to load. Can be provided multiple times or as a comma-separated list. "
            "Examples: --replay replays/a.npz --replay replays/b.npz OR --replay replays/a.npz,replays/b.npz"
        ),
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replays",
        help="Directory to search for replay NPZ files when cycling",
    )

    # Replay -> GIF export
    parser.add_argument(
        "--gif-out",
        type=str,
        default="",
        help=(
            "Write the currently loaded replay as a GIF. "
            "If a directory is provided, the GIF filename is derived from the replay filename."
        ),
    )
    parser.add_argument("--gif-fps", type=int, default=15, help="GIF playback fps")
    parser.add_argument(
        "--gif-every",
        type=int,
        default=1,
        help="Record every Nth replay step (1 records every step)",
    )
    parser.add_argument(
        "--gif-max-frames",
        type=int,
        default=0,
        help="Stop recording after this many frames (0 = no limit)",
    )
    parser.add_argument(
        "--gif-include-ui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the bottom UI panel in the exported GIF",
    )
    parser.add_argument(
        "--gif-export",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Non-interactive mode: auto-play the replay, export GIF, and exit",
    )
    parser.add_argument(
        "--gif-stop-at",
        type=str,
        choices=["min", "max"],
        default="min",
        help=(
            "When exporting multiple replays, stop when the shortest ends (min) or when all end (max). "
            "min avoids long tails where finished replays freeze on their last frame."
        ),
    )
    args, _ = parser.parse_known_args()

    cfg = make_config()
    env = AgriRobotEnv(cfg, seed=0)

    pygame.init()

    screen_w = cfg.GRID_W * cfg.TILE_SIZE
    screen_h = cfg.GRID_H * cfg.TILE_SIZE + 110
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Agri Robot Gridworld (Viewer + Replay)")

    sprites = SpriteBank(cfg)

    font = pygame.font.SysFont("consolas", 16)
    clock = pygame.time.Clock()

    obs, _ = env.reset()

    show_fog = True
    running = True

    agent = None
    agent_play = False
    if torch is not None:
        try:
            obs_dim = env.observation_space.shape[0]
            num_actions = env.action_space.n
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = DoubleDQNAgent(obs_dim, num_actions, DQNConfig(), device=device)
            if os.path.exists("double_dqn_agri.pt"):
                agent.load("double_dqn_agri.pt")
        except Exception:
            agent = None

    replay: Replay | None = None
    replays: list[Replay] = []
    multi_replay = False
    replay_mode = False
    replay_dir = str(args.replay_dir)
    replay_files = []
    replay_index = -1

    def refresh_replay_files():
        nonlocal replay_files
        try:
            if os.path.isdir(replay_dir):
                files = [os.path.join(replay_dir, f) for f in os.listdir(replay_dir) if f.endswith(".npz")]
                replay_files = sorted(files)
            else:
                replay_files = []
        except Exception:
            replay_files = []

    def load_replay(path: str):
        nonlocal replay, replay_mode, agent_play
        if not path:
            return

        cand = str(path)
        # If the user passed only a filename, try resolving under --replay-dir
        if (not os.path.exists(cand)) and (not os.path.isabs(cand)):
            cand2 = os.path.join(replay_dir, cand)
            if os.path.exists(cand2):
                cand = cand2

        if os.path.exists(cand):
            replay = Replay(cand)
            replay_mode = True
            agent_play = False
        else:
            print(f"[replay] file not found: {path} (also tried {os.path.join(replay_dir, str(path))})")

    def load_replays(paths: list[str]):
        nonlocal replays, replay, multi_replay, replay_mode, agent_play, screen, screen_w, screen_h

        out: list[Replay] = []
        for p in paths:
            p = str(p).strip()
            if not p:
                continue
            # Allow comma-separated lists within a single --replay
            for part in [s.strip() for s in p.split(",") if s.strip()]:
                cand = part
                if (not os.path.exists(cand)) and (not os.path.isabs(cand)):
                    cand2 = os.path.join(replay_dir, cand)
                    if os.path.exists(cand2):
                        cand = cand2
                if os.path.exists(cand):
                    out.append(Replay(cand))
                else:
                    print(f"[replay] file not found: {part} (also tried {os.path.join(replay_dir, part)})")

        replays = out
        multi_replay = len(replays) > 1
        if len(replays) >= 1:
            replay = replays[0]
            replay_mode = True
            agent_play = False

            # Resize window for side-by-side rendering
            if multi_replay:
                screen_w = (cfg.GRID_W * cfg.TILE_SIZE) * len(replays)
                screen_h = cfg.GRID_H * cfg.TILE_SIZE + 110
                screen = pygame.display.set_mode((screen_w, screen_h))
                pygame.display.set_caption("Agri Robot Gridworld (Viewer + Replay) [MULTI]")

    refresh_replay_files()
    if args.replay:
        load_replays([str(x) for x in args.replay])
        if replay is not None:
            try:
                replay_index = replay_files.index(replay.path)
            except Exception:
                replay_index = -1

    # GIF writer setup (only active when a replay is loaded)
    gif_writer = None
    gif_path = ""
    gif_frames_written = 0
    last_recorded_replay_i = -1

    def _resolve_gif_path(replay_path: str, out: str) -> str:
        out = str(out)
        if not out:
            return ""
        if out.lower().endswith(".gif"):
            return out
        if os.path.isdir(out) or out.endswith(os.sep):
            base = os.path.splitext(os.path.basename(replay_path))[0] + ".gif"
            return os.path.join(out, base)
        # Treat as directory-like if it doesn't end with .gif
        os.makedirs(out, exist_ok=True)
        base = os.path.splitext(os.path.basename(replay_path))[0] + ".gif"
        return os.path.join(out, base)

    def _maybe_start_gif():
        nonlocal gif_writer, gif_path, gif_frames_written, last_recorded_replay_i
        if gif_writer is not None:
            return
        if not str(args.gif_out).strip():
            return
        if replay is None and (not replays):
            return
        if imageio is None:
            print("[gif] imageio not installed; run: pip install imageio Pillow")
            return
        # Derive gif name from first replay, or from a generic multi name
        base_replay_path = replay.path if replay is not None else replays[0].path
        if len(replays) > 1:
            base = os.path.splitext(os.path.basename(base_replay_path))[0] + f"_x{len(replays)}.gif"
            out = str(args.gif_out).strip()
            if out.lower().endswith(".gif"):
                gif_path = out
            else:
                os.makedirs(out, exist_ok=True)
                gif_path = os.path.join(out, base)
        else:
            gif_path = _resolve_gif_path(base_replay_path, str(args.gif_out).strip())
        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        gif_writer = imageio.get_writer(gif_path, mode="I", fps=int(args.gif_fps))
        gif_frames_written = 0
        last_recorded_replay_i = -1
        print(f"[gif] recording -> {gif_path} @ {int(args.gif_fps)} fps")

    # If user asked for non-interactive GIF export but didn't load a replay, fail fast.
    if bool(args.gif_export) and str(args.gif_out).strip() and replay is None and (not replays):
        print("[gif] --gif-export was set but no replay is loaded. Use --replay PATH (or a filename in --replay-dir).")
        return

    def draw_one_view(
        *,
        ox: int,
        oy: int,
        rx: int,
        ry: int,
        terrain,
        saturation,
        plant_stage,
        weed,
        fertilized,
    ):
        vis = visible_cells(rx, ry, cfg.VISION_RADIUS, cfg.GRID_W, cfg.GRID_H)
        for y in range(cfg.GRID_H):
            for x in range(cfg.GRID_W):
                terr_name = Terrain(terrain[y, x]).name
                tile_surf = sprites.tiles.get(terr_name, sprites.tiles["UNKNOWN"])

                px = ox + x * cfg.TILE_SIZE
                py = oy + y * cfg.TILE_SIZE
                screen.blit(tile_surf, (px, py))

                if Terrain(terrain[y, x]) == Terrain.TILLED:
                    alpha = int(120 * float(np.clip(saturation[y, x], 0.0, 1.0)))
                    overlay = pygame.Surface((cfg.TILE_SIZE, cfg.TILE_SIZE), pygame.SRCALPHA)
                    overlay.fill((0, 0, 255, alpha))
                    screen.blit(overlay, (px, py))

                    if weed[y, x] == 1:
                        pygame.draw.circle(screen, cfg.FALLBACK_COLORS["WEED"], (px + 10, py + 10), 6)

                    s = int(plant_stage[y, x])
                    if s > 0:
                        stage_sprite = sprites.plant_stages.get(s)
                        if stage_sprite is not None:
                            sw, sh = stage_sprite.get_width(), stage_sprite.get_height()
                            ox2 = px + (cfg.TILE_SIZE - sw) // 2
                            oy2 = py + cfg.TILE_SIZE - sh - 2
                            screen.blit(stage_sprite, (ox2, oy2))
                        else:
                            txt = font.render(str(s), True, cfg.FALLBACK_COLORS["PLANT"])
                            screen.blit(txt, (px + cfg.TILE_SIZE - 14, py + 6))

                    if fertilized[y, x] == 1:
                        pygame.draw.rect(
                            screen,
                            cfg.FALLBACK_COLORS["FERT"],
                            pygame.Rect(px + 6, py + cfg.TILE_SIZE - 14, 8, 8),
                        )

                if show_fog and ((x, y) not in vis):
                    fog = pygame.Surface((cfg.TILE_SIZE, cfg.TILE_SIZE), pygame.SRCALPHA)
                    fog.fill((0, 0, 0, 200))
                    screen.blit(fog, (px, py))

                pygame.draw.rect(
                    screen,
                    cfg.FALLBACK_COLORS["GRID"],
                    pygame.Rect(px, py, cfg.TILE_SIZE, cfg.TILE_SIZE),
                    1,
                )

        screen.blit(sprites.robot, (ox + rx * cfg.TILE_SIZE, oy + ry * cfg.TILE_SIZE))

    def _gif_capture_rect() -> pygame.Rect:
        if bool(args.gif_include_ui):
            return pygame.Rect(0, 0, screen_w, screen_h)
        return pygame.Rect(0, 0, screen_w, cfg.GRID_H * cfg.TILE_SIZE)

    def _gif_append_frame():
        nonlocal gif_frames_written
        if gif_writer is None:
            return
        rect = _gif_capture_rect()
        surf = screen.subsurface(rect)
        arr = pygame.surfarray.array3d(surf)  # [W, H, 3]
        frame = np.transpose(arr, (1, 0, 2))  # -> [H, W, 3]
        gif_writer.append_data(frame)
        gif_frames_written += 1

    move_keys = {
        pygame.K_UP: Action.MOVE_N, pygame.K_w: Action.MOVE_N,
        pygame.K_RIGHT: Action.MOVE_E, pygame.K_d: Action.MOVE_E,
        pygame.K_DOWN: Action.MOVE_S, pygame.K_s: Action.MOVE_S,
        pygame.K_LEFT: Action.MOVE_W, pygame.K_a: Action.MOVE_W,
    }
    spray_keys = {pygame.K_i: Action.SPRAY_N, pygame.K_l: Action.SPRAY_E, pygame.K_k: Action.SPRAY_S, pygame.K_j: Action.SPRAY_W}
    water_keys = {pygame.K_t: Action.WATER_N, pygame.K_h: Action.WATER_E, pygame.K_g: Action.WATER_S, pygame.K_f: Action.WATER_W}
    fert_keys = {pygame.K_r: Action.FERT_N, pygame.K_e: Action.FERT_E, pygame.K_d: Action.FERT_S, pygame.K_q: Action.FERT_W}
    plant_keys = {pygame.K_1: Action.PLANT_N, pygame.K_2: Action.PLANT_E, pygame.K_3: Action.PLANT_S, pygame.K_4: Action.PLANT_W}
    harvest_keys = {pygame.K_5: Action.HARVEST_N, pygame.K_6: Action.HARVEST_E, pygame.K_7: Action.HARVEST_S, pygame.K_8: Action.HARVEST_W}

    while running:
        # In GIF export mode, don't artificially throttle the loop.
        if not bool(args.gif_export):
            clock.tick(30)
        chosen_action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_f:
                    show_fog = not show_fog

                if event.key == pygame.K_p:
                    agent_play = not agent_play
                    replay_mode = False

                if event.key == pygame.K_l:
                    path = "replays/last_episode.npz"
                    if os.path.exists(path):
                        replay = Replay(path)
                        replay_mode = True
                        agent_play = False

                if event.key == pygame.K_LEFTBRACKET:
                    refresh_replay_files()
                    if replay_files:
                        if replay_index < 0:
                            replay_index = 0
                        replay_index = (replay_index - 1) % len(replay_files)
                        load_replay(replay_files[replay_index])

                if event.key == pygame.K_RIGHTBRACKET:
                    refresh_replay_files()
                    if replay_files:
                        if replay_index < 0:
                            replay_index = 0
                        replay_index = (replay_index + 1) % len(replay_files)
                        load_replay(replay_files[replay_index])

                if replay_mode and replay is not None:
                    if event.key == pygame.K_SPACE:
                        if multi_replay and replays:
                            playing = not replays[0].playing
                            for r in replays:
                                r.playing = playing
                        else:
                            replay.toggle()
                    if event.key == pygame.K_PERIOD:
                        if multi_replay and replays:
                            for r in replays:
                                r.step_forward()
                        else:
                            replay.step_forward()
                    if event.key == pygame.K_COMMA:
                        if multi_replay and replays:
                            for r in replays:
                                r.step_back()
                        else:
                            replay.step_back()

                if not replay_mode:
                    if event.key == pygame.K_z:
                        chosen_action = Action.REFILL_WEED
                    if event.key == pygame.K_x:
                        chosen_action = Action.REFILL_FERT
                    if event.key == pygame.K_c:
                        chosen_action = Action.REFILL_WATER
                    if event.key == pygame.K_v:
                        chosen_action = Action.DROPOFF_PRODUCE

                    if event.key == pygame.K_n:
                        start_day = env.day
                        while env.day == start_day and env.day <= cfg.NUM_DAYS:
                            obs, r, term, trunc, info = env.step(int(Action.MOVE_N))
                            if term or trunc:
                                break

                    if event.key in move_keys:
                        chosen_action = move_keys[event.key]
                    if event.key in spray_keys:
                        chosen_action = spray_keys[event.key]
                    if event.key in water_keys:
                        chosen_action = water_keys[event.key]
                    if event.key in fert_keys:
                        chosen_action = fert_keys[event.key]
                    if event.key in plant_keys:
                        chosen_action = plant_keys[event.key]
                    if event.key in harvest_keys:
                        chosen_action = harvest_keys[event.key]

        if replay_mode and (replay is not None or replays):
            if bool(args.gif_export):
                # Auto-play when exporting
                if multi_replay and replays:
                    for r in replays:
                        r.playing = True
                elif replay is not None:
                    replay.playing = True

            if multi_replay and replays:
                for r in replays:
                    if r.playing:
                        r.step_forward()
                # Use first replay for UI summary fields
                frame0 = replays[0].frame()
                day = frame0["day"]
                actions_left = frame0["actions_left"]
                money = frame0["money"]
                produce_count = frame0["produce_count"]
                pending = frame0["pending"]
                weed_tank = frame0["weed_tank"]
                fert_tank = frame0["fert_tank"]
                water_tank = frame0["water_tank"]
                disabled = frame0["disabled"]
            else:
                if replay is not None and replay.playing:
                    replay.step_forward()

                frame = replay.frame()  # type: ignore[union-attr]
                rx, ry = frame["robot_x"], frame["robot_y"]
                terrain = frame["terrain"]
                saturation = frame["saturation"]
                plant_stage = frame["plant_stage"]
                weed = frame["weed"]
                fertilized = frame["fertilized"]

                day = frame["day"]
                actions_left = frame["actions_left"]
                money = frame["money"]
                produce_count = frame["produce_count"]
                pending = frame["pending"]
                weed_tank = frame["weed_tank"]
                fert_tank = frame["fert_tank"]
                water_tank = frame["water_tank"]
                disabled = frame["disabled"]

        else:
            if agent_play and agent is not None:
                mask = env.valid_action_mask()
                chosen_action = Action(agent.act(obs, mask=mask, greedy=True))

            if chosen_action is not None:
                obs, r, term, trunc, info = env.step(int(chosen_action))
                if term or trunc:
                    obs, _ = env.reset()

            rx, ry = env.robot_x, env.robot_y
            terrain = env.terrain
            saturation = env.saturation
            plant_stage = env.plant_stage
            weed = env.weed
            fertilized = env.fertilized

            day = env.day
            actions_left = env.actions_left_today
            money = env.money
            produce_count = env.produce_count
            pending = env.produce_value_pending
            weed_tank = env.weed_tank
            fert_tank = env.fert_tank
            water_tank = env.water_tank
            disabled = int(env.disabled_today)

        screen.fill((0, 0, 0))

        if replay_mode and multi_replay and replays:
            for idx, r in enumerate(replays):
                fr = r.frame()
                ox = idx * (cfg.GRID_W * cfg.TILE_SIZE)
                draw_one_view(
                    ox=ox,
                    oy=0,
                    rx=int(fr["robot_x"]),
                    ry=int(fr["robot_y"]),
                    terrain=fr["terrain"],
                    saturation=fr["saturation"],
                    plant_stage=fr["plant_stage"],
                    weed=fr["weed"],
                    fertilized=fr["fertilized"],
                )
        else:
            draw_one_view(
                ox=0,
                oy=0,
                rx=int(rx),
                ry=int(ry),
                terrain=terrain,
                saturation=saturation,
                plant_stage=plant_stage,
                weed=weed,
                fertilized=fertilized,
            )

        ui_y = cfg.GRID_H * cfg.TILE_SIZE
        pygame.draw.rect(screen, (18, 18, 18), pygame.Rect(0, ui_y, screen_w, 110))

        mode = "REPLAY" if replay_mode else ("AGENT" if agent_play else "MANUAL")
        if replay_mode and multi_replay:
            mode = f"REPLAY x{len(replays)}"
        lines = [
            f"MODE: {mode} | Day {day}/{cfg.NUM_DAYS} | actions left: {actions_left} | disabled_today: {bool(disabled)}",
            f"Money: {money:.2f} | produce: {produce_count}/{cfg.PRODUCE_STORAGE_MAX} (pending ${pending:.1f})",
            f"WeedTank: {weed_tank}/{cfg.WEED_TANK_MAX} | FertTank: {fert_tank}/{cfg.FERT_TANK_MAX} | WaterTank: {water_tank}/{cfg.WATER_TANK_MAX}",
            "Move: WASD/Arrows | Spray: IJKL | Water: TFGH | Fert: R/E/D/Q | Plant: 1/2/3/4 | Harvest: 5/6/7/8",
            "Refill weed/fert/water: Z/X/C | Dropoff: V | Toggle fog: F | End day: N | Agent play: P",
            "Replay: L loads replays/last_episode.npz | [ / ] cycle replays/*.npz | Space play/pause | ,/. step",
        ]
        for i, line in enumerate(lines):
            label = font.render(line, True, cfg.FALLBACK_COLORS["TEXT"])
            screen.blit(label, (10, ui_y + 8 + i * 18))

        # GIF recording: start lazily once a replay is loaded.
        if replay_mode and (replay is not None or replays) and str(args.gif_out).strip():
            _maybe_start_gif()
            # Only record when the leading replay index advances (avoids duplicates when paused)
            lead = replay if replay is not None else replays[0]
            if lead.i != last_recorded_replay_i:
                every = int(max(1, int(args.gif_every)))
                if (lead.i % every) == 0:
                    _gif_append_frame()
                    last_recorded_replay_i = int(lead.i)

            if int(args.gif_max_frames) > 0 and gif_frames_written >= int(args.gif_max_frames):
                print(f"[gif] reached max frames ({gif_frames_written}); stopping")
                running = False

            if bool(args.gif_export):
                if multi_replay and replays:
                    ends = [r.i >= (r.n - 1) for r in replays]
                    stop_at = str(args.gif_stop_at).lower().strip()
                    if stop_at == "max":
                        done_now = all(ends)
                    else:
                        done_now = any(ends)
                    if done_now:
                        lens = ",".join([str(r.n) for r in replays])
                        print(f"[gif] reached end-of-replay condition (stop_at={stop_at}; lens={lens}); stopping")
                        running = False
                else:
                    if replay is not None and replay.i >= (replay.n - 1):
                        print(f"[gif] reached end of replay ({replay.n} steps); stopping")
                        running = False

        pygame.display.flip()

        if bool(args.gif_export):
            # Keep window responsive during non-interactive export
            pygame.event.pump()

    try:
        if gif_writer is not None:
            gif_writer.close()
            print(f"[gif] saved -> {gif_path} | frames={gif_frames_written}")
    except Exception as e:
        print(f"[gif] failed to finalize gif: {e}")

    pygame.quit()


if __name__ == "__main__":
    main()
