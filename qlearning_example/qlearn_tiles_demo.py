#!/usr/bin/env python3
"""qlearn_tiles_demo.py

Small Q-learning demo on a 3x3 grid world.

Features:
- 32x32 tile sprites (ground, lava, chest, explorer) scaled up crisply
- Clean right-side panels:
        TOP    = experience tuple + equation IMAGE + numeric substitution
        BOTTOM = Q-table
- Optional offline training (epsilon-greedy) so you can actually optimize
- Saves the best episode (highest return) as a GIF

Controls (during the visualization window):
    SPACE: step one transition
    ENTER: auto-run
    R: reset episode (keeps learned Q)
    ESC / close window: quit
"""

import argparse
import sys
from pathlib import Path

import pygame
import numpy as np

from PIL import Image

# =========================
# ENV CONFIG
# =========================
GRID = 3
START = (0, 0)
LAVA  = (1, 1)   # terminal, -10
CHEST = (2, 2)   # terminal, +10

STEP_REWARD  = -1.0
LAVA_REWARD  = -10.0
CHEST_REWARD = +10.0

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
# indices: 0    1      2       3
DELTAS  = [(-1, 0), (1, 0), (0, -1), (0, 1)]

alpha   = 0.5
gamma   = 0.95

MAX_STEPS = 30

# =========================
# ASSETS
# =========================
ASSET_DIR = Path("assets")
SPR_GROUND = ASSET_DIR / "ground.png"
SPR_LAVA   = ASSET_DIR / "lava.png"
SPR_CHEST  = ASSET_DIR / "chest.png"
SPR_AGENT  = ASSET_DIR / "explorer.png"
SPR_QEQ    = ASSET_DIR / "q_update.png"   # equation image

# =========================
# HELPERS
# =========================
def s2i(s):
    r, c = s
    return r * GRID + c

def in_bounds(r, c):
    return 0 <= r < GRID and 0 <= c < GRID

def env_step(state, action_idx):
    r, c = state
    dr, dc = DELTAS[action_idx]
    nr, nc = r + dr, c + dc
    if not in_bounds(nr, nc):
        nr, nc = r, c  # bump => stay

    ns = (nr, nc)
    if ns == LAVA:
        return ns, LAVA_REWARD, True
    if ns == CHEST:
        return ns, CHEST_REWARD, True
    return ns, STEP_REWARD, False

def fmt(x: float) -> str:
    return f"{x:6.2f}"

def epsilon_greedy_action(Q: np.ndarray, state, epsilon: float, rng: np.random.Generator) -> int:
    """Pick action using epsilon-greedy with random tie-breaking."""
    if rng.random() < max(0.0, min(1.0, float(epsilon))):
        return int(rng.integers(0, 4))

    si = s2i(state)
    row = Q[si]
    best = float(np.max(row))
    candidates = np.flatnonzero(np.isclose(row, best))
    return int(rng.choice(candidates))

def train_q_learning(
    episodes: int,
    max_steps: int,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    seed: int | None,
):
    """Offline training loop (no rendering). Returns (Q, best_episode)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((GRID * GRID, 4), dtype=np.float32)

    best_return = -float("inf")
    best_states: list[tuple[int, int]] = []
    best_actions: list[int] = []
    best_rewards: list[float] = []

    episodes = max(0, int(episodes))
    eps_decay = max(1e-9, float(eps_decay))

    for ep in range(episodes):
        # Exponential decay: eps_end + (eps_start-eps_end) * exp(-ep/eps_decay)
        epsilon = float(eps_end + (eps_start - eps_end) * np.exp(-ep / eps_decay))

        state = START
        done = False
        total = 0.0

        states = [state]
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps):
            a = epsilon_greedy_action(Q, state, epsilon, rng)
            ns, r, done = env_step(state, a)

            si = s2i(state)
            nsi = s2i(ns)
            max_next = 0.0 if done else float(np.max(Q[nsi]))
            td_target = r + gamma * max_next
            Q[si, a] = Q[si, a] + alpha * (td_target - Q[si, a])

            total += float(r)
            actions.append(a)
            rewards.append(float(r))
            states.append(ns)
            state = ns

            if done:
                break

        if total > best_return:
            best_return = total
            best_states = states
            best_actions = actions
            best_rewards = rewards

    best_episode = {
        "return": best_return,
        "states": best_states,
        "actions": best_actions,
        "rewards": best_rewards,
    }
    return Q, best_episode

def save_gif(frames_rgb: list[np.ndarray], path: Path, fps: int):
    """Save list of RGB frames (H,W,3 uint8) as a GIF."""
    if not frames_rgb:
        return
    fps = max(1, int(fps))
    duration_ms = int(1000 / fps)
    images = [Image.fromarray(frame, mode="RGB") for frame in frames_rgb]
    images[0].save(
        str(path),
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )

# =========================
# PYGAME INIT
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="3x3 Q-learning demo (train + visualize + save best GIF)")
    p.add_argument("--train-episodes", type=int, default=8000)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=2500.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gif", type=str, default="best_run.gif")
    p.add_argument("--gif-fps", type=int, default=6)
    p.add_argument("--no-gif", action="store_true", help="Disable GIF recording")
    p.add_argument("--no-train", action="store_true", help="Skip training (starts with zero Q)")
    p.add_argument("--interactive", action="store_true", help="Keep window open after saving GIF")
    return p.parse_args()


pygame.init()
pygame.display.set_caption("Q-learning (3x3) — tiles + clean Q-table")

WIN_W, WIN_H = 1200, 740
screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)
clock = pygame.time.Clock()

def make_fonts(scale: float):
    scale = max(0.85, min(1.25, scale))
    base  = int(18 * scale)
    small = int(15 * scale)
    title = int(24 * scale)
    return (
        pygame.font.SysFont("Menlo", base),
        pygame.font.SysFont("Menlo", small),
        pygame.font.SysFont("Menlo", base, bold=True),
        pygame.font.SysFont("Menlo", title, bold=True),
    )

FONT, FONT_S, FONT_B, FONT_T = make_fonts(1.0)

# Colors
BG         = (245, 247, 250)
PANEL      = (255, 255, 255)
PANEL_EDGE = (210, 215, 225)
TEXT       = (25, 25, 25)
MUTED      = (110, 115, 130)
HILITE     = (235, 245, 255)
ORANGE     = (230, 160, 60)
GREEN      = (20, 120, 60)

PAD = 24
GAP = 16
FOOTER_H = 40

RIGHT_PANEL_W_MIN = 460
RIGHT_PANEL_W_MAX = 640

UPDATE_H_MIN = 200
UPDATE_H_MAX = 280

# =========================
# ASSET LOADING
# =========================
def load_sprite(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing asset: {path}\n"
            f"Ensure ./assets contains:\n"
            f"  ground.png, lava.png, chest.png, explorer.png, q_update.png"
        )
    return pygame.image.load(str(path)).convert_alpha()

ground_src = load_sprite(SPR_GROUND)
lava_src   = load_sprite(SPR_LAVA)
chest_src  = load_sprite(SPR_CHEST)
agent_src  = load_sprite(SPR_AGENT)
qeq_src    = load_sprite(SPR_QEQ)

def scale_nearest(img, size):
    return pygame.transform.scale(img, size)  # crisp pixel art

_cached_cell = None
ground_img = lava_img = chest_img = agent_img = None

def ensure_scaled_tiles(cell: int):
    global _cached_cell, ground_img, lava_img, chest_img, agent_img
    if _cached_cell == cell:
        return
    _cached_cell = cell
    sz = (cell, cell)
    ground_img = scale_nearest(ground_src, sz)
    lava_img   = scale_nearest(lava_src, sz)
    chest_img  = scale_nearest(chest_src, sz)
    agent_img  = scale_nearest(agent_src, sz)

_qeq_cached_w = None
qeq_img = None

def ensure_scaled_equation(panel_w: int):
    global _qeq_cached_w, qeq_img
    target_w = int(panel_w * 0.92)
    if _qeq_cached_w == target_w:
        return
    _qeq_cached_w = target_w

    w0, h0 = qeq_src.get_size()
    aspect = h0 / w0
    target_h = int(target_w * aspect)
    qeq_img = pygame.transform.smoothscale(qeq_src, (target_w, target_h))

# =========================
# DRAW HELPERS
# =========================
def draw_panel_rect(x, y, w, h, title=None):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, PANEL, rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_EDGE, rect, width=2, border_radius=12)
    if title:
        screen.blit(FONT_T.render(title, True, TEXT), (x + 14, y + 10))
    return rect

def draw_grid(grid_left, grid_top, cell, agent_pos):
    grid_px = GRID * cell
    draw_panel_rect(grid_left - 10, grid_top - 10, grid_px + 20, grid_px + 20)

    for r in range(GRID):
        for c in range(GRID):
            x = grid_left + c * cell
            y = grid_top + r * cell

            if (r, c) == LAVA:
                screen.blit(lava_img, (x, y))
            elif (r, c) == CHEST:
                screen.blit(chest_img, (x, y))
            else:
                screen.blit(ground_img, (x, y))

            pygame.draw.rect(screen, (35, 35, 35), pygame.Rect(x, y, cell, cell), 1)
            screen.blit(FONT_S.render(f"s={s2i((r,c))}", True, MUTED), (x + 6, y + 6))

    ar, ac = agent_pos
    screen.blit(agent_img, (grid_left + ac * cell, grid_top + ar * cell))

def draw_update_panel(panel_left, panel_top, panel_w, panel_h, info):
    rect = draw_panel_rect(panel_left, panel_top, panel_w, panel_h, title="Q update (per step)")
    x = rect.x + 14
    y = rect.y + 52

    screen.blit(FONT_B.render(info["tuple"], True, TEXT), (x, y))
    y += 30

    if qeq_img is not None:
        screen.blit(qeq_img, (x, y))
        y += qeq_img.get_height() + 14

    screen.blit(FONT.render(info["plug"], True, TEXT), (x, y))
    y += 24

    pygame.draw.line(screen, PANEL_EDGE, (x, y + 4), (x + panel_w - 40, y + 4), 1)
    y += 16

    screen.blit(FONT_B.render(info["result"], True, GREEN), (x, y))

def draw_q_table(panel_left, panel_top, panel_w, panel_h, Q, updated=(None, None), cur_state=None):
    rect = draw_panel_rect(panel_left, panel_top, panel_w, panel_h, title="Q-table")

    x0, y0 = rect.x + 14, rect.y + 52
    header = "state    UP    DOWN   LEFT  RIGHT"
    screen.blit(FONT_B.render(header, True, MUTED), (x0, y0))
    y = y0 + 26

    s_upd, a_upd = updated
    row_h = 22

    token_w = FONT.size("  0.00")[0]
    space_w = FONT.size(" ")[0]

    for s in range(GRID * GRID):
        row_rect = pygame.Rect(x0 - 6, y - 2, panel_w - 28, row_h + 2)
        if cur_state is not None and s == cur_state:
            pygame.draw.rect(screen, HILITE, row_rect, border_radius=6)

        state_str = f"{s:>5}  "
        screen.blit(FONT.render(state_str, True, TEXT), (x0, y))
        sx = x0 + FONT.size(state_str)[0]

        for a in range(4):
            val = float(Q[s, a])
            col = MUTED if abs(val) < 1e-6 else TEXT
            token = fmt(val)
            screen.blit(FONT.render(token, True, col), (sx, y))

            if s_upd is not None and a_upd is not None and s == s_upd and a == a_upd:
                pygame.draw.rect(
                    screen, ORANGE,
                    pygame.Rect(sx - 4, y - 2, token_w + 8, row_h + 2),
                    2, border_radius=6
                )
            sx += token_w + space_w

        y += row_h

# =========================
# TEXT BUILDERS
# =========================
def build_update_text(state, action, reward, next_state, done, Q_before, Q_after):
    s = s2i(state)
    ns = s2i(next_state)

    old = float(Q_before[s, action])
    max_next = 0.0 if done else float(np.max(Q_before[ns]))
    new = float(Q_after[s, action])
    dQ = new - old

    tup = f"(s, a, r, s', done) = ({s}, {ACTIONS[action]}, {reward:.1f}, {ns}, {done})"

    plug = f"{old:.3f} + {alpha:.2f} [ {reward:.1f} + {gamma:.2f}·{max_next:.3f} − {old:.3f} ] = {new:.3f}"
    res  = f"Q({s},{ACTIONS[action]}) : {old:.3f} → {new:.3f}    (ΔQ = {dQ:+.3f})"
    return {"tuple": tup, "plug": plug, "result": res}

# =========================
# SIM STATE
# =========================
def reset_all():
    state = START
    done = False
    step_n = 0
    updated_cell = (None, None)
    info = {"tuple": "(s, a, r, s', done) = —", "plug": "—", "result": "—"}
    return state, done, step_n, updated_cell, info

args = parse_args()

if args.no_train:
    Q = np.zeros((GRID * GRID, 4), dtype=np.float32)
    best_episode = {"return": -float("inf"), "states": [], "actions": [], "rewards": []}
else:
    print(f"Training for {args.train_episodes} episodes…")
    Q, best_episode = train_q_learning(
        episodes=args.train_episodes,
        max_steps=MAX_STEPS,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        seed=args.seed,
    )
    print(f"Best return during training: {best_episode['return']:.1f}")

rng = np.random.default_rng(args.seed)

state, done, step_n, updated_cell, last_info = reset_all()
auto = False

# If we trained, replay the best episode once (for GIF) before handing control to the user.
replay_actions: list[int] = list(best_episode.get("actions") or [])
replay_rewards: list[float] = list(best_episode.get("rewards") or [])
replay_active = (not args.no_train) and bool(replay_actions)
replay_i = 0

gif_frames: list[np.ndarray] = []
gif_path = Path(args.gif)
gif_record = (not args.no_gif) and replay_active

# =========================
# MAIN LOOP
# =========================
while True:
    clock.tick(60)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)

        if event.type == pygame.VIDEORESIZE:
            WIN_W, WIN_H = max(860, event.w), max(540, event.h)
            screen = pygame.display.set_mode((WIN_W, WIN_H), pygame.RESIZABLE)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit(0)

            if event.key == pygame.K_r:
                state, done, step_n, updated_cell, last_info = reset_all()
                auto = False

            if event.key == pygame.K_RETURN:
                auto = True

            if event.key == pygame.K_SPACE:
                auto = False
                if not done and step_n < MAX_STEPS:
                    step_n += 1

                    # Random again: epsilon-greedy with high epsilon for interactive stepping.
                    a = epsilon_greedy_action(Q, state, epsilon=0.35, rng=rng)

                    ns, r, done = env_step(state, a)

                    Q_before = Q.copy()
                    si = s2i(state)
                    nsi = s2i(ns)

                    max_next = 0.0 if done else float(np.max(Q[nsi]))
                    td_target = r + gamma * max_next
                    Q[si, a] = Q[si, a] + alpha * (td_target - Q[si, a])

                    Q_after = Q.copy()
                    last_info = build_update_text(state, a, r, ns, done, Q_before, Q_after)
                    updated_cell = (si, a)
                    state = ns

    if replay_active and (not done) and step_n < MAX_STEPS:
        # Fast-ish replay for GIF capture
        pygame.time.delay(140)
        step_n += 1

        a = int(replay_actions[replay_i])
        r_expected = float(replay_rewards[replay_i]) if replay_i < len(replay_rewards) else None
        replay_i += 1

        ns, r, done = env_step(state, a)
        if r_expected is not None:
            r = r_expected

        Q_before = Q.copy()
        si = s2i(state)
        nsi = s2i(ns)

        max_next = 0.0 if done else float(np.max(Q[nsi]))
        td_target = r + gamma * max_next
        Q[si, a] = Q[si, a] + alpha * (td_target - Q[si, a])

        Q_after = Q.copy()
        last_info = build_update_text(state, a, r, ns, done, Q_before, Q_after)
        updated_cell = (si, a)
        state = ns

        if replay_i >= len(replay_actions) or done:
            replay_active = False

    if (not replay_active) and auto and (not done) and step_n < MAX_STEPS:
        pygame.time.delay(420)
        step_n += 1

        a = epsilon_greedy_action(Q, state, epsilon=0.35, rng=rng)
        ns, r, done = env_step(state, a)

        Q_before = Q.copy()
        si = s2i(state)
        nsi = s2i(ns)

        max_next = 0.0 if done else float(np.max(Q[nsi]))
        td_target = r + gamma * max_next
        Q[si, a] = Q[si, a] + alpha * (td_target - Q[si, a])

        Q_after = Q.copy()
        last_info = build_update_text(state, a, r, ns, done, Q_before, Q_after)
        updated_cell = (si, a)
        state = ns

    # Layout
    right_w = int(WIN_W * 0.42)
    right_w = max(RIGHT_PANEL_W_MIN, min(RIGHT_PANEL_W_MAX, right_w))

    avail_h_grid = WIN_H - PAD - PAD - FOOTER_H
    avail_w_grid = WIN_W - (PAD + right_w + PAD + PAD)

    cell_h = avail_h_grid / GRID
    cell_w = avail_w_grid / GRID
    cell = int(min(cell_h, cell_w))
    cell = max(90, min(cell, 240))

    ensure_scaled_tiles(cell)

    grid_px = GRID * cell
    grid_left = PAD
    grid_top  = PAD

    panel_left = grid_left + grid_px + PAD
    panel_top  = PAD

    update_h = int(WIN_H * 0.30)
    update_h = max(UPDATE_H_MIN, min(UPDATE_H_MAX, update_h))

    update_top = panel_top
    table_top  = update_top + update_h + GAP
    table_h = WIN_H - table_top - PAD - FOOTER_H
    table_h = max(280, table_h)

    scale = cell / 120.0
    FONT, FONT_S, FONT_B, FONT_T = make_fonts(scale)

    ensure_scaled_equation(right_w)

    # Draw
    screen.fill((245, 247, 250))
    draw_grid(grid_left, grid_top, cell, state)
    draw_update_panel(panel_left, update_top, right_w, update_h, last_info)
    draw_q_table(panel_left, table_top, right_w, table_h, Q, updated=updated_cell, cur_state=s2i(state))

    footer = f"step {step_n}/{MAX_STEPS}    α={alpha:.2f}  γ={gamma:.2f}"
    screen.blit(FONT_B.render(footer, True, MUTED), (grid_left, WIN_H - PAD - FOOTER_H + 10))

    if gif_record:
        # Capture after drawing, before flip.
        frame = pygame.surfarray.array3d(screen)  # (W,H,3)
        frame = np.transpose(frame, (1, 0, 2)).astype(np.uint8)  # (H,W,3)
        gif_frames.append(frame)

    if gif_record and (not replay_active):
        gif_record = False
        try:
            save_gif(gif_frames, gif_path, fps=args.gif_fps)
            print(f"Saved best run GIF: {gif_path}")
        except Exception as e:
            print(f"Failed to save GIF to {gif_path}: {e}")

        if not args.interactive:
            pygame.quit()
            sys.exit(0)

    pygame.display.flip()
