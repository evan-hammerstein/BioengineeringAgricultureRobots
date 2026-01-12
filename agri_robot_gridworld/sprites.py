# sprites.py
import os
from typing import Dict, Optional

import pygame
from config import Config


def load_scaled_image(path: str, size: int) -> pygame.Surface:
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, (size, size))


class SpriteBank:
    """
    Loads and stores:
      - tile sprites (by name)
      - robot sprite

    If a sprite file is missing or fails to load, we fall back to a colored tile
    so the program still runs.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tiles: Dict[str, pygame.Surface] = {}
        self.robot: Optional[pygame.Surface] = None
        self.plant_stages: Dict[int, pygame.Surface] = {}

        self._load_tiles()
        self._load_robot()
        self._load_plant_stages()

    def _load_tiles(self):
        for key, path in self.cfg.TILE_SPRITES.items():
            self.tiles[key] = self._load_or_fallback_tile(key, path)

    def _load_or_fallback_tile(self, key: str, path: str) -> pygame.Surface:
        size = self.cfg.TILE_SIZE

        if os.path.exists(path):
            try:
                return load_scaled_image(path, size)
            except Exception:
                pass

        # Fallback
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        color = self.cfg.FALLBACK_COLORS.get(key, (255, 0, 255))
        surf.fill(color)
        return surf

    def _load_robot(self):
        size = self.cfg.TILE_SIZE
        path = self.cfg.ROBOT_SPRITE

        if os.path.exists(path):
            try:
                self.robot = load_scaled_image(path, size)
                return
            except Exception:
                pass

        # Fallback robot: simple circle
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 255))
        pygame.draw.circle(surf, (0, 0, 0), (size // 2, size // 2), size // 3, 3)
        self.robot = surf

    def _load_plant_stages(self):
        paths = getattr(self.cfg, "PLANT_STAGE_SPRITES", None) or {}
        scale = float(getattr(self.cfg, "PLANT_STAGE_SPRITE_SCALE", 0.75) or 0.75)
        size = max(1, int(round(self.cfg.TILE_SIZE * scale)))

        for stage, path in paths.items():
            try:
                stage_i = int(stage)
            except Exception:
                continue

            if os.path.exists(path):
                try:
                    self.plant_stages[stage_i] = load_scaled_image(path, size)
                except Exception:
                    pass
