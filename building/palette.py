from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Palette:
    name: str
    wall: str
    trim: str
    pillar: str
    window: str
    roof: str
    roof_trim: str
    floor: str
    light: str


PALETTES: Sequence[Palette] = (
    Palette(
        name="warm_brick",
        wall="brick_block",
        trim="quartz_block",
        pillar="stonebrick",
        window="glass",
        roof="nether_brick",
        roof_trim="stone_slab",
        floor="planks",
        light="glowstone",
    ),
    Palette(
        name="classic_stone",
        wall="stonebrick",
        trim="quartz_block",
        pillar="cobblestone",
        window="glass",
        roof="planks",
        roof_trim="stone_slab",
        floor="planks",
        light="sea_lantern",
    ),
    Palette(
        name="sand_quartz",
        wall="sandstone",
        trim="quartz_block",
        pillar="stonebrick",
        window="glass",
        roof="brick_block",
        roof_trim="stone_slab",
        floor="planks",
        light="glowstone",
    ),
    Palette(
        name="fortress",
        wall="stonebrick",
        trim="cobblestone",
        pillar="stone",
        window="glass",
        roof="nether_brick",
        roof_trim="stone_slab",
        floor="stone",
        light="glowstone",
    ),
    Palette(
        name="mediterranean",
        wall="sandstone",
        trim="quartz_block",
        pillar="quartz_block",
        window="glass",
        roof="brick_block",
        roof_trim="stone_slab",
        floor="sandstone",
        light="sea_lantern",
    ),
    Palette(
        name="timber",
        wall="planks",
        trim="stonebrick",
        pillar="log",
        window="glass",
        roof="nether_brick",
        roof_trim="stone_slab",
        floor="planks",
        light="glowstone",
    ),
)


def _validate_palette(palette: Palette) -> None:
    if palette.wall == palette.window:
        raise ValueError(f"Invalid palette {palette.name}: wall and window must differ.")
    if palette.roof == palette.wall:
        raise ValueError(f"Invalid palette {palette.name}: roof and wall must differ.")
    if palette.trim == palette.wall and palette.trim == palette.roof:
        raise ValueError(f"Invalid palette {palette.name}: trim must provide contrast.")


def choose_palette(rng: random.Random, style_id: Optional[int] = None) -> Palette:
    if style_id is None:
        palette = rng.choice(list(PALETTES))
    else:
        palette = PALETTES[style_id % len(PALETTES)]
    _validate_palette(palette)
    return palette
