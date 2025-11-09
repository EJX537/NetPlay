#!/usr/bin/env python3
"""
Render a NetHack scenario without LLM interaction.
Useful for testing scenario des files and seeing what actually spawns.

Based on debug_render.py approach using EnvAgentShim pattern.
"""

import argparse
import os
import sys

# Add parent directory to path to import netplay modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.nethack_agent.skill_selection import render_ascii_map_cropped, render_tileset_map_cropped


class EnvAgentShim:
    """Minimal shim to make environment look like an agent for rendering functions."""
    def __init__(self, env, seed=None):
        self.env = env
        try:
            if seed is not None:
                obs, info = env.reset(seed=seed)
            else:
                obs, info = env.reset()
            self.last_observation = obs
        except Exception:
            try:
                self.last_observation = env.env.last_observation
            except Exception:
                self.last_observation = {}

        try:
            import numpy as np
            # Get actual agent position from blstats in observation
            blstats_array = self.last_observation.get('blstats', [])
            if len(blstats_array) >= 2:
                agent_x = int(blstats_array[0])
                agent_y = int(blstats_array[1])
            else:
                # Fallback to center if blstats not available
                g = np.array(self.last_observation.get('glyphs'))
                rows, cols = g.shape
                agent_x = cols // 2
                agent_y = rows // 2

            class BlStats: pass
            b = BlStats()
            b.x = agent_x
            b.y = agent_y
            self.blstats = b

            # Also create current_level for ASCII map renderer
            class Level: pass
            level = Level()
            level.glyphs = self.last_observation.get('glyphs')
            level.chars = self.last_observation.get('chars')
            self.current_level = level
        except Exception:
            class BlStats: pass
            b = BlStats()
            b.x = 0
            b.y = 0
            self.blstats = b


def render_scenario(des_file, character='pri-cha-elf', seed=None, text_only=False, radius=20):
    """
    Render a scenario and show the initial state.

    Args:
        des_file: Path to the .des file
        character: Character string (default: priest chaotic elf)
        seed: Random seed (optional)
        text_only: If True, only show ASCII map. If False, also generate PNG.
        radius: Map radius to render (default: 20)
    """

    # Create environment
    print(f"Loading scenario: {des_file}")
    print(f"Character: {character}")
    if seed is not None:
        print(f"Seed: {seed}")
    print(f"Text only: {text_only}")
    print(f"Map radius: {radius}")
    print()

    env = NethackGymnasiumWrapper(
        render_mode='human' if not text_only else None,
        des_file=des_file,
        character=character,
        autopickup=False
    )

    # Create agent shim with seed
    agent = EnvAgentShim(env, seed=seed)

    # Print initial state
    print("="*80)
    print("SCENARIO RENDERED - ASCII MAP WITH LEGEND")
    print("="*80)

    # Always render ASCII map with legend
    ascii_map = render_ascii_map_cropped(agent, radius=radius)
    print("\n" + ascii_map)

    # Optionally render tileset PNG
    if not text_only:
        print("\n" + "="*80)
        print("TILESET PNG RENDERING")
        print("="*80)
        try:
            img_array, legend_text = render_tileset_map_cropped(agent, radius=radius)
            if img_array is not None:
                # Convert numpy array to PIL Image
                from PIL import Image
                img = Image.fromarray(img_array.astype('uint8'), mode='RGB')

                # Save the image
                output_path = os.path.join(os.path.dirname(des_file),
                                          os.path.basename(des_file).replace('.des', '_render.png'))
                img.save(output_path)
                print(f"\nSaved PNG to: {output_path}")
                print(f"\nLegend (glyph IDs and descriptions):")
                print(legend_text if legend_text else 'None')
            else:
                print(f"\nPNG rendering failed: {legend_text}")
        except Exception as e:
            import traceback
            print(f"\nError rendering PNG: {e}")
            traceback.print_exc()

    env.close()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description='Render a NetHack scenario without LLM - shows initial state only.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render with ASCII map only (text)
  python tools/render_scenario.py -des_file scenarios/game_mechanics/wand.des --text_only

  # Render with both ASCII and PNG
  python tools/render_scenario.py -des_file scenarios/game_mechanics/wand.des

  # Render with specific seed and larger radius
  python tools/render_scenario.py -des_file scenarios/game_mechanics/wand.des -seed 12345 -radius 30
        """
    )

    parser.add_argument(
        '-des_file',
        type=str,
        required=True,
        help='Path to the .des scenario file'
    )

    parser.add_argument(
        '-character',
        type=str,
        default='pri-cha-elf',
        help='Character specification (default: pri-cha-elf)'
    )

    parser.add_argument(
        '-seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--text_only',
        action='store_true',
        help='Only output ASCII map, skip PNG rendering'
    )

    parser.add_argument(
        '-radius',
        type=int,
        default=20,
        help='Map radius to render (default: 20)'
    )

    args = parser.parse_args()

    # Check if des file exists
    if not os.path.exists(args.des_file):
        print(f"Error: des file not found: {args.des_file}")
        sys.exit(1)

    render_scenario(
        des_file=args.des_file,
        character=args.character,
        seed=args.seed,
        text_only=args.text_only,
        radius=args.radius
    )

if __name__ == '__main__':
    main()
