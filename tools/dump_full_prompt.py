#!/usr/bin/env python3
"""Dump the complete prompt as assembled by the agent descriptors + skills.

This script creates a real `NetHackAgent` (with a dummy LLM to avoid network calls),
loads the provided `.des` file into a Nethack env, calls `agent.describe_current_state()`
and then assembles the final prompt using the same `construct_prompt` and skills
repository that `create_llm_agent` uses.

It dumps all three prompt variations:
- No map (original implementation)
- ASCII map with semantic legend
- PNG map (currently same as no map, reserved for future image embedding)

Prompts are written to:
- runs/interactive/prompt_no_map.txt
- runs/interactive/prompt_ascii_map.txt
- runs/interactive/prompt_png_map.txt
"""
import os
import sys

from netplay import create_llm_agent, MapMode
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
import netplay.nethack_agent.skills as skills_module
from netplay.core.skill_repository import SkillRepository
from netplay.nethack_agent import skill_selection

from langchain.schema import AIMessage


class DummyLLM:
    def __init__(self):
        pass
    def get_num_tokens(self, content: str) -> int:
        if content is None:
            return 0
        return max(1, len(content) // 4)
    def predict_messages(self, messages):
        # Return an empty JSON so parsing won't accidentally be used; we only
        # need a predict_messages implementation for the skill selector init.
        return AIMessage(content='{}')


def main():
    out_dir = os.path.join('runs', 'interactive')
    os.makedirs(out_dir, exist_ok=True)

    des = None
    map_radius = 20  # Use a larger radius for better visualization

    if len(sys.argv) > 1:
        des = sys.argv[1]
    else:
        print('Usage: dump_full_prompt.py <path/to/file.des> [map_radius]')
        print()
        print('Generates three prompt variations:')
        print('  1. No map (original implementation)')
        print('  2. ASCII map with semantic legend')
        print('  3. PNG map (reserved for future image embedding)')
        print()
        print('Arguments:')
        print('  <path/to/file.des>  Path to NetHack .des scenario file')
        print('  [map_radius]        Optional map radius (default=20)')
        print()
        print('Example:')
        print('  python3 tools/dump_full_prompt.py scenarios/game_mechanics/wand.des 15')
        print()
        print('Output files:')
        print('  runs/interactive/prompt_no_map.txt')
        print('  runs/interactive/prompt_ascii_map.txt')
        print('  runs/interactive/prompt_png_map.txt')
        return

    if len(sys.argv) > 2:
        try:
            map_radius = int(sys.argv[2])
        except ValueError:
            print(f'Warning: Invalid map_radius "{sys.argv[2]}", using default=20')

    # Create env
    env = NethackGymnasiumWrapper(render_mode='human', des_file=des, autopickup=False)
    dummy = DummyLLM()

    # Build skill repo like create_llm_agent
    skill_repo = SkillRepository([
        *skills_module.ALL_COMMAND_SKILLS,
        skills_module.set_avoid_monster_flag,
        skills_module.melee_attack,
        skills_module.explore_level,
        skills_module.move_to,
        skills_module.go_to,
        skills_module.press_key,
        skills_module.type_text,
    ])

    print(f'Generating prompts for: {des}')
    print(f'Map radius: {map_radius}')
    print(f'Output directory: {out_dir}')
    print('=' * 80)

    # Generate prompts for all three map modes
    modes = [
        (MapMode.NONE, 'no_map', 'No Map (Original)'),
        (MapMode.ASCII, 'ascii_map', f'ASCII Map (radius={map_radius})'),
        (MapMode.PNG, 'png_map', 'PNG Map (reserved for future image embedding)'),
    ]

    for map_mode, filename_suffix, description in modes:
        print(f'\n{description}')
        print('-' * 80)

        # Create agent with specific map mode
        agent = create_llm_agent(
            env=env,
            llm=dummy,
            memory_tokens=800,
            log_folder=out_dir,
            render=False,
            map_mode=map_mode,
            map_radius=map_radius
        )

        # Reset the environment and initialize the agent
        try:
            obs, info = env.reset()
        except Exception:
            try:
                obs = env.reset()
            except Exception:
                obs = None

        # Initialize agent
        try:
            agent.init()
        except Exception:
            pass

        # Generate the prompt using the agent's skill selector
        # This ensures we use the exact same logic as during actual gameplay
        state_description = agent.describe_current_state()
        task_text = skill_selection.CHOOSE_SKILL_PROMPT

        # Use assemble_prompt_with_map to get the prompt with the appropriate map
        prompt = skill_selection.assemble_prompt_with_map(
            agent,
            skill_repo,
            task_text,
            map_mode,
            map_radius
        )

        # Write to file
        out_path = os.path.join(out_dir, f'prompt_{filename_suffix}.txt')
        with open(out_path, 'w') as f:
            f.write(prompt)

        print(f'✓ Wrote to: {out_path}')
        print(f'  Length: {len(prompt)} characters, ~{len(prompt.split())} words')

        # Show a preview of the map section if present
        if 'Map:' in prompt or 'Map (' in prompt:
            map_start = prompt.find('\nMap')
            if map_start != -1:
                map_end = prompt.find('\n\n', map_start + 1)
                if map_end == -1:
                    map_end = map_start + 500
                map_preview = prompt[map_start:map_end]
                # Count lines in map section
                map_lines = map_preview.count('\n')
                print(f'  Map section: {map_lines} lines')

    print('\n' + '=' * 80)
    print('All prompts generated successfully!')
    print(f'Files written to: {out_dir}/')
    print('  - prompt_no_map.txt')
    print('  - prompt_ascii_map.txt')
    print('  - prompt_png_map.txt')


if __name__ == '__main__':
    main()
