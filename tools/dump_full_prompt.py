#!/usr/bin/env python3
"""Dump the complete prompt as assembled by the agent descriptors + skills.

This script creates a real `NetHackAgent` (with a dummy LLM to avoid network calls),
loads the provided `.des` file into a Nethack env, calls `agent.describe_current_state()`
and then assembles the final prompt using the same `construct_prompt` and skills
repository that `create_llm_agent` uses.

It prints and writes the prompt to `runs/interactive/complete_prompt.txt`.
"""
import os
import sys

from netplay import create_llm_agent
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
    if len(sys.argv) > 1:
        des = sys.argv[1]
    else:
        print('Usage: dump_full_prompt.py <path/to/file.des>')
        return

    # Create env and agent
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

    # Build agent with the same descriptors (create a minimal one via create_llm_agent)
    agent = create_llm_agent(env=env, llm=dummy, memory_tokens=800, log_folder=out_dir, render=False)

    # Reset the environment and initialize the agent so descriptors have
    # access to the latest observation (last_observation).
    try:
        obs, info = env.reset()
    except Exception:
        # Some wrappers return only obs
        try:
            obs = env.reset()
        except Exception:
            obs = None

    # Initialize agent (this also sets up memory/messages)
    try:
        agent.init()
    except Exception:
        pass

    # Describe current state
    state_description = agent.describe_current_state()

    # Build skills listing text (same as used by construct_prompt)
    skills_text = skill_repo.get_skills_description()

    # Use the CHOOSE_SKILL_PROMPT as the task body to exactly replicate the real prompt
    task_text = skill_selection.CHOOSE_SKILL_PROMPT

    prompt = "\n\n".join([state_description, f"Skills:\n{skills_text}", task_text])

    out_path = os.path.join(out_dir, 'complete_prompt.txt')
    with open(out_path, 'w') as f:
        f.write(prompt)

    print('Wrote complete prompt to', out_path)
    print('--- prompt start ---')
    print(prompt)
    print('--- prompt end ---')


if __name__ == '__main__':
    main()
