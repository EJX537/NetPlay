#!/usr/bin/env python3
"""
Test the wand scenario with ASCII map enabled.
"""
import os

# Enable debug mode to see all LLM prompts and responses
os.environ['DEBUG_LLM_RESPONSES'] = 'true'

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper
from netplay.logging.nethack_monitor import NethackH5PYMonitor

# Create LLM
print("Initializing Gemini 2.5 Flash...")
llm = LiteLLMWrapper(model='gemini/gemini-2.5-flash', temperature=0.0, max_tokens=2048)

# Create environment with wand scenario
print("Loading wand scenario...")
env = NethackGymnasiumWrapper(
    render_mode='rgb_array',
    des_file='scenarios/game_mechanics/wand.des',
    autopickup=False
)

# Create agent
log_folder = './runs/wand_test'
os.makedirs(log_folder, exist_ok=True)

# Wrap with monitor
env = NethackH5PYMonitor(env, os.path.join(log_folder, "trajectories.h5py"))

# Set seed
env.reset(seed=12345)

agent = create_llm_agent(
    env=env,
    llm=llm,
    memory_tokens=500,
    log_folder=log_folder,
    render=False
)

# Enable ASCII map with larger radius to see more of the level
agent.skill_selector.map_radius = 20
print(f"ASCII map enabled with radius: {agent.skill_selector.map_radius}")

agent.init()
agent.set_task('Pick up the wand and use it to zap the statue.')

print('=' * 80)
print('WAND SCENARIO TEST WITH ASCII MAP')
print('Task: Pick up the wand and use it to zap the statue.')
print('=' * 80)
print()

steps = 0
max_steps = 100

try:
    for step in agent.run():
        if step.thoughts:
            print(f'Thinking: {step.thoughts}')
        if step.executed_action():
            print(f'Executed action: {step.step_data.action.name}')

        steps += 1
        if steps >= max_steps:
            print(f'\n⚠️  Reached {max_steps} step limit')
            break

except KeyboardInterrupt:
    print('\n⚠️  Test interrupted by user')
except Exception as e:
    print(f'\n❌ ERROR: {e}')
    import traceback
    traceback.print_exc()
finally:
    agent.close()

print(f'\n{"="*80}')
print(f'TEST COMPLETE')
print(f'{"="*80}')
print(f'Total steps executed: {steps}')
print(f'Log folder: {log_folder}')

# Show final score
print('\nFinal Score:')
os.system(f'python scripts/show_score.py {os.path.join(log_folder, "trajectories.h5py")}')
print(f'{"="*80}\n')
