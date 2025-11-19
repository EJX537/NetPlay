#!/usr/bin/env python3
"""
Example: Using PNG map mode with a vision-capable model.

This demonstrates how to use the PNG map rendering feature with models that support image inputs.
Supported vision models include:
- gemini/gemini-2.0-flash-exp
- gemini/gemini-1.5-pro
- gpt-4-vision-preview
- gpt-4o
- claude-3-opus
- claude-3-sonnet
"""
import os

from netplay import create_llm_agent, MapMode
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper

# Create environment
env = NethackGymnasiumWrapper(
    render_mode='rgb_array',
    des_file='scenarios/game_mechanics/wand.des',
    autopickup=False
)

env.reset(seed=12345)

# Create LLM wrapper with a vision-capable model
# Note: Requires appropriate API key (GEMINI_API_KEY, OPENAI_API_KEY, etc.)
llm = LiteLLMWrapper(
    model='gemini/gemini-2.0-flash-exp',  # Vision-capable model
    temperature=0.0,
    max_tokens=2048
)

# Create agent with PNG map mode
print("Creating agent with PNG map mode for vision model...")
agent = create_llm_agent(
    env=env,
    llm=llm,
    memory_tokens=500,
    log_folder='./runs/png_vision_example',
    render=False,
    map_mode='png',      # PNG mode with image embedding
    map_radius=10        # Crop radius around agent
)

agent.init()
agent.set_task('Pick up the wand and use it to zap the statue.')

print("Agent initialized with PNG map mode!")
print("\nThe agent will receive:")
print("- Text description of the game state")
print("- PNG image of the tileset-rendered map")
print("- Legend of visible objects in the map")
print("\nRunning agent...")

# Run for a few steps
steps = 0
max_steps = 5

try:
    for step in agent.run():
        steps += 1
        print(f"Step {steps} completed")
        if steps >= max_steps:
            print(f"\nStopping after {max_steps} steps (demo)")
            break
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

agent.close()
print("\nExample complete!")
print(f"Check ./runs/png_vision_example/ for logs and rendered map images")
