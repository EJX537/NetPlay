#!/usr/bin/env python3
"""
Example demonstrating the three different map modes for the NetPlay agent.
"""

from netplay import create_llm_agent, MapMode
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper

# Example 1: No map (original implementation)
def create_agent_no_map(env, llm):
    """Create agent without map rendering - original implementation."""
    return create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=500,
        log_folder='./runs/example_no_map',
        map_mode='none'  # or MapMode.NONE
    )

# Example 2: ASCII map with semantic legend
def create_agent_ascii_map(env, llm, radius=20):
    """Create agent with ASCII map and semantic object descriptions."""
    return create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=500,
        log_folder='./runs/example_ascii_map',
        map_mode='ascii',  # or MapMode.ASCII
        map_radius=radius  # default is 10
    )

# Example 3: PNG tileset rendering with vision model support
def create_agent_png_map(env, llm, radius=15):
    """Create agent with PNG map mode for vision models (e.g., GPT-4 Vision, Gemini Vision)."""
    return create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=500,
        log_folder='./runs/example_png_map',
        map_mode='png',  # or MapMode.PNG - renders tileset map as image
        map_radius=radius
    )

if __name__ == '__main__':
    print("NetPlay Map Mode Examples")
    print("=" * 80)
    print("\nAvailable map modes:")
    print("  - 'none' or MapMode.NONE:  No map rendering (original)")
    print("  - 'ascii' or MapMode.ASCII: ASCII map with semantic legend")
    print("  - 'png' or MapMode.PNG:     PNG tileset image for vision models")
    print("\nExample usage:")
    print("  agent = create_llm_agent(..., map_mode='ascii', map_radius=20)")
    print("  agent = create_llm_agent(..., map_mode='png', map_radius=15)  # For vision models")
    print("\nNote: PNG mode requires a vision-capable model (e.g., gemini/gemini-2.0-flash-exp)")
    print("=" * 80)
