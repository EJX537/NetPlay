#!/usr/bin/env python3
"""Debug script to find where the newt statue actually is."""

import sys
sys.path.insert(0, '/workspaces/NetPlay')

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper
from netplay.nethack_agent.describe import describe_glyph

# Create LLM (minimal, won't actually use it)
llm = LiteLLMWrapper(model='gemini/gemini-2.5-flash', temperature=0.0, max_tokens=2048)

# Create environment
env = NethackGymnasiumWrapper(
    render_mode='rgb_array',
    des_file='scenarios/game_mechanics/wand.des',
    autopickup=False
)

# Reset with seed
env.reset(seed=12345)

# Create agent
agent = create_llm_agent(
    env=env,
    llm=llm,
    memory_tokens=500,
    log_folder='./runs/debug_statue',
    render=False
)

agent.init()

glyphs = agent.last_observation['glyphs']
tty_chars = agent.last_observation['tty_chars']

print(f"Scanning {len(glyphs)}x{len(glyphs[0])} grid for glyph 5913 (newt statue)...")
print()

found_statue = False
all_colons = []

for r in range(len(glyphs)):
    for c in range(len(glyphs[0])):
        glyph = glyphs[r][c]
        tty_code = int(tty_chars[r][c])

        # Check for statue
        if glyph == 5913:
            tty_char = chr(tty_code) if 0 < tty_code < 128 else f'(code:{tty_code})'
            print(f"✓ FOUND STATUE: glyph 5913 at row={r}, col={c}")
            print(f"  tty_char: '{tty_char}' (ASCII {tty_code})")
            print(f"  Description: {describe_glyph(glyph)}")
            found_statue = True

        # Track all colons
        if tty_code == 58:  # ASCII for ':'
            all_colons.append((r, c, glyph))

print()
if not found_statue:
    print("✗ Glyph 5913 (newt statue) NOT FOUND!")
    print("  The statue may not have spawned.")

print()
print(f"Found {len(all_colons)} ':' character(s) in tty_chars:")
for r, c, glyph in all_colons:
    desc = describe_glyph(glyph)
    print(f"  row={r}, col={c}, glyph={glyph} ({desc})")
