#!/usr/bin/env python3
"""
Test script to verify PNG map mode works correctly.
This will render a tileset map and encode it as base64 without actually calling a vision model.
"""
import os
os.environ['DEBUG_LLM_RESPONSES'] = 'false'

from netplay import create_llm_agent, MapMode
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper

# Create environment
env = NethackGymnasiumWrapper(
    render_mode='rgb_array',
    des_file='scenarios/game_mechanics/wand.des',
    autopickup=False
)

# Initialize environment
env.reset(seed=12345)

# Create a dummy LLM (we won't actually call it)
llm = LiteLLMWrapper(model='gemini/gemini-2.0-flash-exp', temperature=0.0, max_tokens=2048)

# Create agent with PNG map mode
print("Creating agent with PNG map mode...")
agent = create_llm_agent(
    env=env,
    llm=llm,
    memory_tokens=500,
    log_folder='./runs/test_png',
    render=False,
    map_mode='png',  # PNG mode!
    map_radius=10
)

agent.init()
print("Agent initialized successfully!")

# Trigger skill selection to generate the prompt with image
print("\nGenerating prompt with PNG map...")
try:
    # Access the skill selector's internal method to see if image gets generated
    from netplay.nethack_agent.skill_selection import assemble_prompt_with_map
    from netplay.core.skill_repository import SkillRepository
    import netplay.nethack_agent.skills as skill_module

    # Build skill repository
    skill_list = [getattr(skill_module, name) for name in dir(skill_module)
                  if hasattr(getattr(skill_module, name), 'skill')]
    skills = SkillRepository(skill_list)

    # Generate prompt
    prompt = assemble_prompt_with_map(agent, skills, "Test task", MapMode.PNG, 10)

    # Check if image data was attached
    image_data = getattr(agent, '_pending_image_data', None)

    if image_data:
        print("✓ PNG image successfully generated and encoded!")
        print(f"  Image data length: {len(image_data)} characters")
        print(f"  Image format: {image_data[:30]}...")

        # Save the prompt for inspection
        os.makedirs('./runs/test_png', exist_ok=True)
        with open('./runs/test_png/prompt_with_image.txt', 'w') as f:
            f.write(prompt)
            f.write("\n\n" + "="*80 + "\n")
            f.write("IMAGE DATA:\n")
            f.write(f"Length: {len(image_data)} characters\n")
            f.write(f"Format: {image_data[:100]}...\n")

        print(f"✓ Prompt saved to ./runs/test_png/prompt_with_image.txt")

        # Optionally decode and save the image
        try:
            import base64
            from io import BytesIO
            from PIL import Image

            # Extract base64 data (remove "data:image/png;base64," prefix)
            b64_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(b64_data)
            img = Image.open(BytesIO(img_bytes))

            img.save('./runs/test_png/generated_map.png')
            print(f"✓ Decoded image saved to ./runs/test_png/generated_map.png")
            print(f"  Image size: {img.size}")
        except Exception as e:
            print(f"✗ Failed to decode image: {e}")
    else:
        print("✗ No image data was generated")
        print(f"  Prompt length: {len(prompt)} characters")

    # Check if the prompt mentions the map
    if "Map Image:" in prompt or "tileset" in prompt.lower():
        print("✓ Prompt references the map image")
    else:
        print("✗ Prompt doesn't reference the map")

except Exception as e:
    print(f"✗ Error during test: {e}")
    import traceback
    traceback.print_exc()

agent.close()
print("\nTest complete!")
