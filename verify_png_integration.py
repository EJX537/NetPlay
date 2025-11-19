#!/usr/bin/env python3
"""Verify PNG mode integration in actual agent execution."""

import os
import sys

# Set minimal environment
os.environ['DEBUG_LLM_RESPONSES'] = 'false'

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from langchain.schema import AIMessage

class MockLLM:
    """Mock LLM that captures what it receives."""

    def __init__(self, *args, **kwargs):
        self.model = 'gemini/gemini-2.0-flash-exp'
        self.last_messages = None

    def get_num_tokens(self, text):
        return len(text) // 4

    def predict_messages(self, messages):
        """Capture messages and check for image data."""
        self.last_messages = messages

        # Check last message for image
        last_msg = messages[-1]
        has_image = hasattr(last_msg, 'additional_kwargs') and 'image_data' in last_msg.additional_kwargs

        print(f"\n📨 LLM.predict_messages() called:")
        print(f"  - Total messages: {len(messages)}")
        print(f"  - Last message has image: {has_image}")

        if has_image:
            img_data = last_msg.additional_kwargs['image_data']
            print(f"  - Image data length: {len(img_data)} chars")
            print(f"  - Image prefix: {img_data[:50]}...")

        # Return valid skill selection
        return AIMessage(content='{"skill": "move_to", "args": {"x": 40, "y": 10}}')

def main():
    print("=" * 60)
    print("PNG Mode Integration Verification")
    print("=" * 60)

    # Create environment
    print("\n1️⃣  Creating NetHack environment...")
    env = NethackGymnasiumWrapper(
        render_mode='rgb_array',
        des_file='scenarios/game_mechanics/wand.des',
        autopickup=False
    )
    env.reset(seed=42)
    print("  ✓ Environment ready")

    # Create mock LLM
    print("\n2️⃣  Creating mock LLM...")
    llm = MockLLM()
    print("  ✓ Mock LLM ready")

    # Create agent with PNG mode
    print("\n3️⃣  Creating agent with map_mode='png'...")
    agent = create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=500,
        log_folder='./runs/verify_png',
        map_mode='png',
        map_radius=8
    )
    agent.init()
    agent.set_task('Move around and explore')
    print("  ✓ Agent created with PNG mode")

    # Execute one step
    print("\n4️⃣  Executing agent step...")
    try:
        gen = agent.run()
        next(gen)

        # Verify mock LLM was called with image
        if llm.last_messages is None:
            print("  ❌ LLM was not called!")
            return False

        last_msg = llm.last_messages[-1]
        has_image = hasattr(last_msg, 'additional_kwargs') and 'image_data' in last_msg.additional_kwargs

        if has_image:
            print("  ✅ SUCCESS: Agent passed PNG image to LLM!")
            return True
        else:
            print("  ❌ FAILURE: No image data in message")
            return False

    except Exception as e:
        print(f"  ❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent.close()

    print("\n" + "=" * 60)

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
