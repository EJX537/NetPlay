#!/usr/bin/env python3
"""
Test the wand scenario with 5 seeds and compile results.
"""
import os
import sys
from datetime import datetime

# Default seeds from run_scenarios.py
SEEDS = [779726, 474862, 151437, 10518, 380261]

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper
from netplay.logging.nethack_monitor import NethackH5PYMonitor

def run_test_with_seed(seed, log_folder):
    """Run wand scenario test with a specific seed."""
    print(f"\n{'='*80}")
    print(f"TESTING WITH SEED: {seed}")
    print(f"{'='*80}\n")

    # Create LLM
    llm = LiteLLMWrapper(model='gemini/gemini-2.5-flash', temperature=0.0, max_tokens=2048)

    # Create environment with wand scenario
    env = NethackGymnasiumWrapper(
        render_mode='rgb_array',
        des_file='scenarios/game_mechanics/wand.des',
        autopickup=False
    )

    # Create seed-specific log folder
    seed_log_folder = os.path.join(log_folder, f'seed_{seed}')
    os.makedirs(seed_log_folder, exist_ok=True)

    # Wrap with monitor
    env = NethackH5PYMonitor(env, os.path.join(seed_log_folder, "trajectories.h5py"))

    # Set seed
    env.reset(seed=seed)

    agent = create_llm_agent(
        env=env,
        llm=llm,
        memory_tokens=500,
        log_folder=seed_log_folder,
        render=False
    )

    # Enable ASCII map with larger radius
    agent.skill_selector.map_radius = 20

    agent.init()
    agent.set_task('Pick up the wand and use it to zap the statue.')

    steps = 0
    max_steps = 100
    finished = False

    try:
        for step in agent.run():
            steps += 1
            if steps >= max_steps:
                print(f'\n⚠️  Reached {max_steps} step limit')
                break
    except KeyboardInterrupt:
        print('\n⚠️  Test interrupted by user')
        return None
    except Exception as e:
        print(f'\n❌ ERROR: {e}')
        import traceback
        traceback.print_exc()
    finally:
        try:
            agent.close()
        except Exception as e:
            # Ignore cleanup errors
            pass

    # Get final score
    import h5py
    try:
        f = h5py.File(os.path.join(seed_log_folder, "trajectories.h5py"), 'r')
        trajs = f['trajectories']
        ids = sorted([int(k) for k in trajs.keys()])
        if ids:
            tid = str(ids[-1])
            bl = trajs[tid]['observations']['blstats'][:]
            import numpy as np
            nonzero_rows = np.where(np.any(bl != 0, axis=1))[0]
            if len(nonzero_rows) > 0:
                last_idx = int(nonzero_rows[-1])
                last_bl = bl[last_idx]
                score = int(last_bl[9])
            else:
                score = 0

            # Get rewards
            if 'rewards' in trajs[tid]:
                rewards = trajs[tid]['rewards'][:]
                sum_rewards = float(rewards.sum())
            else:
                sum_rewards = 0.0
        else:
            score = 0
            sum_rewards = 0.0
        f.close()
    except Exception as e:
        print(f"Error reading score: {e}")
        score = 0
        sum_rewards = 0.0

    result = {
        'seed': seed,
        'steps': steps,
        'score': score,
        'rewards': sum_rewards,
        'max_steps_reached': steps >= max_steps
    }

    print(f"\nSeed {seed} Results:")
    print(f"  Steps: {steps}")
    print(f"  Score: {score}")
    print(f"  Rewards: {sum_rewards}")
    print(f"  Success: {not result['max_steps_reached']}")

    return result

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_log_folder = f'./runs/wand_5seeds_{timestamp}'
    os.makedirs(base_log_folder, exist_ok=True)

    print(f"{'='*80}")
    print(f"WAND SCENARIO - 5 SEED TEST")
    print(f"Timestamp: {timestamp}")
    print(f"Log folder: {base_log_folder}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*80}")

    results = []
    for seed in SEEDS:
        result = run_test_with_seed(seed, base_log_folder)
        if result is None:  # Test was interrupted
            print(f"\n⚠️  Testing interrupted. Partial results will be saved.")
            break
        results.append(result)

    if not results:
        print("No results to compile.")
        return

    # Write compiled results
    results_file = os.path.join(base_log_folder, 'compiled_results.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("WAND SCENARIO TEST - COMPILED RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total seeds tested: {len(SEEDS)}\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write("\n")

        f.write("Individual Results:\n")
        f.write("-"*80 + "\n")
        for r in results:
            f.write(f"Seed {r['seed']:6d}: ")
            f.write(f"Steps={r['steps']:3d}, ")
            f.write(f"Score={r['score']:3d}, ")
            f.write(f"Rewards={r['rewards']:6.2f}, ")
            f.write(f"Success={'✓' if not r['max_steps_reached'] else '✗'}\n")

        f.write("\n")
        f.write("Summary Statistics:\n")
        f.write("-"*80 + "\n")

        successful = [r for r in results if not r['max_steps_reached']]
        success_rate = len(successful) / len(results) * 100

        f.write(f"Success Rate: {len(successful)}/{len(results)} ({success_rate:.1f}%)\n")

        if successful:
            avg_steps = sum(r['steps'] for r in successful) / len(successful)
            avg_score = sum(r['score'] for r in successful) / len(successful)
            avg_rewards = sum(r['rewards'] for r in successful) / len(successful)

            f.write(f"Average Steps (successful): {avg_steps:.1f}\n")
            f.write(f"Average Score (successful): {avg_score:.1f}\n")
            f.write(f"Average Rewards (successful): {avg_rewards:.3f}\n")

        f.write("\n")
        f.write("="*80 + "\n")

    # Print summary to console
    print(f"\n{'='*80}")
    print("COMPILED RESULTS")
    print(f"{'='*80}")
    print(f"\nResults written to: {results_file}")
    print(f"\nSuccess Rate: {len(successful)}/{len(results)} ({success_rate:.1f}%)")

    if successful:
        print(f"Average Steps: {avg_steps:.1f}")
        print(f"Average Score: {avg_score:.1f}")

    print(f"\n{'='*80}\n")

    # Display the compiled results
    with open(results_file, 'r') as f:
        print(f.read())

if __name__ == '__main__':
    main()
