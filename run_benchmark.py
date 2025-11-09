#!/usr/bin/env python3
"""
Benchmark script for testing NetPlay agents with different configurations.

This script extends run_full_runs.py to support:
- Multiple LLM models (via LiteLLM)
- Different max_tokens settings
- ASCII map integration (future feature)
- Systematic comparison of configurations

Example usage:
    # Run baseline unguided agent with GPT-4
    python run_benchmark.py -config baseline -num_runs 5

    # Run with Gemini
    python run_benchmark.py -config gemini_baseline -num_runs 5

    # Run with custom model
    python run_benchmark.py -model "claude-3-5-sonnet-20241022" -num_runs 3

    # Run all predefined configs
    python run_benchmark.py -config all -num_runs 10
"""

import numpy as np
import sys
import os
import subprocess
import argparse
from warnings import warn
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

ALLOWED_ROLES = [
    "archeologist",
    "barbarian",
    "cavewoman",
    "caveman",
    "healer",
    "knight",
    "monk",
    "priest",
    "priestess",
    "ranger",
    "rogue",
    "samurai",
    "tourist",
    "valkyrie",
    "wizard"
]

# Task templates
UNGUIDED_TASK = "Win the game."

GUIDED_TASK = """Your task is to win the game. Tips:
- Stay on the first 2 levels (depth=1/2, dungeon number=0) until you reached level 8.
- You can kick open locked doors, but that usually requires multiple tries. Note WHAMM means its still closed.
- When your health gets low, run away before healing.
- Corpses in your inventory will rot, do not eat them. Only eat freshly killed corpses. Keep eating corpses until satiated, but do not keep eating.
- Do not eat kobolds, they are poisonous.
- Only pray every 500 turns to fix bad situations, for example low health or when you are fainting.
- Use the keys "ykulnjbh" for specifying the directions "n ne e se s sw w nw". For example press the key "y" to indicate north.
- The skill "type_text" can be used to press multiple alphanumeric keys, use it to navigate menus faster.
- To move onto a tile occupied by a friendly monster stand next to it and press "m". Use this for shopkeepers.
- Dip long swords into fountains as a lawful character at level 5 or higher to get excalibur.
- Sacrifice corpses at altars for a chance to get powerful artifacts.
- You will timeout if you do not make progress for a while. So if something doesn't work on the first try, shift your focus onto something else."""

CREATIVE_TASK = """You are a creative agent, that is interested in exploring the game as much as possible.
Your goals are to find interesting interactions and to discover new areas to explore.
You despise dying, as it will reset your progress.
Here are some survival tips:
- You can kick open locked doors, but that usually requires multiple tries. Note WHAMM means its still closed.
- When your health gets low, run away before healing.
- Corpses in your inventory will rot, do not eat them. Only eat freshly killed corpses. Keep eating corpses until satiated, but avoid overeating.
- Do not eat kobolds, they are poisonous.
- Only pray every 500 turns to fix bad situations, for example low health or when you are fainting.
- The keys "ykulnjbh" correspond to the directions "n ne e se s sw w nw".
- The skill "type_text" can be used to press multiple alphanumeric keys, use it to navigate menus faster.
- To move onto a tile occupied by a friendly monster stand next to it and press "m". Use this for shopkeepers.
- You will timeout if you do not make progress for a while. So if something doesn't work on the first try, shift your focus onto something else.
"""

RUN_PY = "./run.py"
LOG_FOLDER = "./runs/benchmark"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    model: str
    task: str
    max_tokens: int = 2048
    max_memory_tokens: int = 500
    censor_nethack: bool = False
    update_hidden_objects: bool = True
    description: str = ""


# Predefined configurations for systematic testing
BENCHMARK_CONFIGS = {
    # === Baseline Configurations (using Gemini 2.5 Flash - latest and best) ===
    "baseline": BenchmarkConfig(
        name="baseline_gemini_2_5_flash",
        model="gemini/gemini-2.5-flash",
        task=UNGUIDED_TASK,
        max_tokens=2048,
        description="Baseline unguided agent with Gemini 2.5 Flash (latest, best)"
    ),

    "baseline_guided": BenchmarkConfig(
        name="baseline_gemini_2_5_flash_guided",
        model="gemini/gemini-2.5-flash",
        task=GUIDED_TASK,
        max_tokens=2048,
        description="Guided agent with Gemini 2.5 Flash (latest, best) and strategy tips"
    ),

    # === Gemini Variants (Primary models) ===
    "gemini_pro": BenchmarkConfig(
        name="gemini_2_5_pro_unguided",
        model="gemini/gemini-2.5-pro",
        task=UNGUIDED_TASK,
        max_tokens=8192,
        description="Baseline with Gemini 2.5 Pro (most capable)"
    ),

    "gemini_pro_guided": BenchmarkConfig(
        name="gemini_2_5_pro_guided",
        model="gemini/gemini-2.5-pro",
        task=GUIDED_TASK,
        max_tokens=8192,
        description="Guided agent with Gemini 2.5 Pro (most capable)"
    ),

    "gemini_1_5": BenchmarkConfig(
        name="gemini_1_5_flash_unguided",
        model="gemini/gemini-1.5-flash",
        task=UNGUIDED_TASK,
        max_tokens=2048,
        description="Baseline with Gemini 1.5 Flash (older version for comparison)"
    ),

    # === GPT-4 Variants (requires OpenAI API credits) ===
    "gpt4": BenchmarkConfig(
        name="gpt4_unguided",
        model="gpt-4-turbo-preview",
        task=UNGUIDED_TASK,
        max_tokens=4096,
        description="Unguided agent with full GPT-4 Turbo (requires OpenAI credits)"
    ),

    "gpt4_guided": BenchmarkConfig(
        name="gpt4_guided",
        model="gpt-4-turbo-preview",
        task=GUIDED_TASK,
        max_tokens=4096,
        description="Guided agent with full GPT-4 Turbo (requires OpenAI credits)"
    ),

    "gpt4o_mini": BenchmarkConfig(
        name="gpt4o_mini_unguided",
        model="gpt-4o-mini",
        task=UNGUIDED_TASK,
        max_tokens=2048,
        description="Baseline with GPT-4o-mini (requires OpenAI credits)"
    ),

    # === Gemini Experimental (2.0 Flash) ===
    "gemini_baseline": BenchmarkConfig(
        name="gemini_2_0_flash_unguided",
        model="gemini/gemini-2.0-flash-exp",
        task=UNGUIDED_TASK,
        max_tokens=2048,
        description="Baseline with Gemini 2.0 Flash experimental (may be unstable)"
    ),

    "gemini_guided": BenchmarkConfig(
        name="gemini_2_0_flash_guided",
        model="gemini/gemini-2.0-flash-exp",
        task=GUIDED_TASK,
        max_tokens=2048,
        description="Guided agent with Gemini 2.0 Flash experimental"
    ),

    # === Claude Variants (requires Anthropic API credits) ===
    "claude_baseline": BenchmarkConfig(
        name="claude_sonnet_unguided",
        model="claude-3-5-sonnet-20241022",
        task=UNGUIDED_TASK,
        max_tokens=4096,
        description="Baseline with Claude 3.5 Sonnet (requires Anthropic credits)"
    ),

    "claude_guided": BenchmarkConfig(
        name="claude_sonnet_guided",
        model="claude-3-5-sonnet-20241022",
        task=GUIDED_TASK,
        max_tokens=4096,
        description="Guided agent with Claude 3.5 Sonnet (requires Anthropic credits)"
    ),

    # === Token Limit Experiments (using Gemini 2.5 Flash) ===
    "tokens_1k": BenchmarkConfig(
        name="gemini_2_5_flash_1k_tokens",
        model="gemini/gemini-2.5-flash",
        task=UNGUIDED_TASK,
        max_tokens=1024,
        description="Low token limit (1024) test with Gemini 2.5 Flash"
    ),

    "tokens_4k": BenchmarkConfig(
        name="gemini_2_5_flash_4k_tokens",
        model="gemini/gemini-2.5-flash",
        task=UNGUIDED_TASK,
        max_tokens=4096,
        description="High token limit (4096) test with Gemini 2.5 Flash"
    ),

    # === Memory Experiments (using Gemini 2.5 Flash) ===
    "memory_low": BenchmarkConfig(
        name="gemini_2_5_flash_memory_250",
        model="gemini/gemini-2.5-flash",
        task=UNGUIDED_TASK,
        max_memory_tokens=250,
        description="Low memory (250 tokens) test with Gemini 2.5 Flash"
    ),

    "memory_high": BenchmarkConfig(
        name="gemini_2_5_flash_memory_1000",
        model="gemini/gemini-2.5-flash",
        task=UNGUIDED_TASK,
        max_memory_tokens=1000,
        description="High memory (1000 tokens) test with Gemini 2.5 Flash"
    ),
}


def get_final_score(log_folder: str) -> Optional[int]:
    """Extract the final score from trajectories.h5py file.

    Args:
        log_folder: Path to the log folder containing trajectories.h5py

    Returns:
        Final score as integer, or None if not available
    """
    if not HAS_H5PY:
        return None

    trajectory_file = os.path.join(log_folder, "trajectories.h5py")
    if not os.path.exists(trajectory_file):
        return None

    try:
        # Import BLStats helper if available
        try:
            from netplay.nethack_agent.tracking import make_blstats
        except ImportError:
            from collections import namedtuple
            BLStats = namedtuple('BLStats', 'x y strength_percentage strength dexterity constitution intelligence wisdom charisma score hitpoints max_hitpoints depth gold energy max_energy armor_class monster_level experience_level experience_points time hunger_state carrying_capacity dungeon_number level_number prop_mask alignment')
            def make_blstats(seq):
                s = list(seq)
                expected = len(BLStats._fields)
                if len(s) == expected:
                    return BLStats(*s)
                if len(s) == expected - 1:
                    return BLStats(*s, 0)
                s = (s + [0] * expected)[:expected]
                return BLStats(*s)

        with h5py.File(trajectory_file, 'r') as f:
            if 'trajectories' not in f:
                return None

            trajs = f['trajectories']
            ids = sorted([int(k) for k in trajs.keys()])
            if not ids:
                return None

            tid = str(ids[-1])
            obs_grp = trajs[tid]['observations']

            if 'blstats' not in obs_grp:
                return None

            bl = obs_grp['blstats'][:]
            if bl.shape[0] == 0:
                return None

            # Find last non-zero row
            nonzero_rows = np.where(np.any(bl != 0, axis=1))[0]
            if len(nonzero_rows) > 0:
                last_idx = int(nonzero_rows[-1])
            else:
                last_idx = bl.shape[0] - 1

            last_bl = bl[last_idx]
            bl_named = make_blstats(last_bl)

            # Extract score
            score = getattr(bl_named, 'score', None)
            if score is None:
                try:
                    score = int(last_bl[9])
                except Exception:
                    return None

            return int(score) if score is not None else None

    except Exception as e:
        # Silently fail - score is optional information
        return None


def run_single_benchmark(
    config: BenchmarkConfig,
    seed: int,
    role: str,
    render: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    run_directly: bool = False
) -> bool:
    """Run a single benchmark with the given configuration.

    Args:
        run_directly: If True, run the agent in-process instead of via subprocess
                     (shows live output like interactive mode)

    Returns:
        True if successful, False if failed or skipped
    """
    log_folder = os.path.join(LOG_FOLDER, config.name, role, str(seed))

    if os.path.exists(log_folder):
        print(f"Skipping {config.name} (seed {seed}): folder exists")
        return False

    print(f"Running {config.name} with seed {seed}")
    print(f"  Model: {config.model}")
    print(f"  Task: {config.task[:50]}...")
    print(f"  Max tokens: {config.max_tokens}")

    if dry_run:
        print("  [DRY RUN - not executing]")
        return True

    try:
        os.makedirs(log_folder, exist_ok=True)

        # Save config metadata
        import json
        with open(os.path.join(log_folder, "benchmark_config.json"), "w") as f:
            json.dump({
                "config_name": config.name,
                "model": config.model,
                "max_tokens": config.max_tokens,
                "max_memory_tokens": config.max_memory_tokens,
                "task": config.task,
                "seed": seed,
                "role": role,
                "description": config.description
            }, f, indent=2)

        # Run directly in-process if requested (like interactive mode)
        if run_directly:
            print(f"\n{'='*60}")
            print(f"Starting run with seed {seed}...")
            print(f"{'='*60}\n")

            # Import and run directly
            from netplay import create_llm_agent
            from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
            from netplay.llm_wrapper import LiteLLMWrapper

            llm = LiteLLMWrapper(
                model=config.model,
                temperature=0.0,
                max_tokens=config.max_tokens
            )

            env = NethackGymnasiumWrapper(
                render_mode='human' if render else 'rgb_array',
                autopickup=False,
                character=role
            )

            # Wrap with monitor
            from netplay.logging.nethack_monitor import NethackH5PYMonitor
            env = NethackH5PYMonitor(env, os.path.join(log_folder, "trajectories.h5py"))

            # Set seed
            env.reset(seed=seed)

            agent = create_llm_agent(
                env=env,
                llm=llm,
                memory_tokens=config.max_memory_tokens,
                log_folder=log_folder,
                render=render,
                censor_nethack_context=config.censor_nethack,
                update_hidden_objects=config.update_hidden_objects,
                enable_finish_task_skill=False
            )

            # Don't need to set character/seed again - already done above
            agent.init()
            agent.set_task(config.task)

            print(f"Using llm agent.")
            print(f"Logging in '{log_folder}'.")
            print(f"Initializing LiteLLM with model: {config.model}")
            print(f"Agent is playing as a {role}.")

            try:
                for step in agent.run():
                    if step.thoughts:
                        print(f"Thinking: {step.thoughts}")
                    if step.executed_action():
                        print(f"Executed action '{step.step_data.action.name}'.")
            except KeyboardInterrupt:
                print("\nRun interrupted by user")
            finally:
                agent.close()
                print("Agent is done.\n")

            # Display final score
            score = get_final_score(log_folder)
            if score is not None:
                print(f"\n{'='*60}")
                print(f"FINAL SCORE: {score}")
                print(f"{'='*60}\n")

            return True

        # Otherwise run via subprocess
        process_args = [
            "llm",
            "-task", config.task,
            "-log_folder", log_folder,
            "-seed", str(seed),
            "-character", role,
            "-model", config.model,
            "-max_tokens", str(config.max_tokens),
            "-max_memory_tokens", str(config.max_memory_tokens),
            "--keep_log_folder",
            "--disable_finish_task_skill"
        ]

        if config.censor_nethack:
            process_args.append("--censor_nethack_context")
        if config.update_hidden_objects:
            process_args.append("--update_hidden_objects")
        if render:
            process_args.append("--render")

        # Run the benchmark
        if verbose:
            # Show output in terminal AND save to file (live streaming)
            print(f"\n{'='*60}")
            print(f"Starting run with seed {seed}...")
            print(f"{'='*60}\n")

            with open(os.path.join(log_folder, "out.txt"), "w") as out_file:
                process = subprocess.Popen(
                    ["python", RUN_PY, *process_args],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1  # Line buffered
                )

                # Stream output line by line to both terminal and file
                for line in process.stdout:
                    print(line, end='')  # Print to console
                    out_file.write(line)  # Write to file

                # Wait for process to complete
                returncode = process.wait()
        else:
            # Only save to file (silent mode)
            with open(os.path.join(log_folder, "out.txt"), "w") as out_file:
                result = subprocess.run(
                    ["python", RUN_PY, *process_args],
                    stdout=out_file,
                    stderr=subprocess.STDOUT
                )
                returncode = result.returncode

        if returncode != 0:
            warn(f"Benchmark {config.name} with seed {seed} failed with code {returncode}")
            return False

        # Display final score
        score = get_final_score(log_folder)
        if score is not None:
            print(f"\n{'='*60}")
            print(f"FINAL SCORE: {score}")
            print(f"{'='*60}\n")

        return True
    except Exception as e:
        warn(f"Benchmark {config.name} with seed {seed} failed:\n{e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        prog='NetPlay Benchmark Runner',
        description='Run systematic benchmarks of NetPlay with different LLM configurations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py -config baseline -num_runs 5
  python run_benchmark.py -config gemini_pro -seeds 1 2 3
  python run_benchmark.py -config all -num_runs 2 --dry-run
  python run_benchmark.py -model "gemini/gemini-pro" -task "Win the game." -num_runs 3
"""
    )

    # Configuration selection
    parser.add_argument(
        '-config',
        type=str,
        default=None,
        help="Use a predefined configuration (see list below)"
    )

    # Manual configuration
    parser.add_argument('-model', type=str, help="LLM model to use (overrides config)")
    parser.add_argument('-task', type=str, help="Task text (overrides config)")
    parser.add_argument('-max_tokens', type=int, help="Max tokens (overrides config)")
    parser.add_argument('-max_memory_tokens', type=int, help="Memory tokens (overrides config)")

    # Run parameters
    parser.add_argument('-seeds', nargs="+", type=int, default=None,
                       help="List of seeds for runs")
    parser.add_argument('-num_runs', type=int, default=5,
                       help="Number of runs (ignored if -seeds is set)")
    parser.add_argument('-role', type=str, default="valkyrie", choices=ALLOWED_ROLES,
                       help="Character role")

    # Flags
    parser.add_argument('--render', action='store_true',
                       help="Render each run in a window")
    parser.add_argument('--verbose', action='store_true',
                       help="Show live output in terminal (default: only save to log files)")
    parser.add_argument('--dry-run', action='store_true',
                       help="Print what would be run without executing")
    parser.add_argument('--list-configs', action='store_true',
                       help="List all available configs and exit")

    args = parser.parse_args()

    # List configs if requested
    if args.list_configs:
        print("\nAvailable Benchmark Configurations:")
        print("=" * 80)
        for name, config in sorted(BENCHMARK_CONFIGS.items()):
            print(f"\n{name}:")
            print(f"  Name: {config.name}")
            print(f"  Model: {config.model}")
            print(f"  Max Tokens: {config.max_tokens}")
            print(f"  Memory Tokens: {config.max_memory_tokens}")
            print(f"  Description: {config.description}")
            print(f"  Task: {config.task[:60]}...")
        print("\n" + "=" * 80)
        return

    # Generate or use provided seeds
    if args.seeds:
        print(f"Using provided seeds: {args.seeds}")
        seeds = args.seeds
        num_runs = len(seeds)
    else:
        num_runs = args.num_runs
        seeds = np.random.randint(1000000, size=num_runs).tolist()
        print(f"Generated {num_runs} random seeds: {seeds}")

    # Determine which configurations to run
    configs_to_run: List[BenchmarkConfig] = []

    if args.config:
        if args.config == "all":
            configs_to_run = list(BENCHMARK_CONFIGS.values())
            print(f"\nRunning ALL {len(configs_to_run)} configurations!")
            if not args.dry_run:
                response = input("This will take a long time. Continue? (yes/no): ")
                if response.lower() not in ["yes", "y"]:
                    print("Cancelled.")
                    return
        elif args.config in BENCHMARK_CONFIGS:
            config = BENCHMARK_CONFIGS[args.config]

            # Apply overrides
            if args.model:
                config.model = args.model
            if args.task:
                config.task = args.task
            if args.max_tokens:
                config.max_tokens = args.max_tokens
            if args.max_memory_tokens:
                config.max_memory_tokens = args.max_memory_tokens

            configs_to_run = [config]
        else:
            print(f"Error: Unknown config '{args.config}'")
            print(f"Available configs: {', '.join(BENCHMARK_CONFIGS.keys())}, all")
            print("Use --list-configs to see details")
            return
    else:
        # Create custom config from command line args
        if not args.model or not args.task:
            print("Error: Must specify either -config OR both -model and -task")
            print("Use --list-configs to see available predefined configs")
            return

        custom_config = BenchmarkConfig(
            name=f"custom_{args.model.replace('/', '_')}",
            model=args.model,
            task=args.task,
            max_tokens=args.max_tokens or 2048,
            max_memory_tokens=args.max_memory_tokens or 500,
            description="Custom configuration from command line"
        )
        configs_to_run = [custom_config]

    # Run benchmarks
    total_benchmark_runs = len(configs_to_run) * len(seeds)
    run_directly = (total_benchmark_runs == 1)  # Run in-process if only 1 run

    print(f"\n{'=' * 80}")
    print(f"BENCHMARK RUN SUMMARY")
    print(f"{'=' * 80}")
    print(f"Configurations: {len(configs_to_run)}")
    print(f"Seeds per config: {len(seeds)}")
    print(f"Role: {args.role}")
    print(f"Total runs: {total_benchmark_runs}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'=' * 80}\n")

    total_runs = 0
    successful_runs = 0

    for config in configs_to_run:
        print(f"\n{'=' * 80}")
        print(f"Running configuration: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'=' * 80}\n")

        for seed in seeds:
            total_runs += 1
            success = run_single_benchmark(
                config=config,
                seed=seed,
                role=args.role,
                render=args.render,
                dry_run=args.dry_run,
                verbose=args.verbose,
                run_directly=run_directly
            )
            if success:
                successful_runs += 1
            print()  # Blank line between runs

    # Summary
    print(f"\n{'=' * 80}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed/Skipped: {total_runs - successful_runs}")
    print(f"Results stored in: {LOG_FOLDER}")
    print(f"{'=' * 80}\n")

    if not args.dry_run:
        print("To analyze results, use experiments/evaluate_runs.ipynb")
        print(f"Set runs_folder to './runs/benchmark/<config_name>/{args.role}'")


if __name__ == "__main__":
    main()
