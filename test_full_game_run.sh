#!/bin/bash

# Complete full game runs with all map modes

SEEDS=(12345)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOG_DIR="./runs/full_game_${TIMESTAMP}"
MAP_RADIUS=20

mkdir -p "${BASE_LOG_DIR}"

echo "================================================================================"
echo "FULL GAME RUN TEST - ALL MAP MODES"
echo "Timestamp: ${TIMESTAMP}"
echo "Log folder: ${BASE_LOG_DIR}"
echo "Seeds: ${SEEDS[*]}"
echo "Map radius: ${MAP_RADIUS}"
echo "================================================================================"
echo ""

# Test all map modes
MAP_MODES=("none" "ascii" "png")

for MAP_MODE in "${MAP_MODES[@]}"; do
    echo ""
    echo "================================================================================"
    echo "TESTING MAP MODE: ${MAP_MODE}"
    echo "================================================================================"
    echo ""

    MODE_DIR="${BASE_LOG_DIR}/${MAP_MODE}_map"
    mkdir -p "${MODE_DIR}"

    RESULTS_FILE="${MODE_DIR}/results.txt"

    # Write header to results file
    cat > "${RESULTS_FILE}" << EOF
================================================================================
FULL GAME RUN TEST - ${MAP_MODE^^} MAP MODE
================================================================================
Timestamp: ${TIMESTAMP}
Map mode: ${MAP_MODE}
Map radius: ${MAP_RADIUS}
Total seeds tested: ${#SEEDS[@]}
Seeds: ${SEEDS[*]}

Individual Results:
--------------------------------------------------------------------------------
EOF

    # Run tests for each seed
    for seed in "${SEEDS[@]}"; do
        echo "  Testing seed ${seed} with ${MAP_MODE} map..."

        SEED_LOG_DIR="${MODE_DIR}/seed_${seed}"
        mkdir -p "${SEED_LOG_DIR}"

        # Create test script for this seed and map mode
        cat > "${SEED_LOG_DIR}/test_run.py" << PYEOF
import os
os.environ['DEBUG_LLM_RESPONSES'] = 'false'

from netplay import create_llm_agent
from netplay.nethack_utils.nle_wrapper import NethackGymnasiumWrapper
from netplay.llm_wrapper import LiteLLMWrapper
from netplay.logging.nethack_monitor import NethackH5PYMonitor

llm = LiteLLMWrapper(model='gemini/gemini-2.5-flash', temperature=0.0, max_tokens=2048)

env = NethackGymnasiumWrapper(
    render_mode='rgb_array',
    autopickup=False
)

log_folder = '${SEED_LOG_DIR}'
env = NethackH5PYMonitor(env, os.path.join(log_folder, "trajectories.h5py"))
env.reset(seed=${seed})

agent = create_llm_agent(
    env=env,
    llm=llm,
    memory_tokens=500,
    log_folder=log_folder,
    render=False,
    map_mode='${MAP_MODE}',
    map_radius=${MAP_RADIUS}
)

agent.init()

steps = 0
game_ended = False
final_score = 0
deaths = 0

try:
    for step in agent.run():
        steps += 1

        # Check if game ended (step.step_data.done indicates game over)
        if step.step_data:
            if step.step_data.done:
                deaths += 1
                if not game_ended:  # Only capture first death
                    game_ended = True
                    # Get score from observation before agent resets
                    final_score = step.step_data.observation['blstats'][9]  # Score is at index 9
                    print(f'Game ended at step {steps}, score: {final_score}')
                    break

        # No step limit - run until game naturally ends
except Exception as e:
    print(f'ERROR: {e}')
    final_score = agent.blstats.score

agent.close()
print(f'Total steps: {steps}')
print(f'Total deaths: {deaths}')
print(f'Score: {final_score}')
print(f'Game ended naturally: {game_ended}')
PYEOF

        # Run the test
        python "${SEED_LOG_DIR}/test_run.py" > "${SEED_LOG_DIR}/output.log" 2>&1

        # Extract results
        STEPS=$(grep "^Total steps:" "${SEED_LOG_DIR}/output.log" | tail -1 | awk '{print $3}')
        SCORE=$(grep "^Score:" "${SEED_LOG_DIR}/output.log" | tail -1 | awk '{print $2}')
        ENDED=$(grep "^Game ended naturally:" "${SEED_LOG_DIR}/output.log" | tail -1 | awk '{print $4}')

        # Default values if parsing failed
        STEPS=${STEPS:-0}
        SCORE=${SCORE:-0}
        ENDED=${ENDED:-False}

        # Write to results file
        printf "Seed %6d: Steps=%5d, Score=%6d, Ended=%s\n" \
            "$seed" "$STEPS" "$SCORE" "$ENDED" >> "${RESULTS_FILE}"

        # Also print to console
        printf "    Seed %6d: Steps=%5d, Score=%6d, Ended=%s\n" \
            "$seed" "$STEPS" "$SCORE" "$ENDED"
    done

    # Calculate summary statistics
    echo "" >> "${RESULTS_FILE}"
    echo "Summary Statistics:" >> "${RESULTS_FILE}"
    echo "--------------------------------------------------------------------------------" >> "${RESULTS_FILE}"

    # Count successes using Python for better calculation
    python3 << PYEOF >> "${RESULTS_FILE}"
import sys
results = []
seeds = [int(x) for x in "${SEEDS[*]}".split()]

for seed in seeds:
    log_dir = "${MODE_DIR}/seed_" + str(seed)

    # Read output
    try:
        with open(log_dir + "/output.log", "r") as f:
            output = f.read()
    except:
        output = ""

    # Parse values
    steps = 0
    score = 0
    ended = False

    for line in output.split('\n'):
        if line.startswith('Total steps:'):
            try:
                steps = int(line.split(':')[1].strip())
            except:
                pass
        elif line.startswith('Score:'):
            try:
                score = int(line.split(':')[1].strip())
            except:
                pass
        elif line.startswith('Game ended naturally:'):
            ended = line.split(':')[1].strip() == 'True'

    results.append((seed, steps, score, ended))

total_seeds = len(results)
ended_count = sum(1 for r in results if r[3])
avg_steps = sum(r[1] for r in results) / total_seeds if total_seeds > 0 else 0
avg_score = sum(r[2] for r in results) / total_seeds if total_seeds > 0 else 0
max_score = max(r[2] for r in results) if results else 0
max_steps = max(r[1] for r in results) if results else 0

print(f"Games ended naturally: {ended_count}/{total_seeds} ({100.0 * ended_count / total_seeds:.1f}%)")
print(f"Average steps: {avg_steps:.1f}")
print(f"Average score: {avg_score:.1f}")
print(f"Max score: {max_score}")
print(f"Max steps: {max_steps}")

if max_score > 0:
    best_seed = max(results, key=lambda r: r[2])[0]
    print(f"Best performing seed: {best_seed}")
PYEOF

    echo "" >> "${RESULTS_FILE}"
    echo "================================================================================" >> "${RESULTS_FILE}"

    echo ""
    echo "  ${MAP_MODE^^} MAP MODE COMPLETE"
    echo ""
done

# Generate comparison report
COMPARISON_FILE="${BASE_LOG_DIR}/comparison.txt"

cat > "${COMPARISON_FILE}" << EOF
================================================================================
MAP MODE COMPARISON - FULL GAME RUNS
================================================================================
Timestamp: ${TIMESTAMP}
Seeds tested: ${SEEDS[*]}
Map radius: ${MAP_RADIUS}

EOF

for MAP_MODE in "${MAP_MODES[@]}"; do
    echo "${MAP_MODE^^} MAP MODE RESULTS:" >> "${COMPARISON_FILE}"
    echo "--------------------------------------------------------------------------------" >> "${COMPARISON_FILE}"
    grep -A 20 "Summary Statistics:" "${BASE_LOG_DIR}/${MAP_MODE}_map/results.txt" | tail -n +2 >> "${COMPARISON_FILE}"
    echo "" >> "${COMPARISON_FILE}"
done

echo "================================================================================" >> "${COMPARISON_FILE}"

# Print final summary
echo ""
echo "================================================================================"
echo "COMPARISON COMPLETE"
echo "================================================================================"
echo ""
cat "${COMPARISON_FILE}"
echo ""
echo "Full results saved to: ${BASE_LOG_DIR}"
for MAP_MODE in "${MAP_MODES[@]}"; do
    echo "  ${MAP_MODE} map: ${BASE_LOG_DIR}/${MAP_MODE}_map/results.txt"
done
echo "  Comparison: ${COMPARISON_FILE}"
echo ""
