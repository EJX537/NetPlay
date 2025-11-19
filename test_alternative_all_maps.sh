#!/bin/bash

# Test alternative scenario with 5 seeds for all map modes: none, ascii, png

SEEDS=(12345 23456 34567 45678 56789)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOG_DIR="./runs/alternative_all_maps_${TIMESTAMP}"
MAP_RADIUS=20

mkdir -p "${BASE_LOG_DIR}"

echo "================================================================================"
echo "ALTERNATIVE SCENARIO - ALL MAP MODES TEST"
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
ALTERNATIVE SCENARIO TEST - ${MAP_MODE^^} MAP MODE
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
    des_file='scenarios/instructions/alternative.des',
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
agent.set_task('Drink from a fountain or drink a potion.')

steps = 0
max_steps = 200

try:
    for step in agent.run():
        steps += 1
        if steps >= max_steps:
            print(f'Reached {max_steps} step limit')
            break
except Exception as e:
    print(f'ERROR: {e}')

agent.close()
print(f'Total steps: {steps}')
PYEOF

        # Run the test
        python "${SEED_LOG_DIR}/test_run.py" > "${SEED_LOG_DIR}/output.log" 2>&1

        # Extract results
        STEPS=$(grep "Total steps:" "${SEED_LOG_DIR}/output.log" | tail -1 | awk '{print $3}')

        # Default values if parsing failed
        STEPS=${STEPS:-0}

        # Determine success by checking if finish_task was called in the logs
        if grep "finish_task" "${SEED_LOG_DIR}"/*/*.json 2>/dev/null | grep -v '"prompt"' | grep -q "finish_task"; then
            SUCCESS="✓"
            SUCCESS_TEXT="Task completed (finish_task called)"
        else
            if [ "$STEPS" -ge 200 ]; then
                SUCCESS="✗"
                SUCCESS_TEXT="Failed (timeout at 200 steps)"
            else
                SUCCESS="✗"
                SUCCESS_TEXT="Failed (no finish_task)"
            fi
        fi

        # Write to results file
        printf "Seed %6d: Steps=%3d, Success=%s (%s)\n" \
            "$seed" "$STEPS" "$SUCCESS" "$SUCCESS_TEXT" >> "${RESULTS_FILE}"

        # Also print to console
        printf "    Seed %6d: Steps=%3d, Success=%s\n" \
            "$seed" "$STEPS" "$SUCCESS"
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

    # Parse steps
    steps = 0
    for line in output.split('\n'):
        if 'Total steps:' in line:
            try:
                steps = int(line.split(':')[1].strip())
            except:
                pass

    # Check for finish_task in JSON logs
    import os
    import glob
    import subprocess
    success = False
    json_files = glob.glob(log_dir + "/*/*.json")
    for json_file in json_files:
        try:
            result = subprocess.run(
                ["grep", "finish_task", json_file],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'finish_task' in line and '"prompt"' not in line:
                        success = True
                        break
            if success:
                break
        except:
            pass

    results.append((seed, steps, success))

successes = [r for r in results if r[2]]
success_count = len(successes)
total_seeds = len(results)

print(f"Success Rate: {success_count}/{total_seeds} ({100.0 * success_count / total_seeds:.1f}%)")

if successes:
    avg_steps = sum(r[1] for r in successes) / success_count
    print(f"Average Steps (successful): {avg_steps:.1f}")
    print(f"Successful seeds: {', '.join(str(r[0]) for r in successes)}")

if success_count < total_seeds:
    failures = [r for r in results if not r[2]]
    print(f"Failed seeds: {', '.join(str(r[0]) for r in failures)}")
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
MAP MODE COMPARISON - ALTERNATIVE SCENARIO
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
