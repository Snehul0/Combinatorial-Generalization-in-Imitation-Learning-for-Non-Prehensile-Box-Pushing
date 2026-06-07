#!/bin/bash
for SEED in $(seq 0 9); do
    LOG="logs/normal_A1_seed${SEED}.log"
    echo "=== normal/A1/seed${SEED} ==="
    python demo_collector.py --xml scene.xml \
        --box A1 --scenario normal --seed "$SEED" \
        --max_steps 400 > "$LOG" 2>&1
done
echo "DONE."
