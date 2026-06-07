#!/bin/bash
mkdir -p logs
BOXES=(A1 A4 2BB)
for BOX in "${BOXES[@]}"; do
    for SEED in $(seq 0 9); do
        LOG="logs/normal_${BOX}_seed${SEED}.log"
        echo "=== normal/${BOX}/seed${SEED} ==="
        python demo_collector.py --xml scene.xml \
            --box "$BOX" --scenario normal --seed "$SEED" \
            --max_steps 400 > "$LOG" 2>&1
    done
done
echo "DONE."
