#!/bin/bash
#
# teleop_session.sh
# =================
#
# Batch teleoperation session for multiple demos.
#
# Scenario 1 / normal:
#   ./teleop_session.sh A1 0 9
#
# With one blocker:
#   ./teleop_session.sh A1 0 9 A4
#
# With multiple blockers:
#   ./teleop_session.sh A1 0 9 A4 2BB
#

TARGET=$1
SEED_START=$2
SEED_END=$3
shift 3
BLOCKERS="$@"

if [ -z "$TARGET" ] || [ -z "$SEED_START" ] || [ -z "$SEED_END" ]; then
    echo "Usage:"
    echo "  $0 <target_box> <seed_start> <seed_end> [blocker1 blocker2 ...]"
    echo
    echo "Examples:"
    echo "  $0 A1 0 9              # Scenario 1: no blockers"
    echo "  $0 A1 0 9 A4           # One blocker"
    echo "  $0 A1 0 9 A4 2BB       # Multiple blockers"
    exit 1
fi

echo "============================================================"
echo "TELEOPERATION SESSION"
echo "  Target box: $TARGET"
echo "  Seeds:      $SEED_START to $SEED_END"
echo "  Blockers:   ${BLOCKERS:-none}"
echo "============================================================"
echo

for SEED in $(seq $SEED_START $SEED_END); do
    DEMO_NAME="target_${TARGET}_seed_${SEED}_teleop"

    echo
    echo "----- Demo $((SEED - SEED_START + 1)) of $((SEED_END - SEED_START + 1)) -----"
    echo "Demo name: $DEMO_NAME"
    echo "Press Enter to start, or Ctrl+C to stop session..."
    read

    if [ -z "$BLOCKERS" ]; then
        python teleop_collector.py \
            --xml scene.xml \
            --target "$TARGET" \
            --jitter_seed "$SEED" \
            --name "$DEMO_NAME"
    else
        python teleop_collector.py \
            --xml scene.xml \
            --target "$TARGET" \
            --blockers $BLOCKERS \
            --jitter_seed "$SEED" \
            --name "$DEMO_NAME"
    fi

    echo
    echo "Demo $SEED complete. Take a short break."
done

echo
echo "============================================================"
echo "SESSION COMPLETE"
echo "============================================================"
ls -la demonstrations_teleop/target_${TARGET}_seed_*_teleop.pkl 2>/dev/null | wc -l
echo "demos saved in demonstrations_teleop/"
