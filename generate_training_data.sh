#!/bin/bash
# ABOUTME: Mass training data generation script for ultra-elite transformer training
# ABOUTME: Runs thousands of games with best-performing strategies to collect sub-800 rejection games

set -e  # Exit on any error

# Configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_data_generation_${TIMESTAMP}.log"
RESULTS_DIR="training_data_results_${TIMESTAMP}"

echo "🚀 Starting Mass Training Data Generation - $(date)"
echo "📝 Logging to: $LOG_FILE"
echo "📊 Results will be collected in: $RESULTS_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Track start time
START_TIME=$(date +%s)

log "🎯 Starting mass training data generation"
log "Target: Generate 15,000+ games from top strategies"
log "Goal: Find 500+ games with <750 rejections"

echo "=================================="
echo "PHASE 1: RBCR (5000 games)"
echo "=================================="
log "Starting RBCR run - 5000 games"
START_RBCR=$(date +%s)
python main.py run --strategy rbcr --count 5000 --no-high-score-check --workers 20
END_RBCR=$(date +%s)
RBCR_TIME=$((END_RBCR - START_RBCR))
log "✅ RBCR completed in ${RBCR_TIME}s"

echo ""
echo "=================================="
echo "PHASE 2: RBCR2 (5000 games)"
echo "=================================="
log "Starting RBCR2 run - 5000 games"
START_RBCR2=$(date +%s)
python main.py run --strategy rbcr2 --count 5000 --no-high-score-check --workers 20
END_RBCR2=$(date +%s)
RBCR2_TIME=$((END_RBCR2 - START_RBCR2))
log "✅ RBCR2 completed in ${RBCR2_TIME}s"

echo ""
echo "=================================="
echo "PHASE 3: Ultimate3 (2000 games)"
echo "=================================="
log "Starting Ultimate3 run - 2000 games"
START_ULT3=$(date +%s)
python main.py run --strategy ultimate3 --count 2000 --no-high-score-check --workers 20
END_ULT3=$(date +%s)
ULT3_TIME=$((END_ULT3 - START_ULT3))
log "✅ Ultimate3 completed in ${ULT3_TIME}s"

echo ""
echo "=================================="
echo "PHASE 4: Perfect (2000 games)"
echo "=================================="
log "Starting Perfect run - 2000 games"
START_PERFECT=$(date +%s)
python main.py run --strategy perfect --count 2000 --no-high-score-check --workers 20
END_PERFECT=$(date +%s)
PERFECT_TIME=$((END_PERFECT - START_PERFECT))
log "✅ Perfect completed in ${PERFECT_TIME}s"

echo ""
echo "=================================="
echo "PHASE 5: Apex (2000 games)"
echo "=================================="
log "Starting Apex run - 2000 games"
START_APEX=$(date +%s)
python main.py run --strategy apex --count 2000 --no-high-score-check --workers 20
END_APEX=$(date +%s)
APEX_TIME=$((END_APEX - START_APEX))
log "✅ Apex completed in ${APEX_TIME}s"

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_TIME / 3600))
TOTAL_MINS=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo "🎉 =================================="
echo "    MASS GENERATION COMPLETE!"
echo "🎉 =================================="
log "🎉 All mass generation completed!"
log "⏱️  Total execution time: ${TOTAL_HOURS}h ${TOTAL_MINS}m (${TOTAL_TIME}s)"
log "📊 Phase timings:"
log "   - RBCR (5000):     ${RBCR_TIME}s"
log "   - RBCR2 (5000):    ${RBCR2_TIME}s"
log "   - Ultimate3 (2000): ${ULT3_TIME}s"
log "   - Perfect (2000):  ${PERFECT_TIME}s"
log "   - Apex (2000):     ${APEX_TIME}s"

echo ""
echo "📈 ANALYZING RESULTS..."

# Analyze results
log "🔍 Analyzing generated games..."

# Find best rejection counts
log "📊 Finding ultra-elite games (<750 rejections)..."
ULTRA_ELITE=$(grep -r '"rejected_count"' game_logs/game_*$(date +%Y%m%d)*.json | grep -E '"rejected_count": [1-7][0-9]{2},' | wc -l)
ELITE=$(grep -r '"rejected_count"' game_logs/game_*$(date +%Y%m%d)*.json | grep -E '"rejected_count": [78][0-9]{2},' | wc -l)

log "🏆 Results summary:"
log "   - Ultra-elite games (<750 rej): ${ULTRA_ELITE}"
log "   - Elite games (750-899 rej): ${ELITE}"
log "   - Total games generated: 16,000"

# Get best rejection counts
log "🥇 Top 10 best rejection counts:"
grep -r '"rejected_count"' game_logs/game_*$(date +%Y%m%d)*.json | cut -d: -f3 | cut -d, -f1 | sort -n | head -10 >> "$LOG_FILE"

# Create results summary
cat > "${RESULTS_DIR}/summary.txt" << EOF
Mass Training Data Generation Summary
=====================================
Date: $(date)
Total Execution Time: ${TOTAL_HOURS}h ${TOTAL_MINS}m

Games Generated:
- RBCR: 5,000 games (${RBCR_TIME}s)
- RBCR2: 5,000 games (${RBCR2_TIME}s)  
- Ultimate3: 2,000 games (${ULT3_TIME}s)
- Perfect: 2,000 games (${PERFECT_TIME}s)
- Apex: 2,000 games (${APEX_TIME}s)
- TOTAL: 16,000 games

Quality Metrics:
- Ultra-elite (<750 rej): ${ULTRA_ELITE} games
- Elite (750-899 rej): ${ELITE} games

Next Steps:
1. Filter games with ./filter_ultra_elite_games.py
2. Train new transformer on filtered data
3. Test improved transformer performance

Generated on: $(hostname)
Log file: ${LOG_FILE}
EOF

echo ""
echo "📋 Full summary saved to: ${RESULTS_DIR}/summary.txt"
echo "📝 Detailed log saved to: ${LOG_FILE}"
echo ""
echo "🎯 NEXT STEPS:"
echo "   1. Run: python filter_ultra_elite_games.py"
echo "   2. Retrain transformer with new ultra-elite data"
echo "   3. Test improved transformer performance"
echo ""
log "🏁 Mass training data generation script completed successfully!"