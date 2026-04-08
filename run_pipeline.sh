#!/bin/bash
# Run the NZ Motor Vehicle Register data pipeline
# Usage: ./run_pipeline.sh [options]

set -e  # Exit on error

# Default values
YEARS=""
OUTPUT_DIR="reports"
SAVE_CLEANED="yes"
LOG_LEVEL="INFO"
INCLUDE_PRE1990="yes"

# Help message
show_help() {
    cat << EOF
NZ Vehicle Data Pipeline Runner

Usage: ./run_pipeline.sh [OPTIONS]

Options:
    -y, --years YEARS       Comma-separated years to process (e.g., "2020,2021,2022")
                           Omit to process all available years
    -o, --output DIR        Output directory for reports (default: reports)
    -s, --save PATH         Save cleaned data to this path (.parquet or .csv)
    --no-pre1990           Exclude vehicles from before 1990
    -l, --log-level LEVEL   Log level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    -h, --help             Show this help message

Examples:
    # Process all years
    ./run_pipeline.sh

    # Process specific years only
    ./run_pipeline.sh --years "2020,2021,2022,2023,2024"

    # Save cleaned data as Parquet
    ./run_pipeline.sh --save data/cleaned.parquet

    # Custom output with debug logging
    ./run_pipeline.sh --output reports/run_$(date +%Y%m%d) --log-level DEBUG

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--years)
            YEARS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--save)
            SAVE_CLEANED="$2"
            shift 2
            ;;
        --no-pre1990)
            INCLUDE_PRE1990="no"
            shift
            ;;
        -l|--log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build Python command
CMD="python3 src/pipeline.py"

# Add years if specified
if [ -n "$YEARS" ]; then
    # Convert comma-separated to space-separated
    YEARS_ARRAY=${YEARS//,/ }
    CMD="$CMD --years $YEARS_ARRAY"
fi

# Add output directory
CMD="$CMD --output-dir $OUTPUT_DIR"

# Add save cleaned if specified
if [ -n "$SAVE_CLEANED" ]; then
    CMD="$CMD --save-cleaned $SAVE_CLEANED"
fi

# Add no-pre1990 flag if set
if [ "$INCLUDE_PRE1990" = "no" ]; then
    CMD="$CMD --no-pre1990"
fi

# Add log level
CMD="$CMD --log-level $LOG_LEVEL"

# Print command and run
echo "Running pipeline..."
echo "Command: $CMD"
echo "─────────────────────────────────────────────────────────────"
echo

$CMD

echo
echo "─────────────────────────────────────────────────────────────"
echo "Pipeline complete! Reports available in: $OUTPUT_DIR"
