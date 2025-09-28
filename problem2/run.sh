#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_papers.json> <output_dir> [epochs] [batch_size]"
    echo "Example: $0 ../problem1/sample_data/papers.json output/ 50 32"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_DIR="$2"  
EPOCHS="${3:-50}"
BATCH_SIZE="${4:-32}"

# Validate input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Training embeddings with the following settings:"
echo "  Input: $INPUT_FILE"
echo "  Output: $OUTPUT_DIR" 
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Run training container
docker run --rm \
    --name arxiv-embeddings \
    -v "$(realpath $INPUT_FILE)":/data/input/papers.json:ro \
    -v "$(realpath $OUTPUT_DIR)":/data/output \
    arxiv-embeddings:latest \
    /data/input/papers.json /data/output --epochs "$EPOCHS" --batch_size "$BATCH_SIZE"

echo ""
echo "Training complete. Output files:"
ls -la "$OUTPUT_DIR"