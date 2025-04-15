#!/bin/bash
# A shell script to convert an AVI video into BMP frames using ffmpeg.

# Check if an input file is provided.
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 input.avi [output_directory]"
  exit 1
fi

INPUT="$1"
# Use provided output directory or default to 'frames'
OUTPUT_DIR="${2:-frames}"

# Create output directory if it doesn't exist.
mkdir -p "$OUTPUT_DIR"

# Use ffmpeg to extract frames from the AVI and convert them to BMP images.
# The %04d in the output file name pads the frame number with zeros.
ffmpeg -i "$INPUT" -f "$OUTPUT_DIR/$INPUT/frame_%04d.bmp"

# Print a completion message.
echo "Conversion completed. BMP frames saved in '$OUTPUT_DIR'."

