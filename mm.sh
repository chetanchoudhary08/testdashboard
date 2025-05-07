#!/bin/bash

# Mouse mover script for Mac
# Moves the mouse 5 pixels right and left alternately every 5 minutes

echo "Starting mouse mover script. Press Ctrl+C to exit."
echo "Mouse will move slightly every 5 minutes."

# Check if cliclick is installed
if ! command -v cliclick &> /dev/null; then
    echo "This script requires cliclick. Installing with Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is required. Please install Homebrew first."
        echo "Visit https://brew.sh/ for instructions."
        exit 1
    fi
    brew install cliclick
fi

# Direction flag (0 = right, 1 = left)
direction=0

try_interrupt() {
    echo "Script terminated by user."
    exit 0
}

# Set up trap to catch Ctrl+C
trap try_interrupt INT

# Main loop
while true; do
    # Get current mouse position
    current_position=$(cliclick p)
    x=$(echo $current_position | cut -d "," -f 1)
    y=$(echo $current_position | cut -d "," -f 2)
    
    # Move mouse based on direction
    if [ $direction -eq 0 ]; then
        # Move right
        cliclick m:$((x+5)),$y
        direction=1
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Moved right to $((x+5)),$y"
    else
        # Move left
        cliclick m:$((x-5)),$y
        direction=0
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Moved left to $((x-5)),$y"
    fi
    
    # Wait for 5 minutes (300 seconds)
    sleep 300
done
