#!/bin/bash

# Mouse mover script for Mac
# Moves the mouse 5 pixels right and left alternately every 5 minutes
# Uses built-in AppleScript - no external dependencies

echo "Starting mouse mover script. Press Ctrl+C to exit."
echo "Mouse will move slightly every 5 minutes."

# Direction flag (0 = right, 1 = left)
direction=0

try_interrupt() {
    echo "Script terminated by user."
    exit 0
}

# Set up trap to catch Ctrl+C
trap try_interrupt INT

# Function to get mouse position using AppleScript
get_mouse_position() {
    local position=$(osascript -e 'tell application "System Events" to return "{" & (mouse location)\'s item 1 & "," & (mouse location)\'s item 2 & "}"')
    echo "$position"
}

# Function to move mouse using AppleScript
move_mouse() {
    local x=$1
    local y=$2
    osascript -e "tell application \"System Events\" to set mouse location to {$x, $y}"
}

# Main loop
while true; do
    # Get current mouse position
    current_position=$(get_mouse_position)
    # Extract x and y coordinates, removing the curly braces
    current_position=${current_position//[{}]/}
    x=$(echo $current_position | cut -d "," -f 1)
    y=$(echo $current_position | cut -d "," -f 2)
    
    # Move mouse based on direction
    if [ $direction -eq 0 ]; then
        # Move right
        move_mouse $((x+5)) $y
        direction=1
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Moved right to $((x+5)),$y"
    else
        # Move left
        move_mouse $((x-5)) $y
        direction=0
        echo "$(date +"%Y-%m-%d %H:%M:%S") - Moved left to $((x-5)),$y"
    fi
    
    # Wait for 5 minutes (300 seconds)
    sleep 300
done
