from svg.path import parse_path, Line, CubicBezier, Move
import numpy as np

d = "M 10 10 l 10 0 c 0,10 10,10 10,0"
path = parse_path(d)
print(f"Original: {path}")

offset = 100 + 100j

new_d_parts = []
# svg.path 4.x behavior might be relevant
# Iterating path gives segments
for segment in path:
    # Shift start/end
    segment.start += offset
    segment.end += offset
    if hasattr(segment, 'control1'):
        segment.control1 += offset
    if hasattr(segment, 'control2'):
        segment.control2 += offset
    # arc?
    
    # Re-serialize?
    # Does segment have .d() ? or similar?
    print(f"Segment: {type(segment)} {segment}")
    
# Check if we can reconstruction
