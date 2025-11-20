import cairosvg
import os

input_path = "/app/data/ai-stitching-model/panorama_m2.svg"
output_path = "/app/data/ai-stitching-model/panorama_m2.png"

print(f"Converting {input_path} to {output_path}...")
try:
    cairosvg.svg2png(url=input_path, write_to=output_path)
    if os.path.exists(output_path):
        print(f"SUCCESS: Created {output_path}")
    else:
        print("ERROR: Failed to create PNG")
except Exception as e:
    print(f"ERROR: {e}")
