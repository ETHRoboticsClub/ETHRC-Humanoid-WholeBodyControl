import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Path to your parquet file
parquet_path = "outputs/2026-05-27-11-08-12-G1-sim/data/chunk-000/episode_000000.parquet"

# Load the parquet file
df = pd.read_parquet(parquet_path)

# Extract 2D positions (x, y) from the relevant columns
# observation.robot_base_pose: [x, y, z, qx, qy, qz, qw] -> take first 2
# observation.obj_pos: [x, y, z] -> take first 2
robot_positions_2d = np.stack(df["observation.robot_base_pose"].apply(lambda x: x[:2]).values)
bottle_positions_2d = np.stack(df["observation.obj_pos"].apply(lambda x: x[:2]).values)

# Create the plot
plt.figure(figsize=(10, 8))

# Plot robot trajectory
plt.plot(robot_positions_2d[:, 0], robot_positions_2d[:, 1], 
         'b-', label='Robot Base Trajectory', linewidth=2, alpha=0.7)

# Plot bottle trajectory (if it moves) or position
plt.plot(bottle_positions_2d[:, 0], bottle_positions_2d[:, 1], 
         'r-', label='Bottle Position', linewidth=2, alpha=0.7)

# Mark start positions with larger markers
plt.scatter(robot_positions_2d[0, 0], robot_positions_2d[0, 1], 
            c='blue', s=100, edgecolors='white', zorder=5, label='Robot Start')
plt.scatter(bottle_positions_2d[0, 0], bottle_positions_2d[0, 1], 
            c='red', s=100, edgecolors='white', zorder=5, label='Bottle Start')

# Mark end positions
plt.scatter(robot_positions_2d[-1, 0], robot_positions_2d[-1, 1], 
            c='blue', s=80, marker='x', zorder=5)
plt.scatter(bottle_positions_2d[-1, 0], bottle_positions_2d[-1, 1], 
            c='red', s=80, marker='x', zorder=5)

# Labels and formatting
plt.xlabel('X Position (m)', fontsize=12)
plt.ylabel('Y Position (m)', fontsize=12)
plt.title('Robot Base and Bottle 2D Positions', fontsize=14, fontweight='bold')
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.axis('equal')  # Equal aspect ratio for accurate spatial representation

# Optional: add frame index as color gradient for trajectory
# (uncomment below if you want to visualize temporal progression)
# scatter = plt.scatter(robot_positions_2d[:, 0], robot_positions_2d[:, 1], 
#                       c=df['frame_index'], cmap='viridis', s=10, alpha=0.5, label='Robot (time-colored)')
# plt.colorbar(scatter, label='Frame Index')

plt.tight_layout()
plt.show()

# Optional: print some stats
print(f"Total frames: {len(df)}")
print(f"Robot start position (x, y): {robot_positions_2d[0]}")
print(f"Robot end position (x, y): {robot_positions_2d[-1]}")
print(f"Bottle position (x, y): {bottle_positions_2d[0]}")