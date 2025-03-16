# this file generates point clouds from SMLM data into csv files

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# Function to generate point clouds
def generate_point_clouds(real_data, num_clouds, num_points):
    point_clouds = []
    for _ in range(num_clouds):
        sampled_points = real_data.sample(num_points).reset_index(drop=True)
        point_clouds.append(sampled_points)
    return point_clouds

def main():
    # Ensure you load the correct data before running
    # Replace the placeholders below with actual datasets
    DMSO_data = pd.read_csv('DMSO_data.csv')
    Pitstop_data = pd.read_csv('Pitstop_data.csv')

    # Generate 100 point clouds for each class
    print("Generating point clouds...")
    class_1_point_clouds = generate_point_clouds(DMSO_data[['x', 'y', 'z']], 100, 100)
    class_2_point_clouds = generate_point_clouds(Pitstop_data[['x', 'y', 'z']], 100, 100)

    # Save point clouds as CSV files for later use
    os.makedirs(f"data/class_1_real", exist_ok=True)
    os.makedirs(f"data/class_2_real", exist_ok=True)

    for i, pc in enumerate(class_1_point_clouds):
        pc.to_csv(f"data/class_1_real/point_cloud_{i+1}.csv", index=False)

    for i, pc in enumerate(class_2_point_clouds):
        pc.to_csv(f"data/class_2_real/point_cloud_{i+1}.csv", index=False)

    print("Point clouds generated and saved successfully!")

    # Visualize the first point cloud from each class
    df_first = class_1_point_clouds[0]
    df_second = class_2_point_clouds[0]

    fig = plt.figure(figsize=(14, 6))

    # Ensure columns are correctly named for visualization
    df_first.columns = ['x', 'y', 'z']
    df_second.columns = ['x', 'y', 'z']

    # Plot for Class 1
    ax1 = fig.add_subplot(121, projection='3d')
    x1, y1, z1 = df_first['x'], df_first['y'], df_first['z']
    ax1.scatter(x1, y1, z1, c='blue', marker='o', s=5, alpha=0.5)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate')
    ax1.set_title('3D Scatter Plot of Class 1 Point Cloud')

    # Plot for Class 2
    ax2 = fig.add_subplot(122, projection='3d')
    x2, y2, z2 = df_second['x'], df_second['y'], df_second['z']
    ax2.scatter(x2, y2, z2, c='blue', marker='o', s=5, alpha=0.5)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    ax2.set_zlabel('Z Coordinate')
    ax2.set_title('3D Scatter Plot of Class 2 Point Cloud')

    # Display
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()