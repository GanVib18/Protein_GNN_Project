# this file generates simulated data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_uniform_point_cloud(n_points, range_min=-10, range_max=10):
    """Generate a uniformly distributed 3D point cloud."""
    points = np.random.uniform(range_min, range_max, size=(n_points, 3))
    return pd.DataFrame(points, columns=['x', 'y', 'z'])

def generate_point_cloud_with_cluster(n_points, n_clusters=3, cluster_size=50, range_min=-20, range_max=20, cluster_range=1):
    """
    Generate a 3D point cloud with a dense cluster.
    n_points: Total number of points.
    cluster_size: Number of points in the dense cluster.
    cluster_range: Range for the cluster's coordinates.
    """
    # Generate the uniformly distributed background points
    points = np.random.uniform(range_min, range_max, size=(n_points, 3))   
    # Create clusters
    total_cluster_points = n_clusters * cluster_size
    if total_cluster_points > n_points:
        raise ValueError("Total cluster points exceed the number of available points.")
    
    cluster_indices = np.random.choice(n_points, total_cluster_points, replace=False)
    
    # Generate each cluster around a random center point
    for i in range(n_clusters):
        cluster_center = np.random.uniform(range_min, range_max, size=(1, 3))
        cluster_points = cluster_center + np.random.uniform(-cluster_range, cluster_range, size=(cluster_size, 3))
        points[cluster_indices[i * cluster_size:(i + 1) * cluster_size]] = cluster_points

    return pd.DataFrame(points, columns=['x', 'y', 'z'])

def main():
    os.makedirs("class_1_simulated", exist_ok=True)
    os.makedirs("class_2_simulated", exist_ok=True)
    for i in range(100):
        df_first = generate_uniform_point_cloud(100)
        df_second = generate_point_cloud_with_cluster(160, n_clusters=2, cluster_size=30)
        df_first.to_csv(f"class_1_simulated/point_cloud_{i+1}.csv", index=False)
        df_second.to_csv(f"class_2_simulated/point_cloud_{i+1}.csv", index=False)

    print("Point cloud generation completed!")



if __name__ == '__main__':
    main()
