# preprocessing the data

import pandas as pd
from sklearn.preprocessing import StandardScaler

# load the point cloud data
def load_point_cloud(file_path):
    return pd.read_csv(file_path).values

# scale the given point cloud data using sklearn 
# https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.StandardScaler.html
def scale_point_cloud(point_cloud):
    scaler = StandardScaler()
    return scaler.fit_transform(point_cloud)

###################
# process the data
def preprocess_data(file_path):
    point_cloud = load_point_cloud(file_path)
    scaled_point_cloud = scale_point_cloud(point_cloud)
    return scaled_point_cloud

if __name__ == "__main__":
    # Example usage
    file_path1 = "class_1_simulated/point_cloud_1.csv"
    processed_data1 = preprocess_data(file_path1)
    print(processed_data1)

    file_path2 = "class_2_simulated/point_cloud_1.csv"
    processed_data2 = preprocess_data(file_path2)
    print(processed_data2)