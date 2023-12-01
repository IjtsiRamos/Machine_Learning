#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
import random

def normalize_minmax(df, min_val=-1, max_val=1):
    """
    Normalize a list of values to the specified range.
    Returns:
        list: The normalized values within the specified range.
    """
    normalized_df = df.copy()
    
    # Normalize numeric columns
    for column in normalized_df.select_dtypes(include=['number']).columns:
        min_data = normalized_df[column].min()
        max_data = normalized_df[column].max()

        if min_data == max_data:
            raise ValueError(f"All values in column '{column}' are equal. Cannot normalize.")

        normalized_df[column] = (normalized_df[column] - min_data) / (max_data - min_data) * (max_val - min_val) + min_val
    
    return normalized_df

def one_hot_encode_categorical(df):
    """
    Perform one-hot encoding on categorical columns in a DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with categorical columns one-hot encoded.
    """
    # Select only non-numerical (categorical) columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns)
    df_encoded = df_encoded.astype('int64')
    
    return df_encoded

def k_means(df, k, iterations):
    """
    Perform clustering of a DataFrame.

    Returns:
        Sum of the squared distances between points and centroids
        pd.DataFrame: The DataFrame with the new cluster column.
    """
    # Make a copy of the input DataFrame to avoid modifying the original.
    df_copy = df.copy()

    # Initialize centroids using k-means++ initialization.
    centroids = df_copy.sample(k).values

    # Initialize an empty dictionary to store clusters.
    clusters = {f'c{i}': [] for i in range(1, k + 1)}

    # Iterate until the centroids converge.
    stop = False
    i = 0
    while not stop:
        # Calculate distances and reassign vectors to centroids.
        for idx, row in df_copy.iterrows():
            distances = {}
            for label, centroid in zip(clusters.keys(), centroids):
                distance_to_centroid = np.linalg.norm(row.values - centroid)
                distances[label] = distance_to_centroid

            # Find the closest cluster.
            closest_centroid = min(distances, key=distances.get)

            clusters[closest_centroid].append(idx)

        # Recalculate centroids as the mean of vectors in each cluster.
        centroids_new = pd.DataFrame({centroid: np.mean(df_copy.loc[cluster_indices], axis=0)
                                      for centroid, cluster_indices in clusters.items()})

        # Stopping criteria: if the new calculated centroid is very similar to the actual one or if the number of iterations reaches 50.
        if (abs(centroids_new.T - centroids).max() < 0.1).all() or i == iterations:
            stop = True
        else:
            centroids = centroids_new.values.T
            i += 1

    # Calculate the sum of squared distances within clusters.
    sum_squared_distances = 0
    for centroid, cluster_indices in clusters.items():
        centroid_values = centroids_new[centroid].values
        cluster_points = df_copy.loc[cluster_indices].values
        cluster_distances = np.sum(np.linalg.norm(cluster_points - centroid_values, axis=1) ** 2)
        sum_squared_distances += cluster_distances

    # Add a new column to the original DataFrame indicating cluster membership.
    df_copy['cluster'] = [min(clusters, key=lambda x: idx in clusters[x]) for idx in df_copy.index]

    return sum_squared_distances, df_copy


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def clarans(df, k, max_neighbors, num_trials):
    """
    Perform clustering of a DataFrame.

    Returns:
        Array of best clusters
        Values of best cost
    """
    n_samples, n_features = df.shape
    best_cost = float('inf')
    best_clusters = None

    for _ in range(num_trials):
        # Randomly initialize k cluster centers
        centroids_idx = random.sample(range(n_samples), k)
        centroids = df.iloc[centroids_idx].values

        
        distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids] for point in df.values])
        cluster_assignments = np.argmin(distances, axis=1)

        # Calculate the total cost of the current clustering
        total_cost = sum(np.min(euclidean_distance(df.iloc[i].values, centroids[cluster_assignments[i]])) for i in range(n_samples))

        # Update the best clustering if the cost is lower
        if total_cost < best_cost:
            best_cost = total_cost
            best_clusters = cluster_assignments

        for _ in range(max_neighbors):
            # Randomly choose a data point
            current_idx = random.choice(range(n_samples))
            current_point = df.iloc[current_idx].values

            # Randomly choose another data point that is not the current point
            neighbor_idx = random.choice([i for i in range(n_samples) if i != current_idx])
            neighbor_point = df.iloc[neighbor_idx].values

            # Calculate the total cost before swapping centroids
            current_cost = total_cost

            # Swap a centroid with the neighbor_point
            centroids_copy = centroids.copy()
            centroids_copy[random.choice(range(k))] = neighbor_point

            #Assign data points to the nearest centroid
            distances = np.array([[euclidean_distance(point, centroid) for centroid in centroids_copy] for point in df.values])
            cluster_assignments = np.argmin(distances, axis=1)

            # Calculate the new cost after swapping
            new_cost = sum(np.min(euclidean_distance(df.iloc[i].values, centroids_copy[cluster_assignments[i]])) for i in range(n_samples))

            # If the new cost is lower, update the centroids and total cost
            if new_cost < current_cost:
                centroids = centroids_copy
                total_cost = new_cost

    return best_clusters, best_cost

def corr_Pearson(df):
    """
    Perform Pearson correlation on a DataFrame.

    Returns:
        Array of best clusters
        Plot of Pearson Correlation Matrix
    """
    correlation_matrix = df.corr()

    # Create a heatmap of the Pearson correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Pearson Correlation Heatmap')
    plt.show()

def MICE_Imputation(df):
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df_numerical = df[numerical_columns]

    # Initialize the MICE imputer
    imputer = IterativeImputer(max_iter=10, random_state=0)

    # Perform MICE imputation on numerical attributes
    imputed_data = imputer.fit_transform(df_numerical)

    # Convert the imputed data back to a DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=numerical_columns)

    # Combine the imputed numerical attributes with the non-numerical attributes
    for col in df.columns:
        if col not in numerical_columns:
            imputed_df[col] = df[col]

    for column in df_numerical.columns[df_numerical.isnull().any()]:
        plt.figure(figsize=(10, 4))
        sns.histplot(df_numerical[column], label='Original')
        sns.histplot(imputed_df[column],alpha=0.4, label='Imputed', color='orange')
        plt.title(f'Distribution of {column} (Original vs. Imputed)')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
    
    return imputed_df
    
def cat_to_num(df):
    """
    Map categorical columns (type 'object') in a DataFrame to numerical values.

    Returns:
    - A new DataFrame with categorical columns replaced by numerical values.
    """
    df_numerical = df.copy()  # Create a copy of the original DataFrame

    # Iterate through columns and check if they are of type 'object'
    for column in df.columns:
        if df[column].dtype == 'object':
            # Get the unique categorical values for the column
            attribute_values = df[column].unique()
            
            # Create a dictionary to map categorical values to numerical values
            mapping_dict = {value: index for index, value in enumerate(attribute_values)}
            
            # Map the column to numerical values
            df_numerical[column] = df_numerical[column].map(mapping_dict)

    return df_numerical


# In[ ]:




