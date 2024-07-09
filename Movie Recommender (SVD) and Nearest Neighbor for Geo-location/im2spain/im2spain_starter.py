"""
The goal of this assignment is to predict GPS coordinates from image features using k-Nearest Neighbors.
Specifically, have featurized 28616 geo-tagged images taken in Spain split into training and test sets (27.6k and 1k).

The assignment walks students through:
    * visualizing the data
    * implementing and evaluating a kNN regression model
    * analyzing model performance as a function of dataset size
    * comparing kNN against linear regression

Images were filtered from Mousselly-Sergieh et al. 2014 (https://dl.acm.org/doi/10.1145/2557642.2563673)
and scraped from Flickr in 2024. The image features were extracted using CLIP ViT-L/14@336px (https://openai.com/clip/).
"""

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def plot_data(train_feats, train_labels):
    """
    Input:
        train_feats: Training set image features
        train_labels: Training set GPS (lat, lon)

    Output:
        Displays plot of image locations, and first two PCA dimensions vs longitude
    """
    # Plot image locations (use marker='.' for better visibility)
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title('Image Locations')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Run PCA on training_feats
    ##### TODO(a): Your Code Here #####
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by first two PCA dimensions (use marker='.' for better visibility)
    plt.scatter(transformed_feats[:, 0],     # Select first column
                transformed_feats[:, 1],     # Select second column
                c=train_labels[:, 1],
                marker='.')
    plt.colorbar(label='Longitude')
    plt.title('Image Features by Longitude after PCA')
    plt.show()


def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Input:
        train_features: Training set image features
        train_labels: Training set GPS (lat, lon) coords
        test_features: Test set image features
        test_labels: Test set GPS (lat, lon) coords
        is_weighted: Weight prediction by distances in feature space

    Output:
        Prints mean displacement error as a function of k
        Plots mean displacement error vs k

    Returns:
        Minimum mean displacement error
    """
    # Evaluate mean displacement error (in miles) of kNN regression for different values of k
    # Technically we are working with spherical coordinates and should be using spherical distances, but within a small
    # region like Spain we can get away with treating the coordinates as cartesian coordinates.
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f'Running grid search for k (is_weighted={is_weighted})')

    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []
    for k in ks:
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # Evaluate mean displacement error in miles for each test image
            # Assume 1 degree latitude is 69 miles and 1 degree longitude is 52 miles
            y = test_labels[i]

            ##### TODO(d): Your Code Here #####
            if is_weighted:
              weights = 1/ (distances[i] + 1e-5)
              sum_weights = np.sum(weights)
              predicted_label = np.sum(train_labels[nearest] * weights[:, np.newaxis], axis=0)/sum_weights
            else:
              predicted_label = np.mean(train_labels[nearest], axis=0)
            
            lat_error = (predicted_label[0] - y[0]) * 69
            lon_error = (predicted_label[1] - y[1]) * 52
            e = np.sqrt(lat_error**2 + lon_error**2)                                      
            errors.append(e)
        
        e = np.mean(np.array(errors))
        mean_errors.append(e)
        if verbose:
            print(f'{k}-NN mean displacement error (miles): {e:.1f}')

    # Plot error vs k for k Nearest Neighbors
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel('k')
        plt.ylabel('Mean Displacement Error (miles)')
        plt.title('Mean Displacement Error (miles) vs. k in kNN')
        plt.show()

    return min(mean_errors)


def main():
    print("Predicting GPS from CLIP image features\n")

    # Import Data
    print("Loading Data")
    data = np.load('im2spain/im2spain_data.npz')

    train_features = data['train_features']  # [N_train, dim] array
    test_features = data['test_features']    # [N_test, dim] array
    train_labels = data['train_labels']      # [N_train, 2] array of (lat, lon) coords
    test_labels = data['test_labels']        # [N_test, 2] array of (lat, lon) coords
    train_files = data['train_files']        # [N_train] array of strings
    test_files = data['test_files']          # [N_test] array of strings

    # Data Information
    print('Train Data Count:', train_features.shape[0])

    # Part A: Feature and label visualization (modify plot_data method)
    plot_data(train_features, train_labels)

    # Part B: Find the 3 nearest neighbors of test image 53633239060.jpg
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)

    # Use knn to get the k nearest neighbors of the features of image 53633239060.jpg
    ##### TODO(b): Your Code Here #####
    test_idx = np.where(test_files == '53633239060.jpg')[0][0]
    test_feature = test_features[test_idx]

    distances, indices = knn.kneighbors([test_feature], n_neighbors=3)
    neighbors_files = train_files[indices.flatten()]
    neighbors_coords = train_labels[indices.flatten()]
    print("Nearest neighbors files:", neighbors_files)
    print("Their coordinates:", neighbors_coords)

    # Part C: establish a naive baseline of predicting the mean of the training set
    ##### TODO(c): Your Code Here #####
    centroid = np.mean(train_labels, axis=0)
    print("Training set centroid (lat, lon): ", centroid)

    displacements = np.linalg.norm(test_labels - centroid, axis=1)
    miles_per_degree_lat = 69
    miles_per_degree_lon = 52
    displacements_miles = displacements * np.sqrt(miles_per_degree_lat**2 + miles_per_degree_lon**2)

    mde = np.mean(displacements_miles)
    print("Naive Baseline MDE (miles): ", mde)

    # Part D: complete grid_search to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels)

    # Part F: rerun grid search after modifications to find the best value of k
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)

    # Part H: compare to linear regression for different # of training points
    mean_errors_lin = []
    mean_errors_nn = []
    ratios = np.arange(0.1, 1.1, 0.1)
    for r in ratios:
        num_samples = int(r * len(train_features))
        ##### TODO(h): Your Code Here #####
        subset_features = train_features[:num_samples]
        subset_labels = train_labels[:num_samples]

        ## Find optimal k
        ks = range(1,21)
        errors = []
        for k in ks:
          knn = NearestNeighbors(n_neighbors=k)
          knn.fit(train_features)
          distances, indices = knn.kneighbors(test_features)
          predictions = np.zeros(test_labels.shape)
          for i in range(len(test_features)):
            weights = 1/(distances[i] + 1e-8)
            weighted_labels = train_labels[indices[i]] * weights[:, np.newaxis]
            predictions[i] = np.sum(weighted_labels, axis=0) /np.sum(weights)
          mse = np.mean(np.sqrt(np.sum((predictions - test_labels)**2, axis=1)))
          errors.append(mse)
        optimal_k = ks[np.argmin(errors)]

        # K-NN
        knn = NearestNeighbors(n_neighbors=optimal_k)
        knn.fit(subset_features)
        distances, indices = knn.kneighbors(test_features)
        predictions_knn = np.zeros(test_labels.shape)
        for i in range(len(test_features)):
          weights = 1/(distances[i] +1e-8)
          weighted_labels = subset_labels[indices[i]] * weights[:, np.newaxis]
          predictions_knn[i] = np.sum(weighted_labels, axis=0)/np.sum(weights)

        lat_errors_knn = (predictions_knn[:, 0] - test_labels[:, 0]) * 69
        lon_errors_knn = (predictions_knn[:, 1] - test_labels[:, 1]) * 52
        e_nn = np.mean(np.sqrt(lat_errors_knn**2 + lon_errors_knn**2))

        # Linear Regression
        lr = LinearRegression()
        lr.fit(subset_features, subset_labels)
        predictions_lr = lr.predict(test_features)

        

        lat_errors_lr = (predictions_lr[:, 0] - test_labels[:, 0]) * 69
        lon_errors_lr = (predictions_lr[:, 1] - test_labels[:, 1]) * 52 
        e_lin = np.mean(np.sqrt(lat_errors_lr**2 + lon_errors_lr**2))


        mean_errors_lin.append(e_lin)
        mean_errors_nn.append(e_nn)

        print(f'\nTraining set ratio: {r} ({num_samples})')
        print(f'Linear Regression mean displacement error (miles): {e_lin:.1f}')
        print(f'kNN mean displacement error (miles): {e_nn:.1f}')

    # Plot error vs training set size
    plt.plot(ratios, mean_errors_lin, label='lin. reg.')
    plt.plot(ratios, mean_errors_nn, label='kNN')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Mean Displacement Error (miles)')
    plt.title('Mean Displacement Error (miles) vs. Training Set Ratio')
    plt.legend()
    plt.show()
       

if __name__ == '__main__':
    main()
