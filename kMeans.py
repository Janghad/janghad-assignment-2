import random

# Euclidean distance calculation
def euclidean_distance(p1, p2):
    return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5

# Initialize centroids randomly within the range [-10, 10]
def initialize_random(points, k):
    return [[random.uniform(-10, 10), random.uniform(-10, 10)] for _ in range(k)]

# Initialize centroids using the farthest-first strategy, constrained to [-10, 10]
def initialize_farthest_first(points, k):
    centroids = [[random.uniform(-10, 10), random.uniform(-10, 10)]]
    for _ in range(1, k):
        farthest_point = None
        max_distance = -1
        for point in points:
            min_distance = min([euclidean_distance(point, c) for c in centroids])
            if min_distance > max_distance:
                max_distance = min_distance
                farthest_point = point
        centroids.append(farthest_point)
    return centroids

# Initialize centroids using the KMeans++ strategy, constrained to [-10, 10]
def initialize_kmeans_pp(points, k):
    centroids = [[random.uniform(-10, 10), random.uniform(-10, 10)]]
    for _ in range(1, k):
        distances = [min([euclidean_distance(point, c) for c in centroids]) ** 2 for point in points]
        probabilities = [d / sum(distances) for d in distances]
        next_centroid = random.choices(points, probabilities)[0]
        centroids.append(next_centroid)
    return centroids

# Step-by-step KMeans function
def kmeans_stepwise(points, k, init_method):
    steps = []  # To store each step's centroids and clusters

    # Initialize centroids using the chosen method
    if init_method == 'farthest_first':
        centroids = initialize_farthest_first(points, k)
    elif init_method == 'kmeans++':
        centroids = initialize_kmeans_pp(points, k)
    else:
        centroids = initialize_random(points, k)

    for step_num in range(100):  # Max iterations
        clusters = {i: [] for i in range(k)}
        for point in points:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            if distances:
                nearest_centroid = distances.index(min(distances))
                clusters[nearest_centroid].append(point)

        # Debugging: Print centroids at each step to ensure they are calculated
        print(f"Step {step_num}, Centroids: {centroids}")
        
        # Save current step (centroids and clusters)
        steps.append({'centroids': list(centroids), 'clusters': dict(clusters)})

        # Update centroids based on the cluster assignments
        new_centroids = []
        for cluster_points in clusters.values():
            if cluster_points:
                new_centroid = [sum(dim) / len(cluster_points) for dim in zip(*cluster_points)]
            else:
                new_centroid = random.choice(points)
            new_centroids.append(new_centroid)

        if new_centroids == centroids:
            break
        
        centroids = new_centroids

    return steps
