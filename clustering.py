import csv
import math
import random
import matplotlib.pyplot as plt  

MAX_K = 10
MAX_ITERS = 100

class Point:
    def __init__(self, itc_marks, ds_marks):
        self.itc_marks = itc_marks
        self.ds_marks = ds_marks

def read_csv(filename):
    data = []
    sum_itc, sum_ds = 0, 0
    count_itc, count_ds = 0, 0

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  

        for row in csv_reader:
            itc_marks = 0
            ds_marks = 0

            if row[1] not in ['Absent', '']:
                itc_marks = float(row[1])
                sum_itc += itc_marks
                count_itc += 1
            if row[2] not in ['Absent', '']:
                ds_marks = float(row[2])
                sum_ds += ds_marks
                count_ds += 1

            data.append(Point(itc_marks, ds_marks))

    mean_itc = sum_itc / count_itc if count_itc else 0
    mean_ds = sum_ds / count_ds if count_ds else 0

    for point in data:
        if point.itc_marks == 0:
            point.itc_marks = mean_itc
        if point.ds_marks == 0:
            point.ds_marks = mean_ds

    return data

def euclidean_distance(a, b):
    return math.sqrt((a.itc_marks - b.itc_marks) ** 2 + (a.ds_marks - b.ds_marks) ** 2)

def initialize_centroids(data, k):
    return random.sample(data, k)

def assign_clusters(data, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        cluster_index = distances.index(min(distances))
        clusters[cluster_index].append(point)
    return clusters

def update_centroids(clusters, prev_centroids):
    centroids = []
    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:  
            centroids.append(random.choice(prev_centroids))  
        else:
            avg_itc = sum(point.itc_marks for point in cluster) / len(cluster)
            avg_ds = sum(point.ds_marks for point in cluster) / len(cluster)
            centroids.append(Point(avg_itc, avg_ds))
    return centroids

def k_means(data, k):
    centroids = initialize_centroids(data, k)
    for _ in range(MAX_ITERS):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(clusters, centroids)
        
        if centroids == new_centroids:  
            break
        centroids = new_centroids  

    inertia = sum(
        euclidean_distance(point, centroids[i]) ** 2
        for i, cluster in enumerate(clusters)
        for point in cluster
    )
    return clusters, centroids, inertia

def elbow_method(data):
    inertias = []
    ks = list(range(1, MAX_K + 1))
    for k in ks:
        _, _, inertia = k_means(data, k)
        inertias.append(inertia)
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.show()

def visualize_clusters(clusters):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, cluster in enumerate(clusters):
        x = [point.itc_marks for point in cluster]
        y = [point.ds_marks for point in cluster]
        plt.scatter(x, y, color=colors[i % len(colors)], label=f'Cluster {i+1}')
    plt.xlabel('ITC Marks')
    plt.ylabel('DS Marks')
    plt.title('Student Clusters')
    plt.legend()
    plt.show()

def main():
    filename = 'pw-5 Sample Data CSV File (1).csv'  
    data = read_csv(filename)
    print(f'Total Data Points: {len(data)}')

    elbow_method(data)

    k = 3
    clusters, centroids, _ = k_means(data, k)
    visualize_clusters(clusters)

if __name__ == '__main__':
    main()
