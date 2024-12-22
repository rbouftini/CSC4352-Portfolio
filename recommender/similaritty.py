import numpy as np

# Define films with the IDs of the users that like them
film0 = np.array([1, 2, 6, 3])    
film1 = np.array([0, 1, 8, 7, 4])
film2 = np.array([0, 1, 2, 6, 5])
films = [film0, film1, film2]
film_names = ['Film A', 'Film B', 'Film C']

# Function to compute exact Jaccard similarity
def compute_jaccard_similarity(films):
    n = len(films)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif similarity_matrix[i][j] == 0:
                intersection = len(np.intersect1d(films[i], films[j]))
                union = len(np.union1d(films[i], films[j]))
                sim = intersection / union
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    return similarity_matrix

# Compute exact Jaccard similarity
exact_similarity = compute_jaccard_similarity(films)
print("Exact Jaccard Similarity Matrix:\n", exact_similarity)

# Function to compute Min Hash signatures
def compute_min_hash_signatures(matrix, num_permutations=1000):
    n_users, n_films = matrix.shape
    signatures = np.full((num_permutations, n_films), np.inf)
    
    for i in range(num_permutations):
        permutation = np.random.permutation(n_users)
        for j in range(n_films):
            liked_users = np.where(matrix[:, j] == 1)[0]
            if liked_users.size > 0:
                # Create a boolean mask where True indicates the user likes the film
                mask = np.isin(permutation, liked_users)
                # Find the index of the first True value
                first_liked_index = np.argmax(mask)
                # Assign the user ID to the signature matrix
                signatures[i, j] = permutation[first_liked_index]
    
    return signatures

# Create the user-film matrix
n_users = 9
matrix = np.zeros((n_users, len(films)))
for i in range(n_users):
    if i in film0:
        matrix[i, 0] = 1
    if i in film1:
        matrix[i, 1] = 1
    if i in film2:
        matrix[i, 2] = 1

print("\nUser-Film Matrix:\n", matrix)

# Compute Min Hash signatures
max_permutations = 1000
min_hash_signatures = compute_min_hash_signatures(matrix, max_permutations)

# Function to compute Min Hash Jaccard similarity
def min_hash_jaccard_similarity(signatures):
    n = signatures.shape[1]
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif similarity_matrix[i][j] == 0:
                matches = np.sum(signatures[:, i] == signatures[:, j])
                sim = matches / signatures.shape[0]
                similarity_matrix[i][j] = sim
                similarity_matrix[j][i] = sim
    return similarity_matrix

# Compute Min Hash Jaccard similarity
min_hash_similarity = min_hash_jaccard_similarity(min_hash_signatures)
print("\nMin Hash Jaccard Similarity Matrix:\n", min_hash_similarity)

# Function to get top N similar films for each film
def get_top_n_similar(similarity_matrix, film_names, top_n=1):
    n = similarity_matrix.shape[0]
    recommendations = {}
    for i in range(n):
        # Get indices of top_n similar films (excluding the film itself)
        similar_indices = similarity_matrix[i].argsort()[::-1][1:top_n+1]
        recommendations[film_names[i]] = [film_names[j] for j in similar_indices]
    return recommendations

# Generate recommendations based on exact Jaccard similarity
exact_recommendations = get_top_n_similar(exact_similarity, film_names, top_n=2)
print("\nRecommendations based on Exact Jaccard Similarity:")
for film, recs in exact_recommendations.items():
    print(f"{film}: {recs}")

# Generate recommendations based on Min Hash similarity
min_hash_recommendations = get_top_n_similar(min_hash_similarity, film_names, top_n=2)
print("\nRecommendations based on Min Hash Jaccard Similarity:")
for film, recs in min_hash_recommendations.items():
    print(f"{film}: {recs}")

# (Optional) Visualization of the Similarity Matrix
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_similarity_matrix(similarity_matrix, film_names, title):
    plt.figure(figsize=(8,6))
    sns.heatmap(similarity_matrix, annot=True, xticklabels=film_names, yticklabels=film_names, cmap='Blues')
    plt.title(title)
    plt.show()

# Visualize Exact Jaccard Similarity
visualize_similarity_matrix(exact_similarity, film_names, "Exact Jaccard Similarity")

# Visualize Min Hash Jaccard Similarity
visualize_similarity_matrix(min_hash_similarity, film_names, "Min Hash Jaccard Similarity")
