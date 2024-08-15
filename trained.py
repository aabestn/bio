import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology
from sklearn.cluster import KMeans
from scipy import ndimage as ndi

# Step 1: Load and preprocess the image
image_path = '50ka.png'  # Update this path to your PNG file
image = io.imread(image_path)

# Check if the image has an alpha channel (4 channels) and remove it
if image.shape[-1] == 4:
    image = image[:, :, :3]  # Remove the alpha channel

# Convert to grayscale
gray_image = color.rgb2gray(image)

# Apply Gaussian filter for smoothing
smoothed_image = filters.gaussian(gray_image, sigma=1.0)

# Step 2: Edge detection (e.g., using Sobel filter)
edges = filters.sobel(smoothed_image)

# Step 3: Binary thresholding to segment potential nodes
thresh = filters.threshold_otsu(edges)
binary_image = edges > thresh

# Perform morphological operations to remove small artifacts
cleaned_image = morphology.remove_small_objects(binary_image, min_size=150)

# Label connected regions (nodes)
labels = measure.label(cleaned_image)

# Step 4: Feature extraction and clustering
# Extract region properties
regions = measure.regionprops(labels)

# Collect centroids of the detected regions
centroids = np.array([region.centroid for region in regions])

# Use KMeans clustering (optional step depending on application)
kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
kmeans.fit(centroids)

# Get the cluster centers (nodes locations)
nodes = kmeans.cluster_centers_

# Step 5: Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(gray_image, cmap='gray')
ax.scatter(nodes[:, 1], nodes[:, 0], c='red', marker='o')  # Nodes as red dots
ax.set_title('Detected Nodes in SEM Image')
plt.show()
