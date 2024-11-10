import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import pgeocode

# Load the CSV file
file_path = 'C:/Users/Dell G15 5520/Downloads/DAC_NationalDownloadableFile.csv'
data = pd.read_csv(file_path)

# Select the first 10,000 rows and 'ZIP Code' column for clustering
zip_data = data[['ZIP Code']].head(10000)

# Convert 'ZIP Code' to numeric, truncate to first 5 digits, and drop NaN values
zip_data['ZIP Code'] = pd.to_numeric(zip_data['ZIP Code'], errors='coerce').astype(str).str[:5].astype(int)
zip_data = zip_data.dropna()

# Reshape the data for K-Means
X = zip_data.values.reshape(-1, 1)

# Apply K-Means clustering with 10 clusters
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X)

# Add cluster labels to the DataFrame
zip_data['Cluster'] = kmeans.labels_

# Use pgeocode to find the state, latitude, and longitude for each ZIP Code
nomi = pgeocode.Nominatim('us')
zip_data['State'] = zip_data['ZIP Code'].apply(lambda x: nomi.query_postal_code(x).state_name)
zip_data['Latitude'] = zip_data['ZIP Code'].apply(lambda x: nomi.query_postal_code(x).latitude)
zip_data['Longitude'] = zip_data['ZIP Code'].apply(lambda x: nomi.query_postal_code(x).longitude)

# Drop any rows where latitude or longitude is missing
zip_data = zip_data.dropna(subset=['Latitude', 'Longitude'])

# Group the data by clusters and states
cluster_state_data = zip_data.groupby(['Cluster', 'State']).size().reset_index(name='Count')

# Path to the downloaded Natural Earth shapefile
usa_shapefile_path = r"C:/Users/Dell G15 5520/Downloads/cb_2018_us_nation_5m/cb_2018_us_nation_5m.shp"  # Replace with your actual path

# Load the shapefile
usa = gpd.read_file(usa_shapefile_path)

# Plot the map
fig, ax = plt.subplots(figsize=(15, 10))
usa.plot(ax=ax, color='lightgray')

# Set the x and y limits to zoom in on the USA region
ax.set_xlim(-130, -60)  # Adjust these values for better zoom
ax.set_ylim(20, 55)     # Adjust these values for better zoom

# Plot points for clustered ZIP Codes
ax.scatter(zip_data['Longitude'], zip_data['Latitude'], c=zip_data['Cluster'], cmap='tab10', s=10, edgecolor='black')

plt.title('K-Means Clusters of ZIP Codes on USA Map (First 10,000 Entries)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Print the clustered data categorized by US states
print(cluster_state_data)