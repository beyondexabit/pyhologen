import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# Example: Reading a CSV file into a DataFrame
csv_file_path = '/Users/jakubkostial/Documents/phd/code/mode_generation/pytestholo21.csv'
df = pd.read_csv(csv_file_path)





# Define the size of the desired region (1920 x 1080)
target_width = 1920
target_height = 1080

# Calculate the indices for extracting the central region
start_row = (df.shape[0] - target_height) // 2
end_row = start_row + target_height
start_col = (df.shape[1] - target_width) // 2
end_col = start_col + target_width

# Extract the central region
center_region = df.iloc[start_row:end_row, start_col:end_col]
# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
# Scale the central region to the desired size
scaled_center_region = pd.DataFrame(scaler.fit_transform(center_region), columns=center_region.columns)


# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 128))

# Apply scaling to the DataFrame
scaled_df = pd.DataFrame(scaler.fit_transform(scaled_center_region), columns=scaled_center_region.columns)


# Plot the DataFrame as a 2D image
plt.imshow(scaled_df.values, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Values')
plt.title('DataFrame as 2D Image')
plt.show()


