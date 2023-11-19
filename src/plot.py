import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Replace 'your_file.csv' with the actual filename
file_path = os.path.abspath(os.path.join(os.getcwd(), 'submission.csv'))

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Extract the probabilities column
probabilities = data['target']

# Define the bins for the histogram
bins = np.arange(0, 1.0, 0.1)  # Adjust the bin size as needed

# Plot the histogram
plt.hist(probabilities, bins=bins, edgecolor='black')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.title('Probability Distribution')
plt.xticks(bins)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()