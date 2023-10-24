import pandas as pd

# Create a sample DataFrame
data = {'Category': ['A', 'B', 'C', 'A', 'B', 'B', 'C', 'A', 'A']}
df = pd.DataFrame(data)

# Use the mode() function to find the mode of the 'Category' column
mode_value = df['Category'].mode()[0]

print("Mode:", mode_value)
