import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("Data/hotel_bookings.csv")
df['index'] = df.index

# Split the data into training and test sets
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Split the training set further into training and validation sets
df_train, df_val = train_test_split(df_train, test_size=0.05 / 0.8, random_state=42)  # Adjust the test_size to maintain the correct proportion

# Save the datasets to CSV files
df_train.to_csv('Data/train_data.csv', index=False)
df_val.to_csv('Data/validation_data.csv', index=False)
df_test.to_csv('Data/test_data.csv', index=False)

# Print the sizes of the datasets
print('Train size:', len(df_train), 'Test size:', len(df_test), 'Validation size:', len(df_val), flush=True)