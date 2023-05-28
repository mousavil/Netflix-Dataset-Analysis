import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Load your data
df = pd.read_csv('netflix.csv')

# Drop missing values
df.dropna(subset=['director', 'cast'], inplace=True)

# Convert date_added to datetime
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')

# Extract year and month from date_added
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Fill missing values for year_added and month_added
df['year_added'].fillna(df['year_added'].mode()[0], inplace=True)
df['month_added'].fillna(df['month_added'].mode()[0], inplace=True)

# Drop the original date_added column
df.drop('date_added', axis=1, inplace=True)

# Convert the multi-valued features to lists
multi_valued_features = ['cast', 'country', 'listed_in', 'director']
for feature in multi_valued_features:
    df[feature] = df[feature].apply(lambda x: str(x).split(","))

# # One-hot encode 'type' and 'rating'
# one_hot_features = ['type', 'rating']
# for feature in one_hot_features:
#     encoded_features = pd.get_dummies(df[feature], prefix=feature)
#     df = pd.concat([df.drop(feature, axis=1), encoded_features], axis=1)

# Binarize the multi-valued features
# mlb = MultiLabelBinarizer()
# for feature in multi_valued_features:
#     mlb_results = mlb.fit_transform(df[feature])
#     df = df.join(pd.DataFrame(mlb_results, columns=[f'{feature}_{c}' for c in mlb.classes_], index=df.index))
#     df.drop(feature, axis=1, inplace=True)

# Save the preprocessed data
df.to_csv('Netflix_Shows_Preprocessed.csv', index=False)
