#First Phase

import pandas as pd

# Assuming the data is in a csv file named 'netflix_data.csv'
df = pd.read_csv('Netflix_Shows_Preprocessed.csv')


# Filter out movies from the dataset
df_movies = df[df['type'] == 'Movie']

# Count the number of films each director has on Netflix
director_counts = df_movies['director'].value_counts()

# Calculate the average number of films per director
average_films = director_counts.mean()

print(director_counts)
print(f"Average number of films per director: {average_films}")


#End of first phase 


# #Second phase 


import pandas as pd

# Assuming the data is in a csv file named 'netflix_data.csv'
df = pd.read_csv('Netflix_Shows_Preprocessed.csv')

# Create a new column combining the director, cast, and genre columns
df['director_cast_genre'] = df['director'] + ', ' + df['cast'] + ', ' + df['listed_in']

# Count the occurrences of each combination
combo_counts = df['director_cast_genre'].value_counts()

# Print the most frequent combinations
print(combo_counts.head())


# #End of second phase 


# #Third phase 



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Assuming the data is in a csv file named 'netflix_data.csv'
df = pd.read_csv('Netflix_Shows_Preprocessed.csv')

# Preparing data
X = df['description']
y = df['listed_in']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a pipeline that first transforms the data using TF-IDF and then applies Multinomial Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(X_train, y_train)

# Predicting genres for the test set
y_pred = model.predict(X_test)

# Evaluating the model
print(classification_report(y_test, y_pred))

