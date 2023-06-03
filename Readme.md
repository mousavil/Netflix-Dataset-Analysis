# Netflix Dataset Analysis

This project is focused on analyzing a selected dataset related to Netflix. The dataset includes a list of all the movies and TV shows available on Netflix, along with relevant details such as actors, directors, ratings, release years, and more.

## Dataset Columns

The following columns are included in the dataset:

- `show_id`: A unique identifier for each movie/TV show.
- `type`: The title of the movie/TV show.
- `title`: Director of the movie.
- `director`: Actors involved in the movie/TV show.
- `cast`: The country where the movie/TV show was produced.
- `country`: The date it was added to Netflix.
- `date_added`: The actual release year.
- `release_year`: The TV show/movie rating.
- `rating`: The total duration, in minutes or number of seasons.
- `duration`: The movie/TV show's category.
- `listed_in`: Description.

## Project Phases

This project consists of three phases. The requirements for each phase are described below:

### Phase 1

In this phase, a statistical comparison is performed between the number of movies for each director and the average number of movies for directors.

### Phase 2

In this phase, a pattern is extracted between "director" and "cast" with its genre "listed_in."

### Phase 3

In this phase, the dataset is categorized by assuming that "listed_in" is the label so that the movie genre ("label") can be predicted based on the description for test data.

## Conclusion

Overall, this project aims to provide insights into the Netflix dataset and extract meaningful patterns from it. By analyzing the dataset in different phases, we can gain a deeper understanding of the movies and TV shows available on Netflix and their associated metadata.