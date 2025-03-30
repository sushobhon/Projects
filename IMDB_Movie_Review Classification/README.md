## IMDB Review Classification

I took the dataset of IMDB movie review classification from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

### Cleaning Text
We cleaned the actual text review:
- Removed HTML tags.
- Removed Special Characters.
- Removed Stopwords.
- Stemming the words.

### Embedding
Stemming each token to find its root form.

### Classification Model
We have build 3 Classification Model. Logistic Regression, Support Vector Machine and Random Forest Model. Out of these 2 models Support Vector Machine is outperforming all other models in the test set.
