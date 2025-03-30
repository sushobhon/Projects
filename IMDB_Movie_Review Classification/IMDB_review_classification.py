import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""# IMDB Review Classification""")
    return


@app.cell
def _():
    # Importing Necessary Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
    return np, pd, plt, sns


@app.cell
def _(pd):
    # Loading Raw Data
    df_review = pd.read_csv("./IMDB Dataset.csv")
    df_review.head()
    return (df_review,)


@app.cell
def _(df_review, mo):
    mo.md(f"""The Dataset has total {df_review.shape[0]} reviews and sentiments of each review. We have 2 classes of sentiments - Positive and Negative""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Distribution of each sentiment class
        Let's first check the distribution of each classes in the data.
        """
    )
    return


@app.cell
def _(df_review, plt, sns):
    # Defining Color Palate
    colors = sns.color_palette('deep')

    # Plotting Distribution of sentiments
    plt.figure(figsize= (8,5))
    plt.bar(x=['Positive', 'Negative'],
            height=df_review.value_counts(['sentiment']),
           color = colors[:2])
    plt.show()
    return (colors,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Cleaning The Review Text
        The Review text is containing HTML tags and lot of special characters. These characters will not help while classifying the sentiments of the reviews. So let's remove them.

        Let's look at a actula review first
        """
    )
    return


@app.cell
def _(df_review):
    df_review.review[0]
    return


@app.cell
def _(df_review):
    import re
    import html5lib
    from bs4 import BeautifulSoup

    # Function to remove HTML tags
    def strips_html(text):
        soup = BeautifulSoup(text, 'html.parser')
        return soup.get_text()

    # Function to remove brackets
    def remove_squre_brackets(text):
        return re.sub("\[[^]]*\]", "", text)

    # Remove noisy text
    def remove_noisy_text(text):
        text = strips_html(text)
        text = remove_squre_brackets(text)
        return text

    # Cleanning the review text
    df_review['review'] = df_review['review'].apply(remove_noisy_text)
    df_review.review[0]
    return (
        BeautifulSoup,
        html5lib,
        re,
        remove_noisy_text,
        remove_squre_brackets,
        strips_html,
    )


@app.cell
def _(df_review, re):
    # Removing Spacial characters from review
    def remove_special_char(text):
        pattern = r"[^a-zA-Z0-9\s]"
        text = re.sub(pattern, "", text)
        return text

    df_review['review'].apply(remove_special_char)
    df_review.review[0]
    return (remove_special_char,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Stemming

        Now we have cleaned text review. After this we will be stemming each word so that we get the root words and model does not get confused with different form of same words.
        """
    )
    return


@app.cell
def _(df_review):
    import nltk

    # Defining a function for stemming the text
    def stemming_text(text):
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    # Applying stemming function
    df_review['review'] = df_review['review'].apply(stemming_text)

    df_review.review[0]
    return nltk, stemming_text


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Removing Stopwords
        Stopwords are not very useful while classification. They are present in both type of reviews. So, we will remove them.
        """
    )
    return


@app.cell
def _(df_review, nltk):
    from nltk.tokenize.toktok import ToktokTokenizer
    # nltk.download('stopwords')        # Downloading nltk stop words

    # Defining Tokenizer
    tokenizer = ToktokTokenizer()

    # English stopwords
    stopword = nltk.corpus.stopwords.words('english')
    # print(stopword)

    # Function to remove stopwords
    def remove_stopwords(text, is_lower = False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower:
            filtered_tokens = [token for token in tokens if token not in stopword]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword]

        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    # Removing Stopwords
    df_review['review'] = df_review['review'].apply(remove_stopwords)

    df_review.review[0]
    return ToktokTokenizer, remove_stopwords, stopword, tokenizer


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Train Test Split

        Now Splitting the Reviews into train and test set. We will use 75% for training and 25% for testing.
        """
    )
    return


@app.cell
def _(df_review):
    from sklearn.model_selection import train_test_split

    train, test = train_test_split(df_review, test_size= 0.25, random_state= 42)
    print(f'Train set size: {train.shape[0]}')
    print(f'Test set size: {test.shape[0]}')
    return test, train, train_test_split


@app.cell
def _(test, train):
    train_x, train_y = train['review'], train['sentiment']
    test_x, test_y = test['review'], test['sentiment']
    return test_x, test_y, train_x, train_y


@app.cell
def _(mo):
    mo.md("""Once check the distribution of sentiments in train and test set.""")
    return


@app.cell
def _(test_y, train_y):
    print(f"Train Set: {train_y.value_counts()}\n")
    print(f"Test Set: {test_y.value_counts()}")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        In Train and Test set Sentiments are almost equally distributed.

        ### Text Reprentation

        Now we need to represent the text into numbers before buildin the model. We will be converting text to number using TF-IDF.
        """
    )
    return


@app.cell
def _(test_x, train_x):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer()
    train_x_vectors = tfidf.fit_transform(train_x)
    test_x_vectors = tfidf.transform(test_x)
    return TfidfVectorizer, test_x_vectors, tfidf, train_x_vectors


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Model Training

        Now since we have converted text to numeric vectors we can train out classification model. We will be training 3 classification model and select the best model among them. Those 3 models are:

        1. Logistic Regression
        2. Support Vector Machine
        3. Random Forest

        #### **Logistic Regression**
        """
    )
    return


@app.cell
def _(train_x_vectors, train_y):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression()
    lr.fit(train_x_vectors, train_y)
    return LogisticRegression, lr


@app.cell
def _(lr, tfidf):
    print(lr.predict(tfidf.transform(['A good movie'])))
    return


@app.cell
def _(mo):
    mo.md(r"""#### **Support Vector Machine**""")
    return


@app.cell
def _(train_x_vectors, train_y):
    from sklearn.svm import SVC

    svc = SVC(kernel= 'linear')
    svc.fit(train_x_vectors, train_y)
    return SVC, svc


@app.cell
def _(svc, tfidf):
    print(svc.predict(tfidf.transform(['A good movie'])))
    return


@app.cell
def _(mo):
    mo.md(r"""#### **Random Forest**""")
    return


@app.cell
def _(train_x_vectors, train_y):
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators= 100, random_state=42)
    rf.fit(train_x_vectors, train_y)
    return RandomForestClassifier, rf


@app.cell
def _(rf, tfidf):
    print(rf.predict(tfidf.transform(['The movie is bad'])))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Model Evaluation

        Accuracy of each model in test dataset are -
        """
    )
    return


@app.cell
def _(lr, rf, svc, test_x_vectors, test_y):
    print(f"Logistic Regression: {lr.score(test_x_vectors, test_y)}")
    print(f"Support Vector Machine: {svc.score(test_x_vectors, test_y)}")
    print(f"Random Forest: {rf.score(test_x_vectors, test_y)}")
    return


@app.cell
def _(mo):
    mo.md(r"""F1 Score of each model""")
    return


@app.cell
def _(lr, rf, svc, test_x_vectors, test_y):
    from sklearn.metrics import f1_score

    print(f"Logistic Regression: {f1_score(test_y, lr.predict(test_x_vectors,), labels= ['positive','negative'],average=None)}")
    print(f"Support Vector Machine: {f1_score(test_y, svc.predict(test_x_vectors), labels= ['positive','negative'],average=None)}")
    print(f"Random Forest: {f1_score(test_y, rf.predict(test_x_vectors), labels= ['positive','negative'],average=None)}")
    return (f1_score,)


@app.cell
def _(mo):
    mo.md(r"""Classification Report""")
    return


@app.cell
def _(lr, rf, svc, test_x_vectors, test_y):
    from sklearn.metrics import classification_report, confusion_matrix

    print("Logistic Regression: ")
    print(classification_report(
        test_y,
        lr.predict(test_x_vectors),
        labels= ['positive','negative'])
         )
    print("\nSupport Vector Machine: ")
    print(classification_report(
        test_y,
        svc.predict(test_x_vectors),
        labels= ['positive','negative'])
         )
    print("\nRandom Forest: ")
    print(classification_report(
        test_y,
        rf.predict(test_x_vectors),
        labels= ['positive','negative'])
         )
    return classification_report, confusion_matrix


@app.cell
def _(mo):
    mo.md(r"""Performance of SVM is slightly better than LR and Ramdom Forest. Let's see the confusion metrics of SVC model.""")
    return


@app.cell
def _(confusion_matrix, svc, test_x_vectors, test_y):
    confusion_matrix(test_y, svc.predict(test_x_vectors), labels= ['positive', 'negative'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
