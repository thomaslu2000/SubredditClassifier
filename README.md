# SubredditClassifier

An algorithm that learns to distinguish between which Subreddit a post belongs in based on the posts' title.

The program scrapes the top ~1000 posts of all time for each subreddit, cleans and lemmatizes the text, and trains an LSTM to classify the cleaned titles.
