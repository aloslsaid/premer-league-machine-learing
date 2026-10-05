# gamespredction
A machine learning model that predicts English Premier League match outcomes using data from five seasons. The data is cleaned and ordered chronologically, then each team’s statistics from its previous five matches are calculated, including goals, xG, shots, corners, cards, and form. Team names are encoded, and multiple ML algorithms predict 
EPL Match Prediction

This project is about predicting the result of English Premier League matches using machine learning.

I started by collecting and preparing historical Premier League match data.

The first step was understanding the columns and checking the data types.

I wanted to use information that would realistically be available before each match.

Because of that, I focused on previous match performance instead of using future information.

I created features based on the recent performance of each team.

For example, I used statistics such as goals, shots, corners, cards, xG, and recent form.

I calculated these features using previous matches.

This was important because using information from the current or future match could cause data leakage.

After creating the features, I prepared the target variable representing the match result.

I then separated the features from the target.

Instead of randomly mixing all matches, I kept the historical order in mind.

This makes the prediction problem closer to how it would work in the real world.

I trained machine learning models using the prepared features.

I experimented with different approaches and compared their results.

The goal was not just to get a high score, but to understand which features helped the prediction.

I also checked the predictions against the actual match results.

One of the most interesting parts of the project was feature engineering.

The raw match data by itself was not enough, so I had to turn previous results into useful information.

This project also helped me understand the importance of time in machine learning.

A model can look very good if future information accidentally reaches the training data.

Therefore, avoiding data leakage was an important part of the project.

The project gave me practical experience with sports prediction and historical feature engineering.

It also helped me understand how machine learning can be applied to sequential real-world data.

I used

Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

Future Improvements

I would like to improve the features, test more models, add better time-based validation, and build a simple application for predicting upcoming matches.
