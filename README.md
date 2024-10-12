# NLPSentimentAnalyser

## Sentiment analysis on movie reviews using natural language processing

Just as how humans are able to identify the emotion behind a piece of text given its phrasing, sentiment analysis aims to use Natural Language Processing techniques to mimic this and produce an inferred sentiment of the text. Provided training data this project is able to accurately (87.5%) determine the sentiment of your movie review using 3 different models outlined below.

### Naive Bayes Approach

Naıve Bayes is a classification method built upon Bayes’ theorem which calculates the probability that an input feature belongs to a specific class based on prior knowledge. It makes the assumption that each input feature is independent of the other, which in the case of natural language is not true. However, despite this a Naıve Bayes approach is still able to accurately classify text into classes

### Logistic Regression Approach

Logistic regression is a binary classification model which returns an output be- tween 0 and 1 when predicting the polarity of the context. It evaluates based on a threshold (usually 0.5) so if the value of the function returns anything above the threshold it becomes one otherwise 0. To achieve the bounding of the output between 0 and 1 logistic regression relies on the sigmoid function. Also in Logistic Regression are weights which are optimised through- out the training process through the loss function. The loss function essentially optimises the weights that are applied to the input feature based on how far the model estimated the feature to be from its actual label.

### SVM Approach

An SVM or Support Vector Machine is also very effective for classification prob- lems. It produces classification for a dataset by generating a linear separation between the items where above the line represents one class and below another. The goal of the SVM is to produce this separation between the classes whilst being as far away as possible from the closest data point. The closer a data point is the boundary the more it influences it. An SVM has its main benefit in being able to deal with data that is not linearly separable by using a hyperplane rather than a line to create the boundary.

### How to run the program

Provide two sets of text files containing positive and negative examples for training, development and testing data which will be created in the 80:10:10 split. Run the code.
