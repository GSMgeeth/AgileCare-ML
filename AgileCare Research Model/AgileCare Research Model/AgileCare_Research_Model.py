import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from matplotlib import cm
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

stories = pd.read_excel('story data sample - edited.xlsx')

# print(stories.head(75))
# print(stories.shape)

# stories.drop('label', axis=1).hist(bins=30, figsize=(9,9))
# pl.suptitle('Histogram of story-story point-hours')
# plt.savefig('story_hist_edited')

# stories.drop('label', axis=1).plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for each input variable')
# plt.savefig('story_box_terminal_edited')

feature_stories = ['points', 'hours']
X = stories[feature_stories]
y = stories['label']

# cmap = cm.get_cmap('gnuplot')
# scatter = scatter_matrix(X, c=y, marker='o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
# plt.suptitle('Scatter-matrix for each input variable')
# plt.savefig('story_scatter_matrix_edited')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# imputer = SimpleImputer(missing_value=math.nan, strategy='mean')
# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)

# Logistic Regression
# logreg = LogisticRegression()

# logreg.fit(X_train, y_train)

# print('Accuracy of Logistic regression classifier on training set: {:.2f}'
#         .format(logreg.score(X_train, y_train)))

# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#         .format(logreg.score(X_test, y_test)))

# pred = logreg.predict(X_test)

# Decision Tree
# clf = DecisionTreeClassifier()

# clf.fit(X_train, y_train)

# print('Accuracy of Decision Tree classifier on training set: {:.2f}'
#      .format(clf.score(X_train, y_train)))
# print('Accuracy of Decision Tree classifier on test set: {:.2f}'
#      .format(clf.score(X_test, y_test)))

# pred = clf.predict(X_test)

# Gaussian Naive Bayes
# gnb = GaussianNB()

# gnb.fit(X_train, y_train)

# print('Accuracy of GNB classifier on training set: {:.2f}'
#      .format(gnb.score(X_train, y_train)))
# print('Accuracy of GNB classifier on test set: {:.2f}'
#      .format(gnb.score(X_test, y_test)))

# pred = gnb.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

print('\nAccuracy of K-NN classifier on training set: {:.2f}'
    .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f}'
    .format(knn.score(X_test, y_test)))

pred = knn.predict(X_test)

print('\nX_test')
print(X_test)

print('\npred')
print(pred)

print('\ny_test')
print(y_test)
