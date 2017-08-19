import numpy as np
from sklearn.naive_bayes import GaussianNB

train_X = np.asarray(
    (
        (1,0,1,1),
        (1,1,0,0),
        (1,0,2,1),
        (0,1,1,1),
        (0,0,0,0),
        (0,1,2,1),
        (0,1,2,0),
        (1,1,1,1)
    )
)

train_Y = np.asarray((0,1,1,1,0,1,0,1))
train_Y = np.asarray((0,2,1,1,0,1,0,2))

test_X = np.asarray(
    (
        (1,0,1,0),
        (1,0,1,1),
        (1,1,1,0),
    )
)   

# --- main ---

clf = GaussianNB()
clf.fit(train_X, train_Y)
print(clf.predict(test_X))

print(clf.score(train_X, train_Y))
