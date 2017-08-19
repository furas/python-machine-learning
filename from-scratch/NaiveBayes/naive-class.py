import numpy as np
from collections import Counter

class NaiveBayes(object):

    def __init__(self):
        self.train_X = []
        self.train_Y = []

        self.test_X = []

        self.percent_X = {}
        self.percent_Y = {}
        self.classes = []
        
    def percents(self, data):
        size = len(data)
        counts = Counter(data)
        result = {key:(value/size) for key, value in counts.items()}

        return result

    def fit(self, train_X, train_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        
        #self.classes = set(train_Y)
        self.classes = np.unique(train_Y)

        self.percent_Y = self.percents(train_Y)
        self.percent_X = {}

        for cls in self.classes:
            subset = train_X[train_Y==cls]
            #print(subset)
            self.percent_X[cls] = []
            for col in subset.T:
                self.percent_X[cls].append(self.percents(col))
        
    def predict(self, test_X):
        self.text_X = test_X

        self.results = []

        result = []
        for row in test_X:
        
            temp = []

            for cls in self.classes:
                subset = self.percent_X[cls]
                p = self.percent_Y[cls]
                for col, val in zip(subset, row):
                    if val in col:
                        p *= col[val]
                    else:
                        p = 0
                        break
                temp.append((p,cls))
            
            self.results.append(temp)
            result.append(max(temp)[1])
            
        return result
        
# --- main ---

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

nb = NaiveBayes()

nb.fit(train_X, train_Y)

result = nb.predict(test_X)
print(test_X, result)
print(nb.results)

for row in nb.results:
    print(row)
