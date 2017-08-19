#!/usr/bin/env python3

import numpy as np
from collections import Counter

def percents(data):
    size = len(data)
    counts = Counter(data)
    result = {key:(value/size) for key, value in counts.items()}

    #print('data size   :', size)
    #print('result #    :', counts.most_common())
    #print('result P(X) :', result)
    
    return result

# -----

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

test_X = np.asarray((1,0,1,0))

# --- train --

percent_Y = percents(train_Y)
#classes = set(train_Y)
classes = np.unique(train_Y)

percent_X = {}

for cls in classes:
    subset = train_X[train_Y==cls]
    #print(subset)
    percent_X[cls] = []
    for col in subset.T:
        percent_X[cls].append(percents(col))
        
# --- fit --

result = {}

for cls in classes:
    subset = percent_X[cls]
    p = percent_Y[cls]
    for col, val in zip(subset, test_X):
        p *= col[val]
    result[cls] = p

# --- result ---

for cls, p in result.items():
    print('cls:', cls, '  p: %.5f' % p)
