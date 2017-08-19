#!/usr/bin/env python3

''' temporary version before 'using-functions', 'using-classes' '''

import numpy as np

from collections import Counter

training = np.asarray(
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

result = np.asarray((0,1,1,1,0,1,0,1))

new_sample = np.asarray((1,0,1,0))

# -----

def percents(data):
    size = len(data)
    counts = Counter(data)
    results = {key:(value/size) for key, value in counts.items()}

    print('data size   :', size)
    print('result #    :', counts.most_common())
    print('result P(X) :', results)
    
    return results

# -----

percents(results)

fc = [Counter(column) for column in training.T]
print(fc)

print('features count:')
for n, s in enumerate(new_sample):
    print('  ', n, '|', s, '|', fc[n][s], '| %.3f' % (fc[n][s]/size))

# -----

print('---')
for m, t in rc.most_common(): 
    print(m, t)
    for n, s in enumerate(new_sample):
        print('  ', s, '|', fc[n][s], "| %.3f | %.3f" % (fc[n][s]/size, fc[n][s]/t))    

# -----

print('=== groups ===')

classes = rc.keys()

groups = {}

for cls in classes:
    print('cls:', cls)
    p = training[result == cls]
    s = len(p)
    x = [Counter(col) for col in p.T]

    i = 1
    print(i)
    for n, y in enumerate(new_sample):
        v = x[n][y]
        print(y, ' %i: %i/%i = %.3f' % (y, v, s, v/s))
        i *= v/s
    i *= rk[g]
    print("%.3f" % i)
