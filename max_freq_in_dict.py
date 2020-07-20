dict = {'a': 5, 'b':10, 'c':3}
best = max(dict, key=dict.get)
print(best