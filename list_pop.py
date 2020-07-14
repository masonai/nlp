

a = (['a', 'b','cc','d','e'], [1,2,3,4,5], [6,7,8,9,10])
print(a)

a_len = len(a[0])
idx = 0
for t in range(0, a_len):
    ch = a[0][idx]

    if len(ch) == 1:
        for i in a:
            i.pop(idx)
    else:
        idx += 1

print(a)















