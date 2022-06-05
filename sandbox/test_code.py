
# Scan Labware
rows = 8
columns = 12

pos_list = []

for j in range(rows):
    for i in range(columns):
        if j % 2 == 1:
            i = columns - i - 1
        print(i, j)
        pos_list.append([i, j])

print(pos_list)

[i, j] = pos_list[0]
print(i, j)
