import json

with open('./data/objectives.json', "r") as read_file:
    objectives = json.load(read_file)

# for i in range(len(objectives)):
#     print(objectives[i])

for i in objectives:
    print(i)
    print(objectives[i])
    objectives[i]['ID']= i
    print(objectives[i])
