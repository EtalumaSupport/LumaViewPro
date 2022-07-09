import json

with open('./data/objectives.json', "r") as read_file:
    objectives = json.load(read_file)

print(list(objectives.keys()))

# for i in objectives:
#     print(i)
#     print(objectives[i])
#     objectives[i]['ID']= i
#     print(objectives[i])
