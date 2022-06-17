#
# # Scan Labware
# rows = 8
# columns = 12
#
# pos_list = []
#
# for j in range(rows):
#     for i in range(columns):
#         if j % 2 == 1:
#             i = columns - i - 1
#         print(i, j)
#         pos_list.append([i, j])
#
# print(pos_list)
#
# [i, j] = pos_list[0]
# print(i, j)


# import time
# a = time.time()
# time.sleep(2)
# b = time.time()
# print(b-a)




# import numpy as np
#
# sec_remaining = 9284654.232
# min_remaining = sec_remaining / 60
# hrs_remaining = min_remaining / 60
#
# hrs = np.floor(hrs_remaining)
# minutes = np.floor((hrs_remaining - hrs)*60)
#
# hrs = '%d' % hrs
# minutes = '%02d' % minutes
#
# print(hrs + ':' + minutes + ' remaining')
# #
# # print('%02d' % hrs)
# # print('%02d' % minutes)


try:
    print(qwerty)
except:
    pass
