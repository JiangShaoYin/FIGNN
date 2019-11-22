import json
from collections import Counter

data_dict_train = {}
attr_list = []
print()

with open('train.csv', 'r') as f:
    for row, line in enumerate(f):
        if row // 100000 == 1:
            break;
        if row == 0:
            attr_list = line.strip().split(',')
            for jdx, attribute in enumerate(attr_list):
                data_dict_train[attribute] = []
        else:
            temp = line.strip().split(',')
            for jdx, feature in enumerate(temp):
                data_dict_train[attr_list[jdx]].append(temp[jdx])

with open('alltraindata.json', 'w') as f:
    f.write(json.dumps(data_dict_train))

attr_list = data_dict_train.keys()

user_id = data_dict_train['device_id']
count_user = Counter(user_id)

data_list_by_user = []
user2id = dict(zip(*(count_user.keys(), range(len(count_user)))))
user_idx = 2
user_list = list(count_user.keys()) # 根据device_id，10W条记录里面有7201个user
with open("user_list.json", 'w') as f:
    f.write(json.dumps(user_list))

little_user_list = user_list[user_idx * int(len(user_list)/10): # 从idx = 1440开始, 截取部分用户，720名
                             (user_idx + 1) * int(len(user_list)/10)]
user2id = dict(zip(*(little_user_list, range(len(little_user_list))))) # 同一个device是一个人，重新编号

data_list_by_user = []
for _ in user2id:
    data_list_by_user.append([])
for idx in range(len(data_dict_train['device_id'])):
    if data_dict_train['device_id'][idx] in user2id:
        cur_device_id = data_dict_train['device_id'][idx] # cur_device_id = 4516d70e
        useridx = user2id[cur_device_id] # useridx = 0
        data_list_by_user[useridx].append(tuple([data_dict_train[attr][idx] for attr in attr_list]))
        if idx % 100000 == 0:
            print(idx)



data_list_by_user = list(filter(lambda x: len(x) > 3, data_list_by_user))

all_length = len(data_list_by_user)
for user_jdx in range(10):
    with open('data_list_by_user_' + str(user_idx) + '_' + str(user_jdx) + '.json', 'w') as f:
        f.write(json.dumps(data_list_by_user[int(user_jdx * (all_length * 0.1)): int((user_jdx + 1) * (all_length * 0.1))]))
