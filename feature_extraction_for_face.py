import math
import numpy as np
import pandas

f = open('(20200408)collectimageface_for_code.csv', 'r')
d = f.readlines()

lf = open('신뢰도_결과.CSV', 'r')
total_l = lf.readlines()

label = []
for l in total_l[1:]:
    l = l.strip().split(',')[1:]
    l = list(map(float, l))
    max_value = max(l)
    label.append(l.index(max_value))

# print(label)

ppl = []
ppl_dist = []
for data_ in d:
    data = data_.strip()
    data = data[1:-1].split("|")

    person = []
    person_dist = []
    for person_data in data:
        pd = person_data.split(",")
        person.append([int(pd[0]), int(pd[1])])

    for i, dp_1 in enumerate(person):
        for j, dp_2 in enumerate(person):
            if i >= j:
                pass
            else:
                dist = math.sqrt((dp_1[0] - dp_2[0]) ** 2 + (dp_1[1] - dp_2[1]) ** 2)
                person_dist.append(dist)

    ppl.append(person)
    ppl_dist.append(person_dist)

attr = []
for i in range(1, 69):
    for j in range(1, 69):
        if i >= j:
            pass
        else:
            attr.append(str(i) + "to" + str(j))
attr.append("label")

# print(ppl)
# print(ppl_dist)

ppl_dist = np.array(ppl_dist)
label = np.array(label)
attr = np.array(attr)

# attr = np.expand_dims(attr, axis=0)
label = np.expand_dims(label, axis=0)
label = np.transpose(label)
dataset = np.concatenate((ppl_dist, label), axis=1)
# dataset = np.concatenate((attr, dataset), axis=0)
# np.savetxt("face_distance_feature_test.csv", dataset, delimiter=",")
df = pandas.DataFrame(dataset, columns=attr)
# print(df)
df.to_csv("face_distance_features.csv", index=None)
