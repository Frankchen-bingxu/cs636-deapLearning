import json
import numpy as np
import matplotlib.pyplot as plt

fp =  open('extra_8_action.json', 'r',encoding='utf-8')
s = json.load(fp)
print(s)
store = []
pro = []
time = []
for i in range(len(s)):
    if s[i]["label"] == "reading book":
        # print(s[i])
        store.append(s[i])
    i += 1
print(store)
seen = set()
new_l = []
for d in store:
    t = tuple(d.items())
    if t not in seen:
        seen.add(t)
        new_l.append(d)
print(new_l)

for item in range(len(new_l)):
    time.append(new_l[item]['time']*140/24000)
    pro.append(new_l[item]['pro'])


time.insert(2,71)
time.insert(3,105)
pro.insert(2,0.915)
pro.insert(3,0.915)


x=np.arange(0,2633)
l1=plt.plot(time,pro,'r--',label='reading book tiem/pro')

plt.plot(time,pro,'ro-')
plt.title('Time/Pro')
plt.xlabel('row')
plt.ylabel('column')
plt.legend()
plt.show()
