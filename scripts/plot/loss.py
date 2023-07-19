import json
import numpy as np
import matplotlib.pyplot as plt

# 读取 JSON 文件
with open('/home/sezer/Downloads/DDPG.json', 'r') as f:
    actor_ddpg = json.load(f)

with open('/home/sezer/Downloads/SAC.json', 'r') as f:
    actor_sac = json.load(f)

with open('/home/sezer/Downloads/TD3.json', 'r') as f:
    actor_td3 = json.load(f)

with open('/home/sezer/Downloads/DDPG(1).json', 'r') as f:
    critic_ddpg = json.load(f)

with open('/home/sezer/Downloads/SAC(1).json', 'r') as f:
    critic_sac = json.load(f)

with open('/home/sezer/Downloads/TD3(1).json', 'r') as f:
    critic_td3 = json.load(f)

steps1, loss1 = zip(*[(item[1], item[2]) for item in actor_ddpg])
steps2, loss2 = zip(*[(item[1], item[2]) for item in actor_sac])
steps3, loss3 = zip(*[(item[1], item[2]) for item in actor_td3])

steps4, loss4 = zip(*[(item[1], item[2]) for item in critic_ddpg])
steps5, loss5 = zip(*[(item[1], item[2]) for item in critic_sac])
steps6, loss6 = zip(*[(item[1], item[2]) for item in critic_td3])

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

loss4 = smooth(np.array(loss4), 50)
loss5 = smooth(np.array(loss5), 50)
loss6 = smooth(np.array(loss6), 50)

plt.rcParams["figure.figsize"] = (6, 8)
fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(steps1, loss1, label='DDPG')
ax1.plot(steps2, loss2, label='SAC')
ax1.plot(steps3, loss3, label='TD3')

ax1.set_title('Actor Loss')
ax1.set_ylabel('Loss')

ax2.plot(steps4, loss4, label='DDPG')
ax2.plot(steps5, loss5, label='SAC')
ax2.plot(steps6, loss6, label='TD3')

ax2.set_title('Critic Loss')

ax1.legend(loc='upper right', bbox_to_anchor=(1.1, 1.2))

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('Offline_loss.png', dpi=600)

# 显示图形
plt.show()