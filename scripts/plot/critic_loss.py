import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
with open('/home/sezer/Downloads/DDPG.json', 'r') as f:
    data1 = json.load(f)

with open('/home/sezer/Downloads/SAC.json', 'r') as f:
    data2 = json.load(f)

with open('/home/sezer/Downloads/TD3.json', 'r') as f:
    data3 = json.load(f)


steps1, loss1 = zip(*[(item[1], item[2]) for item in data1])
steps2, loss2 = zip(*[(item[1], item[2]) for item in data2])
steps3, loss3 = zip(*[(item[1], item[2]) for item in data3])


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Apply smoothing to the loss data
smoothed_loss1 = smooth(np.array(loss1), 50)  # you can adjust the window size 50
smoothed_loss2 = smooth(np.array(loss2), 50)
smoothed_loss3 = smooth(np.array(loss3), 50)

plt.plot(steps1, smoothed_loss1, label='DDPG')
plt.plot(steps2, smoothed_loss2, label='SAC')
plt.plot(steps3, smoothed_loss3, label='TD3')

# Add legend and title
plt.legend()
plt.title('Critic Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')

plt.savefig('Critic_loss_smoothed.png', dpi=600)

# Show plot
plt.show()