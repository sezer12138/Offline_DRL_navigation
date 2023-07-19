import torch
import os

# list of files to combine
files = ["collect_data.pth", "dwa_data.pth", 'dwa_data1.pth', 'dwa_data2.pth', 'dwa_data3.pth', 'dwa_data4.pth', 'dwa_data5.pth', 'dwa_data6.pth', 'dwa_data7.pth']

# initialize empty tensors
state_tensor = torch.empty(0)
action_tensor = torch.empty(0)
reward_tensor = torch.empty(0)
new_state_tensor = torch.empty(0)
done_tensor = torch.empty(0)

# iterate over files
for file in files:
    if os.path.isfile(file):
        # load tensors from file
        curr_state_tensor, curr_action_tensor, curr_reward_tensor, curr_new_state_tensor, curr_done_tensor = torch.load(file)
        
        # concatenate tensors
        state_tensor = torch.cat((state_tensor, curr_state_tensor))
        action_tensor = torch.cat((action_tensor, curr_action_tensor))
        reward_tensor = torch.cat((reward_tensor, curr_reward_tensor))
        new_state_tensor = torch.cat((new_state_tensor, curr_new_state_tensor))
        done_tensor = torch.cat((done_tensor, curr_done_tensor))

# print the tensors
print("State Tensor Shape:", state_tensor.shape)
print("Action Tensor Shape:", action_tensor.shape)
print("Reward Tensor Shape:", reward_tensor.shape)
print("New State Tensor Shape:", new_state_tensor.shape)
print("Done Tensor Shape:", done_tensor.shape)

# save the combined tensors into a single file
torch.save((state_tensor, action_tensor, reward_tensor, new_state_tensor, done_tensor), 'combined_data.pth')
