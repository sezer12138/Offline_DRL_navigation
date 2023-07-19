import torch

# load data from file
state_tensor, action_tensor, reward_tensor, new_state_tensor, done_tensor = torch.load('new_data1.pth')

# print the tensors
print("State Tensor Shape:", state_tensor.shape)
print("Action Tensor Shape:", action_tensor.shape)
print("Reward Tensor Shape:", reward_tensor.shape)
print("New State Tensor Shape:", new_state_tensor.shape)
print("Done Tensor Shape:", done_tensor.shape)
