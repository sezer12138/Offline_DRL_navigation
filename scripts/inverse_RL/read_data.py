import pickle

# load data from pickle file
with open('human_data.pkl', 'rb') as f:
    data = pickle.load(f)

# display data
for i, item in enumerate(data):
    state, action, reward, new_state, done = item
    if reward > 0.1:
        print(f"Step {i+1}:")
        print(f"State: {state}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"New state: {new_state}")
        print(f"Done: {done}")
        print("\n")
