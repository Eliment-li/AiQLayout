path_len = [1,2,3,4]
rewards = {}
for i in range(4 ):
    r = (5 - path_len[i])
    rewards.update({f'agent_{i + 1}': r})
print('rewards :', rewards)
