# AiQLayout

under development


The code is tested on server with 1 GPU or 0 GPU(only CPU) 
If wish to try multi GPU to train, please see https://docs.ray.io/en/latest/rllib/package_ref/algorithm-config.html#rllib-config-learners

And check enhance_config() function to make sure the config is correct



# Chip Size
We set three chip size:
- 10x10 for 0  < qubits number <= 15
- 12*12 for 15 < qubits number <= 25
- 15*15 for 25 < qubits number <= 49

If there are corrupted qubits in the chip, the qubits number above is also reduced