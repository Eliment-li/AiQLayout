import pickle

# 读取.pkl文件
with open(r'D:\checkpoint\env_runner\module_to_env_connector\class_and_ctor_args.pkl', 'rb') as f:
    data = pickle.load(f)

# 使用读取的数据
print(data)