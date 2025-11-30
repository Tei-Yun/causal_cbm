import pickle

#file_path = '/home/taehui/.cache/c2bm/celeba/graph.pkl'
file_path = '/home/taehui/causally-reliable-cbm/outputs/multirun/2025-11-30/19-06-30_cbm_mlp_real/0/results/c_hat_all.pkl'

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print("Data loaded successfully:")
        print(data)
        print(f"\nType: {type(data)}")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")