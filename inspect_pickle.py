import pickle

file_path = '/home/taehui/.cache/c2bm/celeba/graph.pkl'
#file_path = '/home/taehui/causally-reliable-cbm/outputs/multirun/2025-12-02/16-20-27_cbm_mlp_real/0/results/c_hat_all.pkl'
#file_path = '/home/taehui/causally-reliable-cbm/outputs/multirun/2025-12-02/16-20-27_cbm_mlp_real/0/results/all_ground_truth_concepts.pkl'
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