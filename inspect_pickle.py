import pickle
import os
import torch  # 텐서가 포함된 경우를 위해 임포트

# 대상 디렉토리 설정
target_dir = '/home/taehui/causally-reliable-cbm/outputs/multirun/2025-12-23/15-30-46_cbm_mlp_asia_indep_cnf_prob_cf/0/results'

# 출력하고 싶은 특정 파일 목록
target_files = [
    'c_accuracy.pkl',
    'cnf_cf_interventions_on_y.pkl',
    #'level_interventions_on_c.pkl',
    'level_interventions_on_y.pkl',
    'single_c_interventions_on_y.pkl',
    #'y_accuracy.pkl'
    'cbm_cumulative_interventions_on_c.pkl',
    'cbm_cumulative_interventions_on_y.pkl',
    'cnf_cf_cumulative_interventions_on_y.pkl',
    'cnf_cf_cumulative_interventions_on_c.pkl',
]

print(f"Inspecting specific .pkl files in: {target_dir}\n")

if os.path.exists(target_dir):
    # 디렉토리 내 파일 중 target_files에 있는 것만 필터링
    files = sorted([f for f in os.listdir(target_dir) if f in target_files])
    
    if not files:
        print("No matching .pkl files found.")
    
    for filename in files:
        file_path = os.path.join(target_dir, filename)
        print("="*80)
        print(f"FILE: {filename}")
        print("="*80)
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                print(f"Type: {type(data)}")
                
                # 데이터 타입에 따라 보기 좋게 출력
                if isinstance(data, dict):
                    print(f"Keys: {list(data.keys())}")
                    for k, v in data.items():
                        print(f"\n  [Key: {k}]")
                        if hasattr(v, 'shape'): # Tensor나 Numpy array인 경우
                            print(f"    Shape: {v.shape}")
                            print(f"    Value: {v}")
                        else:
                            print(f"    Value: {v}")
                elif hasattr(data, 'shape'):
                    print(f"Shape: {data.shape}")
                    print(f"Content: {data}")
                else:
                    print(f"Content: {data}")
                    
        except Exception as e:
            print(f"Error reading file: {e}")
        print("\n")
else:
    print(f"Directory not found: {target_dir}")