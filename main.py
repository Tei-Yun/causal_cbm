import random
import numpy as np
import torch
import os
import warnings
import hydra
import pickle
# import jpype
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

# pytorch lightning
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

# data loading
from src.data.dataset_block import get_dataset

# causal discovery
from src.causal_discovery.causal_discovery_block import causal_discovery

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
#from src.metrics import hamming_distance
from src.plots import maybe_plot_graph
from src.utils import get_intervention_policy, remove_cycles, remove_problematic_edges
from src.utils import clean_empty_configs, update_config_from_data, maybe_update_config_with_graph
from src.utils import finetune_model

# Suppress specific warning
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")
    
def seed_everything(seed: int):
    print(f"Seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="conf", config_name="my_sweep", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # various preliminaries, it set the seed for reproducibility
    torch.set_num_threads(cfg.get("num_threads", 1))
    seed_everything(cfg.get("seed"))
    os.mkdir('results')
    with open_dict(cfg): cfg.update(device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {cfg.device} device")

    # adjust config
    cfg = clean_empty_configs(cfg) #causal_discovery, llm, rag 관련 config 참고 하여 초기화 => discovery 관련 설정

    # instantiate the dataset, split into train, val, test
    # preprocess all of them and save the preprocessed dataset
    dataset, true_graph, dataset_directory = get_dataset(cfg) #전처리 + true_graph 얻기(discovery or cache에서 로드)

    # get the causal graph
    if cfg.dataset.load_true_graph:
        graph = true_graph
    else:
        if cfg.dataset.load_graph:
            with open(os.path.join(dataset_directory, "graph.pkl"), 'rb') as f:
                graph = pickle.load(f)
        else:
            # estimate causal graph with causal structural learning algorithms
            predicted_graph = causal_discovery(cfg, dataset, true_graph)
            if true_graph is not None:
                hamming = hamming_distance(true_graph, predicted_graph)
                print('(after CD) structural hamming distance: ', hamming)    


            graph = predicted_graph

            # save graph
            with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
                pickle.dump(graph, f)

    #y_index 는 DAG에서 task 노드 위치 => 반드시 마지막 노드여야 한다고 assert로 강제하는 코드
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1

    #tei 수정 11/29
    cnf_int_policy = None
    cnf_bundle_path = None
    if cfg.policy == 'none':
        interv_policy = []
        ip_names = []
        print(f'Intervention policy: None (Disabled)')
    elif cfg.policy in ['cnf_cf', 'cnf_int', 'cnf_prob_cf']:
        cnf_int_policy = cfg.policy
        cnf_bundle_path = cfg.cnf_bundle_path
        interv_policy = []
        ip_names = []
        print(f'CNF intervention policy: {cfg.policy}')

    else:
        interv_policy, ip_names = get_intervention_policy(cfg.policy, graph, true_graph, y_index)
        print('intervention policy:', interv_policy) #ex)interv_policy = [[0], [2], [4], [1], [3], [5]]
        print('intervention policy names:', ip_names)

    # update config based on the dataset
    # e.g., set input and output size of the model
    cfg = update_config_from_data(cfg, dataset)
    cfg = maybe_update_config_with_graph(cfg, graph, interv_policy, [cnf_int_policy, cnf_bundle_path])
    
    ############ model block ########################################################################################
    [dataset.data[split].register_graph(graph) for split in dataset.data]
    train_dataloader = DataLoader(dataset.data['train'], 
                                  batch_size=cfg.dataset.batch_size, 
                                  collate_fn=static_graph_collate,
                                  num_workers=cfg.dataset.num_workers)
    val_dataloader = DataLoader(dataset.data['val'], 
                                batch_size=cfg.dataset.batch_size, 
                                collate_fn=static_graph_collate,
                                num_workers=cfg.dataset.num_workers)
    test_dataloader = DataLoader(dataset.data['test'], 
                                 batch_size=cfg.dataset.batch_size, 
                                 collate_fn=static_graph_collate,
                                 num_workers=cfg.dataset.num_workers)
    
    #tei 추가 11/29
    # [추가] 전체 데이터셋(Train/Val/Test)의 Ground Truth Concept 추출 및 저장
    print("Extracting Ground Truth Concepts from ALL splits for CNF training...")
    
    splits = ['train', 'val', 'test']
    dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    
    # [수정] 두 가지 버전의 딕셔너리 생성
    all_concepts_only = {}      # Concept만 저장
    all_concepts_with_task = {} # Concept + Task 저장

    for split_name, loader in zip(splits, dataloaders):
        print(f"Processing {split_name} split...")
        split_c_list = []
        split_y_list = [] # [Add] y 수집용 리스트
        
        # 모델 학습에 영향 없도록 torch.no_grad() 사용
        with torch.no_grad():
            for batch in loader:
                if 'c' in batch:
                    split_c_list.append(batch['c'].cpu())
                if 'y' in batch: # [Add] y 수집
                    split_y_list.append(batch['y'].cpu())
        
        if split_c_list:
            # 하나의 텐서로 병합 [N_samples, N_concepts]
            c_tensor = torch.cat(split_c_list, dim=0)
            
            # 1. Concept Only 저장
            all_concepts_only[split_name] = c_tensor
            print(f"  > {split_name}: {c_tensor.shape} (Concepts only)")

            # 2. Concept + Task 저장
            if split_y_list:
                y_tensor = torch.cat(split_y_list, dim=0)
                # 차원 맞추기 (y가 1차원이면 2차원으로 확장)
                if y_tensor.ndim == 1:
                    y_tensor = y_tensor.unsqueeze(1)
                
                # c와 y를 합쳐서 저장 (보통 y가 마지막에 옴)
                # [N, C] + [N, 1] -> [N, C+1]
                combined_tensor = torch.cat([c_tensor, y_tensor], dim=1)
                all_concepts_with_task[split_name] = combined_tensor
                print(f"  > {split_name}: {combined_tensor.shape} (Concepts + Task)")
            else:
                # y가 없으면 concept only와 동일하게 저장하거나 생략
                all_concepts_with_task[split_name] = c_tensor
        else:
            print(f"  > {split_name}: No concepts found.")

    # 파일로 저장 (두 가지 버전)
    save_path_only = "results/all_ground_truth_concepts_only.pkl"
    with open(save_path_only, 'wb') as f:
        pickle.dump(all_concepts_only, f)
    
    save_path_task = "results/all_ground_truth_concepts_with_task.pkl"
    with open(save_path_task, 'wb') as f:
        pickle.dump(all_concepts_with_task, f)
    
    print(f"Saved Ground Truth (Concepts Only) to {save_path_only}")
    print(f"Saved Ground Truth (Concepts + Task) to {save_path_task}")
    # ---------------------------------------------------------

    
    print("DEBUG engine cfg:", cfg.engine)
    print("Trying to instantiate:", cfg.engine._target_)
    engine = instantiate(cfg.engine)

    # tei 수정 12/4: 학습 모드에 따른 분기 처리
    # engine.model에서 training_mode를 확인 (기본값 'joint')
    training_mode = getattr(engine.model, 'training_mode', 'joint')
    print(f"Training Mode detected: {training_mode}")

    # [추가] 체크포인트 경로 확인 (cfg에 ckpt_path 항목이 있다고 가정)
    # 예: python main.py ckpt_path="checkpoints/best_model.ckpt"
    # [추가] 설정에서 test_only 플래그와 체크포인트 경로 확인
    test_only = cfg.get("test_only", False)
    ckpt_path = cfg.get("ckpt_path", None)
    
    trainer = None # finally 블록 에러 방지용 초기화

    try:
        # [Case 0] Test Only Mode (학습 건너뛰기)
        if test_only:
            print(f"Running in TEST ONLY mode.")
            
            if ckpt_path is None:
                print("WARNING: test_only=True but ckpt_path is None. Testing with initialized weights (random).")
            else:
                print(f"Loading model from checkpoint: {ckpt_path}")

                # [수정] strict=False로 로드하여 CNF 등 새로 추가된 모듈이 체크포인트에 없어도 에러 무시
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
                # Lightning 체크포인트는 보통 'state_dict' 키 안에 가중치가 있음
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                
                missing_keys, unexpected_keys = engine.load_state_dict(state_dict, strict=False)
                print(f"Checkpoint loaded with strict=False.")
                print(f"  > Missing keys (initialized randomly): {len(missing_keys)}")
                if missing_keys:
                    print(f"    Example: {missing_keys[:3]} ...")
                print(f"  > Unexpected keys (ignored): {len(unexpected_keys)}")


            # 테스트용 Trainer 생성
            trainer = Trainer(cfg)
            
            # # 테스트 수행 (ckpt_path가 있으면 해당 가중치 로드)
            # trainer.test(engine, test_dataloader, ckpt_path=ckpt_path)
            
            # 테스트 수행 (위에서 수동으로 로드했으므로 ckpt_path=None 전달)
            trainer.test(engine, test_dataloader, ckpt_path=None)
            
            trainer.logger.finalize("success")

        # [Case 1 & 2] Training Mode (학습 진행)
        else:
            # [Case 1] Joint Training
            if training_mode == 'joint':
                trainer = Trainer(cfg)
                trainer.logger.log_hyperparams(parse_hyperparams(cfg))
                trainer.fit(engine, train_dataloader, val_dataloader)
            
            # [Case 2] Stepwise / Independent Training (2단계 학습)
            else:
                # --- Stage 1: Encoder 학습 ---
                print(f"\n=== Starting {training_mode} training: Stage 1 (Encoder) ===")
                engine.model.train_stage = 'encoder'
                trainer_enc = Trainer(cfg)
                trainer_enc.logger.log_hyperparams(parse_hyperparams(cfg))
                trainer_enc.fit(engine, train_dataloader, val_dataloader)
                
                # --- Stage 2: Decoder 학습 ---
                print(f"\n=== Starting {training_mode} training: Stage 2 (Decoder) ===")
                engine.model.train_stage = 'decoder'
                
                # Encoder 파라미터 Freeze
                if hasattr(engine.model, 'c_mlp'):
                    print("Freezing Concept Encoder weights...")
                    for param in engine.model.c_mlp.parameters():
                        param.requires_grad = False
                
                trainer = Trainer(cfg) 
                trainer.fit(engine, train_dataloader, val_dataloader)

            # ---- finetune the encoder (eventually)
            if cfg.dataset.loader.ftune_size > 0: 
                trainer, engine = finetune_model(cfg, engine, dataset)
            
            # ----- test after training
            # 학습 직후에는 가장 성능이 좋았던 체크포인트('best')를 로드하여 테스트
            if trainer is not None:
                trainer.test(engine, test_dataloader, ckpt_path='best')
                trainer.logger.finalize("success")
            
    finally:
        if trainer is not None and isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
    ############################################################################################


if __name__ == "__main__":
    main()
    print('done')