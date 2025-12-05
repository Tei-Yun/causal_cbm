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

# graph completion block
#tei 수정 11/29 => llm 안쓸거임
#from src.completion.completion_block import complete_graph_with_llm

# training and utils
from src.trainer import Trainer
from src.hydra_parsing import parse_hyperparams
from src.data.utils import static_graph_collate
from src.metrics import hamming_distance
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

            # complete the causal graph with LLM and RAG

            #tei 수정 11/29 => llm 안쓸거임
            # completed_graph = complete_graph_with_llm(cfg, predicted_graph, cfg.dataset.name)
            # if true_graph is not None:
            #     hamming = hamming_distance(true_graph, completed_graph)
            #     print('(after LLM + RAG) structural hamming distance: ', hamming)
            # graph = completed_graph

            graph = predicted_graph

            # save graph
            with open(os.path.join(dataset_directory, "graph.pkl"), 'wb') as f:
                pickle.dump(graph, f)

    # fix the graph
    # (part 1): remove bidirected and undirected edges + add virtual nodes
    # edge can only be directed at this stage, the following function is just here in 
    # case the CD + LLM + RAG pipeline is modified and could produce bidirected or undirected edges


    #discovery 따로 할거라면, 이쪽 부분도 없애도 될듯
    graph, dataset = remove_problematic_edges(graph, dataset)

    #y_index 는 DAG에서 task 노드 위치 => 반드시 마지막 노드여야 한다고 assert로 강제하는 코드
    y_index = list(graph.index).index(dataset.y_info['names'][0]); assert y_index == len(graph) - 1

    
    # (part 2): remove cycles
    graph = remove_cycles(graph, y_index)

    if true_graph is not None:
        hamming = hamming_distance(true_graph, graph)
        print('(after fix) structural hamming distance: ', hamming)
    maybe_plot_graph(graph, 'fixed_graph')

    # use the graph to define an intervention policy at test time
    #tei 수정 11/29
    cnf_int_policy = None
    if cfg.policy == 'none':
        interv_policy = []
        ip_names = []
        print(f'Intervention policy: None (Disabled)')
    elif cfg.policy  in ['cnf_int', 'cnf_cf']:
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
    # [Add] 전체 데이터셋(Train/Val/Test)의 Ground Truth Concept 추출 및 저장
    print("Extracting Ground Truth Concepts from ALL splits for CNF training...")
    
    splits = ['train', 'val', 'test']
    dataloaders = [train_dataloader, val_dataloader, test_dataloader]
    
    all_concepts_data = {}

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
            
            # y도 병합 [N_samples] or [N_samples, 1]
            if split_y_list:
                y_tensor = torch.cat(split_y_list, dim=0)
                # 차원 맞추기 (y가 1차원이면 2차원으로 확장)
                if y_tensor.ndim == 1:
                    y_tensor = y_tensor.unsqueeze(1)
                
                # c와 y를 합쳐서 저장 (보통 y가 마지막에 옴)
                # [N, C] + [N, 1] -> [N, C+1]
                combined_tensor = torch.cat([c_tensor, y_tensor], dim=1)
                all_concepts_data[split_name] = combined_tensor
                print(f"  > {split_name}: {combined_tensor.shape} (Concepts + Task)")
            else:
                all_concepts_data[split_name] = c_tensor
                print(f"  > {split_name}: {c_tensor.shape} (Concepts only)")
        else:
            print(f"  > {split_name}: No concepts found.")

    # 파일로 저장
    print("Current working directory:", os.getcwd())
    os.makedirs('./results', exist_ok=True)
    save_path = "./results/all_ground_truth_concepts.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(all_concepts_data, f)
    
    print(f"Saved all ground truth concepts to {save_path}")
    print("Structure: {'train': Tensor, 'val': Tensor, 'test': Tensor}")
    # ---------------------------------------------------------

    
    print("DEBUG engine cfg:", cfg.engine)
    print("Trying to instantiate:", cfg.engine._target_)
    
    # Check if we should skip training and load from checkpoint
    test_only = cfg.get("test_only", False)
    checkpoint_path = cfg.get("checkpoint_path", None)
    
    try:
        engine = instantiate(cfg.engine)
        print("Engine instantiated successfully")
    except Exception as e:
        print(f"ERROR: Failed to instantiate engine: {e}")
        import traceback
        traceback.print_exc()
        raise  # 예외를 다시 발생시켜 프로그램 종료

    print("here")
    try:
        trainer = Trainer(cfg)
        trainer.logger.log_hyperparams(parse_hyperparams(cfg))
        
        if test_only:
            # Test only mode: skip training and load from checkpoint
            if checkpoint_path:
                print(f"Running test only with checkpoint: {checkpoint_path}")
                # PyTorch Lightning이 checkpoint에서 자동으로 가중치를 로드합니다
                trainer.test(engine, test_dataloader, ckpt_path=checkpoint_path)
            else:
                print("WARNING: test_only=True but checkpoint_path not provided. Using 'best' checkpoint.")
                trainer.test(engine, test_dataloader, ckpt_path='best')
        else:
            # Normal training mode
            # ---- train
            trainer.fit(engine, train_dataloader, val_dataloader)
            # ---- finetune the encoder (eventually)
            if cfg.dataset.loader.ftune_size > 0: 
                trainer, engine = finetune_model(cfg, engine, dataset)
            # ----- test
            trainer.test(engine, test_dataloader, ckpt_path='best')
        
        trainer.logger.finalize("success")
  
    finally:
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.finish()
    ############################################################################################


if __name__ == "__main__":
    main()
    print('done')