import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import networkx as nx

class ConceptFashionMNISTDataset(FashionMNIST):
    def __init__(
        self,
        root: str,
        concept_dir: str,
        train: bool = True,
        transform = None,
        download: bool = False,
    ):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.concept_dir = concept_dir
        
        # Ensure concept directory exists
        if not os.path.exists(self.concept_dir):
            os.makedirs(self.concept_dir, exist_ok=True)
            print(f"Warning: Concept directory {self.concept_dir} created but empty.")
            # 실제로는 여기에 concept 파일 다운로드 로직이나 생성 로직이 필요할 수 있습니다.
            # 파일이 없으면 에러가 발생할 수 있으므로, 더미 데이터를 생성하거나 사용자가 파일을 넣어야 합니다.

        self.concepts = self._load_concepts()

    def _load_concepts(self) -> torch.Tensor:
        # 파일명 설정 (사용자 코드 기반)
        prefix = 'train' if self.train else 'test'
        pt_path = os.path.join(self.concept_dir, f"{prefix}_concept_tensor.pt")
        csv_path = os.path.join(self.concept_dir, f"{prefix}_concept_vectors_with_index.csv")

        if os.path.exists(pt_path):
            return torch.load(pt_path)
        elif os.path.exists(csv_path):
            print(f"Loading concepts from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            # Index 컬럼 제거 및 텐서 변환
            if "Index" in df.columns:
                df = df.drop(columns=["Index"])
            return torch.tensor(df.values, dtype=torch.float32)
        else:
            # 파일이 없을 경우 (테스트용 더미 생성 - 실제 사용시 제거 권장)
            print(f"Warning: Concept file not found at {pt_path}. Generating random concepts.")
            return torch.randint(0, 2, (len(self.data), 5)).float()

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        
        if self.transform is not None:
            img = self.transform(img)
            
        concept = self.concepts[index]
        
        # 프로젝트 파이프라인에 맞는 딕셔너리 반환
        return {'x': img, 'c': concept, 'y': target}

class FashionMNISTContainer:
    def __init__(self, dataset_name, data_dir, concept_dir, **kwargs):
        self.dataset_name = dataset_name
        
        # ResNet Backbone 사용을 위한 Transform (1채널 -> 3채널, 224x224 리사이즈)
        # 만약 작은 CNN을 쓴다면 리사이즈 불필요
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Train / Test Split 로드
        full_train = ConceptFashionMNISTDataset(
            root=data_dir, 
            concept_dir=concept_dir, 
            train=True, 
            transform=self.transform, 
            download=True
        )
        
        self.test_dataset = ConceptFashionMNISTDataset(
            root=data_dir, 
            concept_dir=concept_dir, 
            train=False, 
            transform=self.transform, 
            download=True
        )

        # Train / Val Split (55000 / 5000)
        train_size = 55000
        val_size = 5000
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_train, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )

        self.data = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }

        # Metadata 설정
        # Concept 개수 확인
        n_concepts = full_train.concepts.shape[1]
        self.c_info = {
            'names': [f'c_{i}' for i in range(n_concepts)],
            'cardinality': [2] * n_concepts
        }
        self.c_info_complete = self.c_info
        
        self.y_info = {
            'names': ['label'],
            'cardinality': [10] # FashionMNIST classes
        }

        # 기본 Graph 생성 (모든 Concept -> Y)
        self.true_graph = nx.DiGraph()
        for c_name in self.c_info['names']:
            self.true_graph.add_edge(c_name, 'label')

    def load_ground_truth_graph(self):
        return self.true_graph