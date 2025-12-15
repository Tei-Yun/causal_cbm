import torch
import torch.nn as nn
from src.models.layers.base import MLP
from src.models.layers.intervention import maybe_intervene

class CBM(nn.Module):
    """
    Concept bottleneck model. It predicts both task and concept labels.
    """
    def __init__(self, 
                 input_size, 
                 hidden_size,
                 output_size=2,
                 n_layers_concept_encoder=1,
                 n_layers_decoder=1,
                 activation='leaky_relu',
                 concept_loss_weight=0.5,
                 normalize_concept_loss=False,
                 decoder_type='mlp',
                 dropout=0.0,
                 # tei 추가 12/4: 학습 모드 설정 ('joint', 'independent', 'stepwise')
                 training_mode='joint', 
                 train_stage = 'joint',           # 'joint', 'encoder', 'decoder'
                 c_info={},
                 y_info={}):
        super(CBM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False
        self.normalize_concept_loss = normalize_concept_loss

        #tei 추가 12/4
        self.training_mode = training_mode  # 'joint', 'independent', 'stepwise'
        self.train_stage = train_stage          # 'joint', 'encoder', 'decoder'

        # concepts info
        self.concept_names = c_info['names']
        self.virtual_roots = [name for name in c_info['names'] if name.startswith('#virtual_')] #유령 root 제거 (BN dataset에서만 생김)
        self.concept_loss_weight = concept_loss_weight

        ##concept 하나가 binary라면 cardinality=2
        self.filtered_c_info = {
            "names": [name for name in c_info["names"] if name not in self.virtual_roots],
            "cardinality": [card for name, card in zip(c_info["names"], c_info["cardinality"]) if name not in self.virtual_roots]
        }
        
        # Concept encoder
        self.c_mlp = MLP(input_size=input_size,
                         hidden_size=hidden_size,
                         output_size=sum(self.filtered_c_info['cardinality']),
                         n_layers=n_layers_concept_encoder,
                         activation=activation)
        
        # Decoder
        if decoder_type == 'mlp':
            self.decoder = MLP(input_size=sum(self.filtered_c_info['cardinality']),
                               hidden_size=sum(self.filtered_c_info['cardinality'])//2,
                               output_size=output_size,
                               n_layers=n_layers_decoder,
                               activation=activation,
                               dropout=dropout)
        elif decoder_type == 'linear':
            self.decoder = nn.Linear(sum(self.filtered_c_info['cardinality']), 
                                     output_size)
        else:
            raise ValueError(f"Decoder type {decoder_type} not supported")

    #tei 수정 12/3
    def forward(self, x, c=None, intervention_index=None):
        # 0. virtual root 제거
        idx_keep = [i for i, name in enumerate(self.concept_names) if name not in self.virtual_roots]
        filtered_c = c[:, idx_keep] if c is not None else None
        filtered_intervention_index = intervention_index[:, idx_keep] if intervention_index is not None else None

        
        # 1. 항상 encoder는 한 번 돌려서 c_hat_logits / c_hat_probs / c_hat_hard 계산
        c_hat_logits = self.c_mlp(x)
        c_hat_probs = {}
        c_hat_hard  = {}

        # Concept별 Softmax 및 Hard 값 계산
        for i, name in enumerate(self.filtered_c_info['names']):
            logits = c_hat_logits[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1])]
            
            c_hat_probs[name] = torch.softmax(logits, dim=1)
            
            #gumbel softmax를 사용하여 hard one-hot 벡터 생성 => thersholding 없이 미분 가능 -> 0.5 기준이랑은 다른듯...
            c_hat_hard[name] = torch.nn.functional.gumbel_softmax(logits, tau=1, hard=True, dim=1)

            # [수정] Encoder 학습 단계가 아닐 때만 Intervention 수행
            # Encoder 학습 때는 스스로 맞추도록 해야 함 (정답을 알려주면 Loss가 0이 되어 학습 안됨)
            if self.train_stage != 'encoder':
                if filtered_c is not None and filtered_intervention_index is not None:
                    # 해당 배치, 해당 컨셉에 대해 intervention이 켜져 있다면 교체
                    if filtered_intervention_index[:,i] is not None: 
                        c_hat_probs[name] = maybe_intervene(c_hat_probs[name], filtered_c[:,i], filtered_intervention_index[:,i])
                        c_hat_hard[name] = maybe_intervene(c_hat_hard[name], filtered_c[:,i], filtered_intervention_index[:,i])
        # [Case 1] Encoder 학습 단계 (Stepwise/Independent의 Encoder phase)
        if self.train_stage == "encoder":
            # Decoder 실행 안 함
            return None, c_hat_probs

        # [Case 2] Decoder 학습 단계 또는 Joint 학습, 또는 Inference
        else:
            # Decoder Input 결정 로직
            
            # A. Independent 모드 + Decoder 학습 중 + GT 존재
            # => Ground Truth Concept을 입력으로 사용
            if self.training_mode == 'independent' and self.train_stage == 'decoder' and self.training and filtered_c is not None:
                gt_inputs = []
                for i, name in enumerate(self.filtered_c_info['names']):
                    gt_idx = filtered_c[:, i].long()
                    card = self.filtered_c_info['cardinality'][i]
                    gt_one_hot = torch.nn.functional.one_hot(gt_idx, num_classes=card).float()
                    gt_inputs.append(gt_one_hot)
                decoder_input = torch.cat(gt_inputs, dim=1)
            
            # B. Stepwise 모드 + Decoder 학습 중
            # => Predicted Concept을 사용하되, Gradient 차단 (Detach)
            elif self.training_mode == 'stepwise' and self.train_stage == 'decoder' and self.training:
                c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
                c_hard_concat = torch.cat(list(c_hat_hard.values()), dim=1)

                decoder_input = c_probs_concat.detach() # Encoder로 Gradient 흐르지 않음
                #decoder_input = c_hard_concat.detach() # Encoder로 Gradient 흐르지 않음


            # C. Joint 모드 또는 Inference (Test/Val)
            # => Predicted Concept 사용 (Gradient 흐름 유지)
            else:
                c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
                c_hard_concat = torch.cat(list(c_hat_hard.values()), dim=1)

                decoder_input = c_probs_concat 
                #decoder_input = c_hard_concat

            # Decoder 실행
            y_hat_logits = self.decoder(decoder_input)
            y_hat_probs = torch.softmax(y_hat_logits, dim=1)
            
            return y_hat_probs, c_hat_probs


    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output


    #tei 수정 12/4
    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function based on training mode and stage."""
        y = y.flatten().long()
        loss_form = torch.nn.NLLLoss()
        # 1. Concept Loss 계산 (항상 계산 가능)
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            c_hat = torch.log(c_hat + 1e-6)
            concept_loss += loss_form(c_hat, c[:,self.concept_names.index(name)].long())
        
        if self.normalize_concept_loss:
            concept_loss /= len(c_hat_dict)

        # 2. Task Loss 계산 (y_hat이 있을 때만)
        task_loss = 0
        if y_hat is not None:
            y_hat_log = torch.log(y_hat + 1e-6)
            task_loss = loss_form(y_hat_log, y)

        # 3. 최종 Loss 반환 (Mode/Stage에 따라 분기)
        
        # [Encoder Only Stage] (Stepwise/Independent)
        if self.train_stage == 'encoder':
            return concept_loss
        
        # [Decoder Only Stage] (Stepwise/Independent)
        elif self.train_stage == 'decoder':
            return task_loss
        
        # [Joint Stage]
        else:
            return self.concept_loss_weight * concept_loss + (1 - self.concept_loss_weight) * task_loss

