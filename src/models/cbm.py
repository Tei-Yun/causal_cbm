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
                 c_info={},
                 y_info={}):
        super(CBM, self).__init__()

        # to be stored for every model
        self.has_concepts = True
        self.is_causal = False
        self.normalize_concept_loss = normalize_concept_loss


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

      
    def forward(self, x, c=None, intervention_index=None):
        # filter out virtual roots from c and intervention_index
        filtered_c = c[:, [i for i, name in enumerate(self.concept_names) if name not in self.virtual_roots]]
        filtered_intervention_index = intervention_index[:, [i for i, name in enumerate(self.concept_names) if name not in self.virtual_roots]] 
        
        c_hat_logits = self.c_mlp(x) #여기까지는 하나의 큰 tensor로 나옴
        c_hat_probs = {}

        #concept별로 softmax 쪼개서 dict로 저장
        for i, name in enumerate(self.filtered_c_info['names']):
            c_hat_probs[name] = torch.softmax(c_hat_logits[:,sum(self.filtered_c_info['cardinality'][:i]):sum(self.filtered_c_info['cardinality'][:i+1])], dim=1)

            # Intervention 수행
            ##intervention_index==1인 concept의 predicted probability를 ground truth one-hot로 교체
            if filtered_c[:,i] is not None and filtered_intervention_index[:,i] is not None:
                c_hat_probs[name] = maybe_intervene(c_hat_probs[name], filtered_c[:,i], filtered_intervention_index[:,i])
        
        c_probs_concat = torch.cat(list(c_hat_probs.values()), dim=1)
        # Task predictions
        y_hat_logits = self.decoder(c_probs_concat)
        y_hat_probs = torch.softmax(y_hat_logits, dim=1)
        return y_hat_probs, c_hat_probs
    
    def filter_output_for_loss(self, y_output, c_output):
        """Filter output for loss function"""
        return y_output, c_output
    
    def filter_output_for_metric(self, y_output, c_output):
        """Filter output for metric function"""
        return y_output, c_output

    def loss(self, y_hat, y, c_hat_dict, c):
        """Compute loss function.
        Args:
            y_hat (torch.Tensor): Predicted task probabilities
            y (torch.Tensor): True task labels
            c_hat_dict (Dict): Predicted concept probabilities
            c (torch.Tensor): True concept labels"""
        y = y.flatten().long()
        # c = c.long() # later to avoid nan to disappear
        loss_form = torch.nn.NLLLoss()

        # -- task loss
        y_hat = torch.log(y_hat + 1e-6)
        task_loss = loss_form(y_hat, y)

        # -- concepts loss
        concept_loss = 0
        for name, c_hat in c_hat_dict.items():
            c_hat = torch.log(c_hat + 1e-6)
            concept_loss += loss_form(c_hat, c[:,self.concept_names.index(name)].long()) #각 concept 별로 독립적인 cross entropy 계산
        if self.normalize_concept_loss:
            concept_loss /= len(c_hat_dict)
            
        return self.concept_loss_weight * concept_loss + (1-self.concept_loss_weight) * task_loss