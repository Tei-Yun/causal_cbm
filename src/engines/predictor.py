from typing import Any, Optional, Mapping, Type
import pickle
import itertools

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
#tei 수정 11/29
from torchmetrics.collections import _remove_prefix
import pytorch_lightning as pl

from src.models.layers.intervention import get_test_intervention_index

#tei 추가 12/11 w/ yeom
from causalflows.flows import CausalMAF
from src.utils import post_process, add_noise, make_cf_batch, get_c_hat_tensor, make_int_batch, to_binary
import copy

class Predictor(pl.LightningModule):    
    def __init__(self,
                model: Optional[nn.Module] = None,
                metrics: Optional[Mapping[str, Metric]] = None,
                optim_class: Optional[Type] = None,
                optim_kwargs: Optional[Mapping] = None,
                scheduler_class: Optional[Type] = None,
                scheduler_kwargs: Optional[Mapping] = None,
                intervention_prob: Optional[float] = 0.2,
                c_names: Optional[list] = None,
                #tei 수정 11/29
                # [Add] Task 이름 인자 추가
                task_name: Optional[str] = 'y_hat',
                test_interv_policy: Optional[str] = None,
                test_interv_noise: Optional[float] = 0.,
                # [Add] 외부 Concept 파일 경로 인자 추가
                external_concept_path: Optional[str] = None,
                cnf_int_policy: Optional[str] = None, # [Add] CNF intervention policy
                cnf_bundle_path: Optional[str] = None,
                ):
        super(Predictor, self).__init__()         
        self.model = model
        self.save_hyperparameters(ignore=["model"], logger=False)

        #tei 수정 11/29
        # [Add] Task 이름 저장
        self.task_name = task_name

        # [Add] 외부 Concept 관련 변수 초기화
        self.external_concept_path = external_concept_path
        self.external_c_tensor = None
        self.test_sample_offset = 0


        self.optim_class = optim_class
        self.optim_kwargs = optim_kwargs or dict()
        self.scheduler_class = scheduler_class
        self.scheduler_kwargs = scheduler_kwargs or dict()

        # for regularization
        self.intervention_prob = intervention_prob
        # store the intervention policy
        self.test_interv_policy = test_interv_policy

        #tei 수정 11/29
        ## [Fix] test_interv_policy가 None이면 빈 리스트로 초기화 (len() 에러 방지)
        if self.test_interv_policy is None:
            self.test_interv_policy = []

        #tei 수정 12/11 w/ yeom
        #[Add] from causal-flows
        self.cnf_int_policy = cnf_int_policy # cnf_int or cnf_cf
        self.cnf_bundle_path = cnf_bundle_path
        assert not (self.cnf_int_policy is not None and self.cnf_bundle_path is None), "CNF bundle path is required for CNF intervention"
        if self.cnf_int_policy is not None:
            bundle = torch.load(self.cnf_bundle_path, map_location="cpu")
            self.cnf_flow = CausalMAF(bundle["features"], bundle["context"], adjacency=bundle["adjacency"])
            self.cnf_flow.load_state_dict(bundle["state_dict"])
            self.cnf_flow = self.cnf_flow.to("cpu").eval()
            self.cnf_flow_loaded = self.cnf_flow()
            self.topo_order_idx = bundle["topo_order_idx"]
            self.original_order = bundle["original_order"]
            self.topological_order = bundle["topological_order"]
            self.binary_dims = bundle["binary_dims"]
            self.binary_min_values = bundle["binary_min_values"]
            self.binary_max_values = bundle["binary_max_values"]


        self.test_interv_noise = test_interv_noise  


        self.c_names = c_names
        self.n_concepts = len(c_names)

        if metrics is None:
            metrics = dict()
        self._set_metrics(metrics)

        #tei 수정 11/29
        self.c_hat_accumulator = {name: [] for name in self.c_names}

        # [Add] Task prediction(y_hat)을 저장할 리스트 초기화
        self.y_hat_accumulator = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    @staticmethod
    def _check_metric(metric):
        metric = metric.clone()
        metric.reset()
        return metric
    
    def _set_metrics(self, metrics):
        # --- accuracy metrics ---
        y_acc_metrics = {'y_accuracy': metrics.get('classification_acc')}
        c_acc_metrics = {k: metrics.get('classification_acc') for k in self.c_names}

        # task accuracy metrics
        self.train_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="train/y/")
        self.val_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="val/y/")
        self.test_y_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in y_acc_metrics.items()},
            prefix="test/y/")
        
        # --- concept accuracy metrics ---
        self.train_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="train/c/")
        self.val_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="val/c/")
        self.test_c_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
            prefix="test/c/")      
          
        if self.model.has_concepts:
            # --- ground truth intervention metrics ---
            c_acc_metrics['_baseline'] = metrics.get('classification_acc')
            c_acc_levels_metrics = {f'level {n}': metrics.get('classification_acc')
                                    for n in range(0, len(self.test_interv_policy)+1)}
            
            # task accuracy after invervention on each individual concept 
            # (one metric for each concept)
            self.test_intervention_single_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
                prefix="test_intervention/single/y/")
            
            # task accuracy after intervention of each graph level
            self.test_intervention_level_y = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_levels_metrics.items()},
                prefix="test_intervention/level/y/")

            # # individual child concept accuracy after 
            # # intervention on ancestors in the graph
            # childs_per_level = {}
            # for l in range(0, len(self.test_interv_policy)+1):
            #     childs = list(itertools.chain(*self.test_interv_policy[l:]))
            #     for child in childs:
            #         child_name = self.c_names[child]
            #         childs_per_level[f'level {l}/child {child_name}'] = metrics.get('classification_acc')
            # self.test_intervention_level_c = MetricCollection(
            #     metrics={k: self._check_metric(m) for k, m in childs_per_level.items()},
            #     prefix="test_intervention/level/c/")

            # individual concept accuracy (task ancestors only, according to the policy) after 
            # intervention on levels defined by the policy
            nodes_per_level = {}
            indices_in_policy = list(itertools.chain(*self.test_interv_policy))
            c_names_in_policy = [self.c_names[i] for i in indices_in_policy]
            nodes_per_level.update({
                f'level {l}/node {c}': metrics.get('classification_acc')
                for l in range(len(self.test_interv_policy) + 1)
                for c in c_names_in_policy
            })
            self.test_intervention_level_c = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in nodes_per_level.items()},
                prefix="test_intervention/level/c/")
            
            #tei 수정 12/11 w/ yeom
            # --- CNF counterfactual metrics ---
            self.test_intervention_cnf_cf = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
                prefix="test_intervention/cnf_cf/")

            self.test_intervention_cnf_int = MetricCollection(
                metrics={k: self._check_metric(m) for k, m in c_acc_metrics.items()},
                prefix="test_intervention/cnf_int/")

            # --- fairness metrics ---
            self.cace = MetricCollection(
                metrics = {'before': self._check_metric(metrics.get('cace')),
                           'after': self._check_metric(metrics.get('cace'))},   
                prefix="test_intervention/cace/")

    def log_metrics(self, metrics, **kwargs):
        """"""
        self.log_dict(
            metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True, **kwargs
        )

    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(
            name + "_loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _unpack_batch(self, batch):
        """
        Unpack a batch into data and preprocessing dictionaries.
        """
        return batch['x'], batch['c'], batch['y']
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # add batch_size to batch
        if isinstance(batch, dict):
            batch['batch_size'] = batch['x'].shape[0]
        else:
            raise NotImplementedError("Only dict batches are supported")
        return batch

    def get_intervention_index(self, c_shape, step):
        """
        Get intervention index for training time intervention.
        Args:
            c_shape: shape of the concept tensor
            step: (str) 'train' or 'val'
        """
        # for regularization only
        if step=='train':
            intervention_index = torch.bernoulli(torch.ones(c_shape) * self.intervention_prob)
        else:
            intervention_index = torch.zeros(c_shape)
        return intervention_index.to("cuda" if torch.cuda.is_available() else "cpu")
    
    def test_intervention(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)
            # maybe add noise
            if self.test_interv_noise > 0:
                x = x + torch.randn_like(x) * self.test_interv_noise

            #Baseline (개입 없음)
            # 빈 리스트 [] 전달 -> intervention_index는 모두 0
            intervention_index = get_test_intervention_index(c.shape, [])
            inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
            # forward pass with intervention at test time
            y_output, c_output = self.forward(**inputs)
            y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)

            #tei 추가 12/11 w/ yeom
            c_hat_factual = copy.deepcopy(c_hat)

            # update metric after intervention:
            # how well can we predict y?
            self.test_intervention_single_y['_baseline'].update(y_hat, y)            

            #tei 추가 12/11 w/ yeom
            if self.cnf_int_policy == 'cnf_cf':
                print("INTERVENTION ON CONCEPTS BY CNF (CF)")
                for i, c_name_i in enumerate(self.c_names):
                    if c_name_i in self.model.virtual_roots: continue
           
                    '''
                    [Causal intervention] : single concept intervention using cauasl-flows
                    '''

                    #tei 수정 12/16
                    topo_idx = self.topological_order.index(c_name_i)

                    intervention_index = torch.ones(c.shape, device=c.device)
                    c_tensor_topo = batch['c'][:, self.topo_order_idx]
                    c_hat_tensor = get_c_hat_tensor(self.topological_order, c_hat_factual, c_tensor_topo, prob_values=False) # binary values
                    c_hat_cf_topo = make_cf_batch(c_hat_tensor, 
                                                    #index=i, 
                                                    index=topo_idx, #tei 수정 12/16
                                                    c=c_tensor_topo, 
                                                    binary_dims=self.binary_dims, 
                                                    binary_min_values=self.binary_min_values, 
                                                    binary_max_values=self.binary_max_values, 
                                                    flow_loaded=self.cnf_flow_loaded,
                                                    prob_values=False)
                    incomplete_idx = [self.topological_order.index(n) for n in self.c_names] # original order index of selected concepts
                    c_hat_cf = c_hat_cf_topo[:, incomplete_idx].to(c.device) # rearrange to original order
                    inputs = {'x':x, 'c':c_hat_cf, 'intervention_index':intervention_index}
                    y_output, c_output = self.forward(**inputs)
                    y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                    self.test_intervention_cnf_cf[c_name_i].update(y_hat, y)
            
            if self.cnf_int_policy == 'cnf_prob_cf':
                print("INTERVENTION ON CONCEPTS BY CNF (PROB CF)")
                for i, c_name_i in enumerate(self.c_names):
                    if c_name_i in self.model.virtual_roots: continue
           
                    '''
                    [Causal intervention] : single concept intervention using cauasl-flows
                    '''

                    #tei 수정 12/16
                    topo_idx = self.topological_order.index(c_name_i)

                    intervention_index = torch.ones(c.shape, device=c.device)
                    c_tensor_topo = batch['c'][:, self.topo_order_idx] if 'c' in batch else batch['c'][:, self.topo_order_idx]
                    c_hat_tensor = get_c_hat_tensor(self.topological_order, c_hat_factual, c_tensor_topo, prob_values=True) # probabliity values
                    c_hat_cf_topo = make_cf_batch(c_hat_tensor, 
                                                    #index=i,
                                                    index=topo_idx, #tei 수정 12/16 
                                                    c=c_tensor_topo, 
                                                    binary_dims=self.binary_dims, 
                                                    binary_min_values=self.binary_min_values, 
                                                    binary_max_values=self.binary_max_values, 
                                                    flow_loaded=self.cnf_flow_loaded,
                                                    prob_values=True)
                    incomplete_idx = [self.topological_order.index(n) for n in self.c_names] # original order index of selected concepts
                    c_hat_cf = c_hat_cf_topo[:, incomplete_idx].to(c.device) # rearrange to original order
                    inputs = {'x':x, 'c':c_hat_cf, 'intervention_index':intervention_index}
                    y_output, c_output = self.forward(**inputs)
                    y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                    self.test_intervention_cnf_cf[c_name_i].update(y_hat, y)      

            
            # if self.cnf_int_policy == 'cnf_int':
            #     print("INTERVENTION ON CONCEPTS BY CNF (INT)")
            #     for i, c_name_i in enumerate(self.c_names):
            #         if c_name_i in self.model.virtual_roots: continue
            #         intervention_index = torch.ones(c.shape, device=c.device)
            #         c_tensor_topo = batch['c'][:, self.topo_order_idx] 
            #         c_hat_int_topo = make_int_batch(index=i, 
            #                                         c=c_tensor_topo.to("cpu"), 
            #                                         binary_dims=self.binary_dims, 
            #                                         binary_min_values=self.binary_min_values.to("cpu"), 
            #                                         binary_max_values=self.binary_max_values.to("cpu"), 
            #                                         flow_loaded=self.cnf_flow_loaded)
            #         incomplete_idx = [self.topological_order.index(n) for n in self.c_names] # original order index of selected concepts
            #         c_hat_int = c_hat_int_topo[:, incomplete_idx].to(c.device) # rearrange to original order
            #         inputs = {'x':x, 'c':c_hat_int, 'intervention_index':intervention_index}
            #         y_output, c_output = self.forward(**inputs)
            #         y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
            #         self.test_intervention_cnf_int[c_name_i].update(y_hat, y)




            #Single Concept Intervention (하나씩 개입)
            # interventions on individual concepts
            for i, c_name_i in enumerate(self.c_names):
                if c_name_i in self.model.virtual_roots: continue

                ## i번째 Concept만 개입 (마스크의 i번째 컬럼만 1)
                intervention_index = get_test_intervention_index(c.shape, i)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}

                # 모델 Forward -> 이때 내부적으로 maybe_intervene이 호출되어 i번째 예측값이 정답으로 바뀜
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # 결과 기록 (이 Concept을 알면 y 예측이 얼마나 좋아지는가?)
                self.test_intervention_single_y[c_name_i].update(y_hat, y)

            # level intervention
            #Level/Group Intervention (그룹 개입)
            for l in range(0, len(self.test_interv_policy)+1):

                ## 정책(Policy)에 따라 여러 Concept을 동시에 개입
                nodes = list(itertools.chain(*self.test_interv_policy[:l]))
                intervention_index = get_test_intervention_index(c.shape, nodes)
                inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}

                # forward pass with intervention at test time
                y_output, c_output = self.forward(**inputs)
                y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict y?
                self.test_intervention_level_y[f'level {l}'].update(y_hat, y)
                # update metric after intervention:
                # after interveening on a level of the policy, how well can we predict each child concept?
                indices_in_policy = list(itertools.chain(*self.test_interv_policy))
                for node_index in indices_in_policy:
                    c_name = self.c_names[node_index]
                    if c_name in c_hat:
                        self.test_intervention_level_c[f'level {l}/node {c_name}'].update(c_hat[c_name], c[:,node_index])
                    else:
                        # if the concept is not in the output, we cannot compute the metric for that concept
                        # this can happen if the model does not predict all concepts
                        pass

    def test_intervention_fairness(self, batch):
        if self.model.has_concepts:
            x, c, y = self._unpack_batch(batch)

            # get a concept pair i,j (node j has to be a bottleneck for node i to the task)
            i = self.c_names.index('Attractive')
            j = self.c_names.index('Qualified')

            # compute the cace before the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=1)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, i, values=0)
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_before_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['before'].update(y_hat_before_do_1, y_hat_before_do_0)

            # on causal models like causal cem, because of the way they are implemented, is not necessary to strip eedges
            # after interventions, as interventions fix the values of the concept and previous calculations are useless
            # at most there is a little overhead in the forward pass
            # if self.model.is_causal:
            #     self.model.remove_edges(j)

            # compute the cace after the do-intervention on concept j
            # different do-interventions on concept i, effect on the task
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,1])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_1, _ = self.model.filter_output_for_metric(y_output, c_output)
            interv_index, interv_values = get_test_intervention_index(c.shape, [j,i], values=[1,0])
            y_output, c_output = self.forward(**{'x':x, 'c':interv_values, 'intervention_index':interv_index})
            y_hat_after_do_0, _ = self.model.filter_output_for_metric(y_output, c_output)
            self.cace['after'].update(y_hat_after_do_1, y_hat_after_do_0)

            self.log_metrics(self.cace, batch_size=batch['batch_size'])


    def update_and_log_metrics(self, step, y_hat, y, c_hat, c, batch):
        # update and log task metrics
        
        #tei 수정 12/3
        # [수정] y_hat이 None이 아닐 때만 Task Metric 업데이트
        if y_hat is not None:
            y_collection = getattr(self, f"{step}_y_metrics")
            y_collection.update(y_hat, y)
            self.log_metrics(y_collection, batch_size=batch['batch_size'])
            
        # update and log concept metrics
        c_collection = getattr(self, f"{step}_c_metrics")
        # log metrics for all predicted concepts 
        # (the collection contains all concepts, but some models predicts only a subset)
        if c_hat is not None:
            for k, v in c_hat.items():  
                c_collection[k].update(v, c[:,self.c_names.index(k)])
        self.log_metrics(c_collection, batch_size=batch['batch_size'])

    def shared_step(self, batch, step):
        x, c, y = self._unpack_batch(batch)
        intervention_index = self.get_intervention_index(c.shape, step=step)
        inputs = {'x':x, 'c':c, 'intervention_index':intervention_index}
        # model forward
        y_output, c_output = self.forward(**inputs)
        # Compute loss
        y_hat_loss, c_hat_loss = self.model.filter_output_for_loss(y_output, c_output)
        loss = self.model.loss(y_hat_loss, y, c_hat_loss, c)
        return loss, y_output, c_output, y, c

    def training_step(self, batch, batch_idx):
        loss, y_output, c_output, y, c = self.shared_step(batch, step='train')
        if torch.isnan(loss).any():
            print(f'at epoc: {self.current_epoch}, batch: {batch_idx}')
            print('Loss has nan')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("train", y_hat, y, c_hat, c, batch)
        self.log_loss("train", loss, batch_size=batch['batch_size'])
        return loss
    
    def on_train_epoch_end(self):
        # Set the current epoch for SCBM and update the list of concept probs for computing the concept percentiles
        if type(self.model).__name__ == 'SCBM':
            self.model.training_epoch = self.current_epoch
            # self.model.concept_pred = torch.cat(self.model.concept_pred_tmp, dim=0) 
            # self.model.concept_pred_tmp = []        

    def validation_step(self, batch, batch_idx):
        val_loss, y_output, c_output, y, c = self.shared_step(batch, step='val')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("val", y_hat, y, c_hat, c, batch)
        self.log_loss("val", val_loss, batch_size=batch['batch_size'])
        return val_loss
    



    def test_step(self, batch, batch_idx):
        test_loss, y_output, c_output, y, c = self.shared_step(batch, step='test')
        # Update metrics and log
        y_hat, c_hat = self.model.filter_output_for_metric(y_output, c_output)
        self.update_and_log_metrics("test", y_hat, y, c_hat, c, batch)
        self.log_loss("test", test_loss, batch_size=batch['batch_size'])
        
        # test-time interventions (기존 로직 - CBM 자체 예측값에 대한 Intervention)
        self.test_intervention(batch)
        if 'Qualified' in self.c_names:
            self.test_intervention_fairness(batch)

        #tei 수정 11/29
        # accumulate predicted concepts for later analysis
        # [Fix] Blackbox 모델은 c_hat이 None이므로 체크 필요
        if c_hat is not None:
            for name, pred in c_hat.items():
                # c_hat_accumulator에 해당 키가 있는지 확인 (안전장치)
                if name in self.c_hat_accumulator:
                    self.c_hat_accumulator[name].append(pred.detach().cpu())

        # [Add] Task 예측값(y_hat) 수집 (이것이 Mouth_Slightly_Open 예측값임)
        if y_hat is not None:
            self.y_hat_accumulator.append(y_hat.detach().cpu())

        return test_loss
    






    def on_test_epoch_end(self):
        # baseline task accuracy
        y_baseline = self.test_y_metrics['y_accuracy'].compute().item()
        print(f"Baseline task accuracy: {y_baseline}")
        pickle.dump({'_baseline':y_baseline}, open(f'results/y_accuracy.pkl', 'wb'))

        # baseline concept accuracy
        c_baseline = {}
        for k, metric in self.test_c_metrics.items():
            k = _remove_prefix(k, self.test_c_metrics.prefix)
            c_baseline[k] = metric.compute().item()
            print(f"Baseline concept accuracy for {k}: {c_baseline[k]}")
        pickle.dump(c_baseline, open(f'results/c_accuracy.pkl', 'wb'))

        if self.model.has_concepts:
            # task accuracy after invervention on each individual concept
            y_int = {}
            for k, metric in self.test_intervention_single_y.items():
                c_name = _remove_prefix(k, self.test_intervention_single_y.prefix)
                y_int[c_name] = metric.compute().item()
                print(f"Task accuracy after intervention on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/single_c_interventions_on_y.pkl', 'wb'))

            # task accuracy after intervention of each policy level
            y_int = {}
            for k, metric in self.test_intervention_level_y.items():
                level = _remove_prefix(k, self.test_intervention_level_y.prefix)
                y_int[level] = metric.compute().item()
                print(f"Task accuracy after intervention on {level}: {y_int[level]}")
            pickle.dump(y_int, open(f'results/level_interventions_on_y.pkl', 'wb'))

            # individual concept accuracy after intervention of each policy level
            c_int = {}
            for k, metric in self.test_intervention_level_c.items():
                level = _remove_prefix(k, self.test_intervention_level_c.prefix)
                c_int[level] = metric.compute().item()
                print(f"Concept accuracy after intervention on {level}: {c_int[level]}")
            pickle.dump(c_int, open(f'results/level_interventions_on_c.pkl', 'wb'))


            #tei CNF 추가 12/11 w/ yeom 

            # CNF counterfactual task accuracy
            y_int = {}
            for k, metric in self.test_intervention_cnf_cf.items():
                c_name = _remove_prefix(k, self.test_intervention_cnf_cf.prefix)
                y_int[c_name] = metric.compute().item()
                print(f"Task accuracy after CNF(CF) on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/cnf_cf_interventions_on_y.pkl', 'wb'))


            # CNF interventional task accuracy
            y_int = {}
            for k, metric in self.test_intervention_cnf_int.items():
                c_name = _remove_prefix(k, self.test_intervention_cnf_int.prefix)
                y_int[c_name] = metric.compute().item()
                print(f"Task accuracy after CNF(INT) on {c_name}: {y_int[c_name]}")
            pickle.dump(y_int, open(f'results/cnf_int_interventions_on_y.pkl', 'wb'))





            # save graph and concepts
            pickle.dump({'concepts':self.c_names,
                         'policy':self.test_interv_policy}, open("graph.pkl", 'wb'))
            
            pickle.dump({'policy':self.test_interv_policy}, open("policy.pkl", 'wb'))
        
        #tei 수정 11/29 => 이부분을 아예 p(0)만 저장하도록 해도 될듯
        # save all predicted concept probabilities
        # [Fix] 데이터가 수집된 경우에만 저장 (Blackbox 제외)
        # [Modified] c_hat 뿐만 아니라 y_hat도 함께 저장하도록 수정
        if any(self.c_hat_accumulator.values()) or self.y_hat_accumulator:
            # [Modified] 딕셔너리 분리 (Concept Only)
            final_data_binary_c = {}
            final_data_probs_c = {}
            
            # [Add] 이진화 변환 함수 (확률 -> 0 or 1)
            def to_binary(t):
                # 1. 차원이 1개거나 [N, 1] 형태인 경우 (Sigmoid 확률값) -> 0.5 기준 thresholding
                if t.ndim == 1 or (t.ndim == 2 and t.shape[1] == 1):
                    return (t > 0.5).float() 
                # 2. 차원이 [N, C] 형태인 경우 (Softmax 확률값) -> 가장 높은 확률의 인덱스(argmax)
                elif t.ndim == 2 and t.shape[1] > 1:
                    return t.argmax(dim=1).float()
                return t

            # [Add] 확률값 추출 함수 (Soft Prediction)
            def to_prob(t):
                # 1. [N, 1] 형태인 경우 (Sigmoid 확률값) -> [N]으로 차원 축소
                if t.ndim == 2 and t.shape[1] == 1:
                    return t.squeeze(1)
                # 2. 이미 [N] 형태인 경우 -> 그대로 반환
                elif t.ndim == 1:
                    return t
                # 3. [N, C] 형태인 경우 (Softmax 확률값) -> Class 1 (두번째) 확률 반환
                # 예: [0.7, 0.3] -> 0.3
                elif t.ndim == 2 and t.shape[1] > 1:
                    return t[:, 1]
                return t

            # 1. Concepts 병합 (Concept Only 딕셔너리에 저장)
            if any(self.c_hat_accumulator.values()):
                for name in self.c_hat_accumulator:
                    if self.c_hat_accumulator[name]:
                        concat_t = torch.cat(self.c_hat_accumulator[name], dim=0)
                        final_data_binary_c[name] = to_binary(concat_t)
                        final_data_probs_c[name] = to_prob(concat_t)
            
            # 2. Task 포함 버전 딕셔너리 생성 (Concept Only 복사)
            final_data_binary_cy = final_data_binary_c.copy()
            final_data_probs_cy = final_data_probs_c.copy()

            # 3. Task Prediction (Mouth_Slightly_Open) 병합 및 추가
            if self.y_hat_accumulator:
                # [Fix] y_hat_accumulator는 리스트이므로 for loop 없이 바로 cat
                y_concat = torch.cat(self.y_hat_accumulator, dim=0)
                
                # [Fix] task_name이 ListConfig, list, 혹은 문자열 형태일 때 깨끗한 문자열로 변환
                # 예: "['Mouth_Slightly_Open']" -> "Mouth_Slightly_Open"
                t_name = str(self.task_name)
                for char in ['[', ']', "'", '"']:
                    t_name = t_name.replace(char, "")
                t_name = t_name.strip()

                # Task 포함 버전에만 추가
                final_data_binary_cy[t_name] = to_binary(y_concat)
                final_data_probs_cy[t_name] = to_prob(y_concat)

            # Save Files (총 4개)
            
            # 1) Binary - Concepts Only
            pickle.dump(final_data_binary_c, open("results/c_hat_only_concepts_binary.pkl", "wb"))
            
            # 2) Binary - Concepts + Task
            pickle.dump(final_data_binary_cy, open("results/c_hat_with_task_binary.pkl", "wb"))
            
            # 3) Probs - Concepts Only
            pickle.dump(final_data_probs_c, open("results/c_hat_only_concepts_probs.pkl", "wb"))
            
            # 4) Probs - Concepts + Task
            pickle.dump(final_data_probs_cy, open("results/c_hat_with_task_probs.pkl", "wb"))

            print("Saved prediction results (Binary/Probs x Only/WithTask) to results/ folder.")



    def configure_optimizers(self):
        """"""
        cfg = dict()
        optimizer = self.optim_class(self.parameters(), **self.optim_kwargs)
        cfg["optimizer"] = optimizer
        if self.scheduler_class is not None:
            metric = self.scheduler_kwargs.pop("monitor", None)
            scheduler = self.scheduler_class(optimizer, **self.scheduler_kwargs)
            cfg["lr_scheduler"] = scheduler
            if metric is not None:
                cfg["monitor"] = metric
        return cfg
 