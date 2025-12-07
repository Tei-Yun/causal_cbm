
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class MyLogger(Logger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.values = {}

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for k, v in metrics.items():
            if k in self.values:
                self.values[k].append(v)
            else:
                self.values[k] = [v]

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        pass

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass

class LitFlow(L.LightningModule):
    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def training_step(self, batch, batch_idx):
        x = batch[0]
        loss = -self.flow().log_prob(x).mean()
        self.log('train_loss', loss.detach())
        self.log('log_prob', -loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer



def create_df(tensors_dict, col_names = None):
    df_list = []
    for i, (k, x) in enumerate(tensors_dict.items()):
        if col_names is not None:
            df = pd.DataFrame(x.numpy(), columns=col_names)
        else:
            df = pd.DataFrame(x.numpy(), columns=[f'$x_{j}$' for j in range(x.shape[1])])
        df['name'] = k
        df_list.append(df)

    df = pd.concat(df_list)
    return df


def plot_data(tensors_dict, mode = "kde", col_names = None):
    df = create_df(tensors_dict, col_names)
    g = sns.PairGrid(df, diag_sharey=False, hue='name')
    g.map_upper(sns.scatterplot, s=15, alpha=0.5)
    if mode == "kde":
        g.map_lower(sns.kdeplot, common_norm=False)
        g.map_diag(sns.kdeplot, lw=2, common_norm=False)
    elif mode == "hist":
        g.map_lower(sns.histplot, bins=20)
        g.map_diag(sns.histplot, bins=20)
    g.add_legend()
    plt.show()



from mutual_information import (
    estimate_MI_concepts_task,
    estimate_MI_interconcept,
)

def compute_ctl_icl(c_hat, c_true, y_true, n_neighbors=3):
    """
    c_hat : (N, d) torch.Tensor or np.ndarray, inferred concepts (binary)
    c_true: (N, d) torch.Tensor or np.ndarray, ground-truth concepts (binary)
    y_true: (N,)   torch.Tensor or np.ndarray, task labels (int class index)
    """
    # 텐서 -> numpy 변환 (이미 mutual_information 내부에서도 처리하지만, 명시적으로 해둘게)
    if isinstance(c_hat, torch.Tensor):
        c_hat_np = c_hat.detach().cpu().numpy()
    else:
        c_hat_np = np.asarray(c_hat)

    if isinstance(c_true, torch.Tensor):
        c_true_np = c_true.detach().cpu().numpy()
    else:
        c_true_np = np.asarray(c_true)

    if isinstance(y_true, torch.Tensor):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.asarray(y_true)

    N, d = c_hat_np.shape

    # ----------------------------------
    # 1) CTL (concepts-task leakage)
    # ----------------------------------
    # I(ĉ_i; y)/H(y)  (길이 d 벡터)
    I_chat_y = estimate_MI_concepts_task(
        c_hat_np,
        y_true_np,
        n_concepts=d,
        n_neighbors=n_neighbors,
        normalise=True,   # /H(y) 로 정규화
    )

    # I(c_i; y)/H(y)  (길이 d 벡터)
    I_ctrue_y = estimate_MI_concepts_task(
        c_true_np,
        y_true_np,
        n_concepts=d,
        n_neighbors=n_neighbors,
        normalise=True,
    )

    # CTL_i = max(0, I(ĉ_i,y)/H(y) - I(c_i,y)/H(y))
    CTL_i = np.maximum(0.0, I_chat_y - I_ctrue_y)   # shape (d,)
    CTL = CTL_i.mean()                              # scalar

    # ----------------------------------
    # 2) ICL (interconcept leakage)
    # ----------------------------------
    # I(ĉ_i; ĉ_j)/sqrt(H(ĉ_i)H(ĉ_j))
    MI_chat = estimate_MI_interconcept(
        c_hat_np,
        n_concepts=d,
        flatten=False,       # 전체 (d,d) 행렬로 받기
        n_neighbors=n_neighbors,
        normalise=True,
    )                         # shape (d, d), diag = 0

    # I(c_i; c_j)/sqrt(H(c_i)H(c_j))
    MI_ctrue = estimate_MI_interconcept(
        c_true_np,
        n_concepts=d,
        flatten=False,
        n_neighbors=n_neighbors,
        normalise=True,
    )                         # shape (d, d), diag = 0

    # ICL_ij = max(0, MI_chat - MI_ctrue)
    ICL_ij = np.maximum(0.0, MI_chat - MI_ctrue)
    np.fill_diagonal(ICL_ij, 0.0)  # 논문 정의상 i=j 항은 0

    # per-concept ICL_i: 각 행에서 j≠i 평균
    ICL_i = ICL_ij.sum(axis=1) / (d - 1)   # shape (d,)
    ICL = ICL_i.mean()                     # scalar

    return {
        "CTL_i": CTL_i,       # (d,)
        "CTL": CTL,           # scalar
        "ICL_ij": ICL_ij,     # (d, d)
        "ICL_i": ICL_i,       # (d,)
        "ICL": ICL,           # scalar
    }