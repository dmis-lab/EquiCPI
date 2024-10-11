import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
import torch
import random
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import rankdata, spearmanr, pearsonr
from lifelines.utils import concordance_index

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def data_making(data_name):
    dataset_folder = '/ssd1/quang/moldock/DiffDock/results/bindingdb_score_temp/rerank'
    # data_name = 'bindingdb_diff_rerank.csv'
    data_complex_name=[]
    data_comps = []
    data_prots = []
    data_labels = []

    for complex_name in tqdm(os.listdir(dataset_folder)):
        complex_dir = os.path.join(dataset_folder,complex_name)
        if len(os.listdir(complex_dir))>=3:
            # i = 0
            index = 0
            for file in os.listdir(complex_dir):

                if file.endswith('rank0.sdf'):
                    data_comps.append(os.path.join(complex_dir,file))
                    index+=1
                if file.endswith('fold.pdb'):
                    data_prots.append(os.path.join(complex_dir,file))
                    index+=1
                if file.endswith('label.csv'):
                    label = pd.read_csv(os.path.join(complex_dir,file))
                    data_labels.append(label['label'].iloc[0])
                    index+=1
            if index ==3:
                data_complex_name.append(complex_name)

    # import pdb
    # pdb.set_trace()
    df = pd.DataFrame({'complex_name':data_complex_name,'ligand':data_comps,'receptor':data_prots,'label':data_labels})
    df.to_csv(data_name)
    return data_name

def save_snapshot(epoch, model, snapshot_path = "checkpoint.pt"):
    snapshot = {
            "MODEL_STATE": model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
    torch.save(snapshot, snapshot_path)
def save_snapshot_single(epoch, model, snapshot_path = "checkpoint.pt"):
    snapshot = {
            "MODEL_STATE": model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
    torch.save(snapshot, snapshot_path)
def load_snapshot(model, snapshot_path, rank):
    loc = f"cuda:{rank}"
    snapshot = torch.load(snapshot_path, map_location = loc)
    model.load_state_dict(snapshot["MODEL_STATE"])
    epoch = snapshot["EPOCHS_RUN"]
    return model, epoch


def df_for_ex(df_all, df_small):
    dupp = np.intersect1d(np.asarray(df_all['complex_name']).astype(int),np.asarray(df_small['complex_name']).astype(int))
    df_use = df_all[pd.to_numeric(df_all['complex_name'], downcast='integer').isin(dupp)]
    df_aligned = df_use[df_use['complex_name'].isin(np.asarray(df_small['complex_name']).astype(int))].sort_values(by='complex_name', key=lambda x: x.map({k: i for i, k in enumerate(np.asarray(df_small['complex_name']).astype(int))}))
    return df_aligned


def metrics(labels, predictions):
    nums = [len(labels)]
  
    for num in nums:
        labels_rank = rankdata(labels[:num])
        predictions_rank = rankdata(predictions[:num])
        label_pred_dict = dict(zip(labels_rank,predictions_rank))

        mse_metrics = mean_squared_error(labels, predictions)
        print('MSE:', mse_metrics)
        C_index = concordance_index(labels[:num], predictions[:num])
        print('CI:', C_index) 
        correlation_coefficient_spear, p_value = spearmanr(list(label_pred_dict.keys()), list(label_pred_dict.values()))
        print("Spearman's rank correlation coefficient:", correlation_coefficient_spear)
        print("p-value:", p_value)
        correlation_coefficient_pear, p_value = pearsonr(list(label_pred_dict.keys()), list(label_pred_dict.values()))
        print("Pearson's rank correlation coefficient:", correlation_coefficient_pear)
        print("p-value:", p_value)
        
    return mse_metrics, C_index,correlation_coefficient_spear,correlation_coefficient_pear, p_value

def get_optimization(args, model, optimization):
    if optimization == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.w_decay)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.w_decay)
    return optim
def get_scheduler(args, optim, scheduler):
    if scheduler == 'Plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                                mode='min', 
                                                                patience=args.scheduler_patience, 
                                                                min_lr=args.lr/ args.n_epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = args.gamma)
    return scheduler
def get_loss(task_ml):
    if task_ml == 'classification':
        loss_fn = torch.nn.BCELoss()
    else:
        loss_fn = torch.nn.MSELoss()
    return loss_fn


class ExponentialMovingAverage:
    """ from https://github.com/yang-song/score_sde_pytorch/blob/main/models/ema.py
    Maintains (exponential) moving average of a set of parameters. """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the result of
            `model.parameters()`.
          decay: The exponential decay.
          use_num_updates: Whether to use number of updates when computing
            averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; usually the same set of
            parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [tensor.to(device) for tensor in state_dict['shadow_params']]