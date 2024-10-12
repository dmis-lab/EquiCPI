import pandas as pd
import numpy as np 
import os
from tqdm import tqdm
import torch
import torch_geometric
import wandb
from time import ctime

from torch_geometric.loader import DataLoader
from common.ultils import *
from dataset import CPI3DDataset
from model import TensorProductModel_one_hot3
from common.parsing import parse_train_args

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

def main(args):
    wandb.login()
    task = args.task
    result_name = args.result_name

    wandb.init(
        dir='/ssd1/quang/moldock/e3nn_cpi_project',
        # Set the project where this run will be logged
        project="{}{}{}{}".format(ctime().replace(' ','_').replace(':','_'),args.model_name,task,result_name),
        config={
            "batch_size": args.batch_size,
            "epochs": args.n_epochs,
        },
    )
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(2024)
    tasks = [task]
    save_fold = './checkpoint_dir'
    if not os.path.isdir(save_fold):  os.makedirs(save_fold, exist_ok=True)

    for task in tasks:
        for fold in range(5):
            if args.task_ml == 'classification':
                train_dataset = CPI3DDataset(processed_data_pt = '/ssd1/quang/moldock/e3nn_cpi_project/processed_data/bindingdb_classificationdiff_classification/data_data_bindingDB_train_bindingdb_classificationlabel_encode.pt')
                test_dataset = CPI3DDataset(processed_data_pt = '/ssd1/quang/moldock/e3nn_cpi_project/processed_data/dude_classificationdiff_classification/data_data_bindingDB_train_dude_classificationlabel_encode.pt')                
                indices = list(range(len(train_dataset)))
                np.random.shuffle(indices)
                split = int(np.floor(0.2 * len(train_dataset)))
                val_indices = indices[:split]
                train_indices = indices[split:]

                train_loader = DataLoader(train_dataset[train_indices], batch_size=args.batch_size, drop_last=True, shuffle=True)
                val_loader = DataLoader(train_dataset[val_indices], batch_size=args.batch_size, drop_last=True, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            else:
                train_dataset = CPI3DDataset(processed_data_pt ='/ssd1/quang/moldock/e3nn_cpi_project/processed_data/bindingDBdiffrerank/data_data_bindingDB_train_{}{}label_encode.pt'.format(task,fold))
                val_dataset = CPI3DDataset(processed_data_pt ='/ssd1/quang/moldock/e3nn_cpi_project/processed_data/bindingDBdiffrerank/data_data_bindingDB_val_{}{}label_encode.pt'.format(task,fold))
                test_dataset = CPI3DDataset(processed_data_pt ='/ssd1/quang/moldock/e3nn_cpi_project/processed_data/bindingDBdiffrerank/data_data_bindingDB_test_{}{}label_encode.pt'.format(task,fold))   

                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
                test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

            model = TensorProductModel_one_hot3(device = rank, 
                                        in_lig_edge_features=4, 
                                        sh_lmax=2,
                                        ns=args.ns, 
                                        nv=args.nv, 
                                        num_conv_layers=args.num_conv_layers, 
                                        lig_max_radius=args.lig_max_radius, 
                                        rec_max_radius=args.rec_max_radius, 
                                        cross_max_distance=args.cross_max_distance,
                                        distance_embed_dim=args.distance_embed_dim, 
                                        cross_distance_embed_dim=args.cross_distance_embed_dim, 
                                        use_second_order_repr='1', batch_norm=True,
                                        dropout=args.dropout, 
                                        confidence_dropout=args.dropout, confidence_no_batchnorm=False, 
                                        num_confidence_outputs=1, 
                                        morgan_net=[int(x) for x in args.morgan_net.split(',')],
                                        task_ml = args.task_ml,
                                        interaction_net=[int(x) for x in args.interaction_net.split(',')],)
            # reset layers
            for layer in model.children():
                if hasattr(layer, 'reset_parameters'):
                    layer.reset_parameters()
            # load model
            model.to(rank)
            if os.path.isfile(args.snapshot_path):
                model, run_epochs = load_snapshot(model, snapshot_path, rank = '0')
            else:
                run_epochs = 0

            # get optimization, lr_scheduler, loss
            optim = get_optimization(args, model, optimization ='adam')
            scheduler = get_scheduler(args, optim, scheduler=args.scheduler)
            loss_fn = get_loss(args.task_ml)
            mins_eval = np.inf
            
            #train model
            for epoch in tqdm(range(run_epochs,args.n_epochs)):
                model.train()
                err_epoch, eval_mse, test_err = [], [], []
                for _, data in enumerate(tqdm(train_loader)):
                    optim.zero_grad()
                    y_ml = model(data.to(rank))
                    err = loss_fn(y_ml,data.y.to(rank).float()).cpu()
                    err.backward()
                    optim.step()
                    err_epoch.append(err.detach().cpu().item())
                print(f'epoch_loss: {np.mean(err_epoch)}')
                # eval model
                if epoch % args.eva_epochs == 0:
                    model.eval()
                    with torch.no_grad():
                        for _, data_val in enumerate(tqdm(val_loader)):
                            y_ml_val = model(data_val.to(rank))
                            err_val = loss_fn(y_ml_val, data_val.y.to(rank).float()).cpu()
                            eval_mse.append(err_val.item())
                        print(f'eval_loss: {np.mean(eval_mse)}')
                        if  np.mean(eval_mse) < mins_eval:
                            mins_eval = np.mean(eval_mse)
                            save_snapshot_single(epoch, model, snapshot_path = os.path.join(save_fold, "{}_{}_{}_{}best_checkpoint.pt".format(task,epoch,fold,result_name)))
                            epoch_best = epoch
                    if args.scheduler == 'Plateau':
                        scheduler.step(np.mean(eval_mse))
                    else:
                        scheduler.step()
                    wandb.log({"epoch_val_loss": np.mean(eval_mse), 
                                "epoch_loss": np.mean(err_epoch),
                                "learning_rate": optim.param_groups[0]["lr"]})
                else:
                    wandb.log({ "epoch_loss": np.mean(err_epoch),
                                "learning_rate": optim.param_groups[0]["lr"]})
            # test model and give inference
            snapshot_path = os.path.join(save_fold, "{}_{}_{}_{}best_checkpoint.pt".format(task,epoch_best,fold,result_name))
            model, run_epochs = load_snapshot(model, snapshot_path, rank = '0')
            model.eval()
            test_err = []
            predictions = []
            labels = []
            with torch.no_grad():
                for _, data_test in enumerate(tqdm(test_loader)):
                    y_ml_test = model(data_test.to(rank))
                    predictions = np.append(predictions,y_ml_test.detach().cpu().numpy())
                    labels = np.append(labels,data_test.y.detach().cpu().numpy())

            print('Final_result:{}'.format(np.mean(test_err)))

            final_result = {'labels':labels, 'predictions': predictions}
            df_test = pd.DataFrame.from_dict(final_result)
            df_test.to_csv('{}_{}_{}.csv'.format(task,fold,result_name))
            metrics(final_result['labels'], final_result['predictions'],'{}_{}_{}'.format(result_name,task,fold))
        df_metric = pd.DataFrame()
        df_metric['result'] = [metrics(final_result['labels'], final_result['predictions'],'{}_{}_{}'.format(result_name,task,fold))]
        df_metric.to_csv('metric{}_{}_{}.csv'.format(task,fold,result_name))

if __name__ == '__main__':
    
    world_size = torch.cuda.device_count()
    args = parse_train_args()
    main(args)