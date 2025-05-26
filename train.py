import os
import numpy as np
import pandas as pd
import torch
import wandb
from time import ctime
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch_geometric

from common.ultils import *
from dataset import CPI3DDataset
from model import TensorProductModel_one_hot3
from common.parsing import parse_train_args

# Environment settings
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

os.environ.update({
    'CUDA_LAUNCH_BLOCKING': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1'
})
torch.set_num_threads(1)

def prepare_data_loaders(args, task, fold):
    if args.task_ml == 'classification':
        train_dataset = CPI3DDataset(processed_data_pt=args.train_path)
        test_dataset = CPI3DDataset(processed_data_pt=args.test_path)

        indices = np.random.permutation(len(train_dataset))
        split = int(0.2 * len(train_dataset))
        val_indices, train_indices = indices[:split], indices[split:]

        train_loader = DataLoader(train_dataset[train_indices], batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(train_dataset[val_indices], batch_size=args.batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        base_path = args.processed_path_template.format(task, fold)
        train_loader = DataLoader(CPI3DDataset(processed_data_pt=base_path.format('train')), batch_size=args.batch_size, drop_last=True)
        val_loader = DataLoader(CPI3DDataset(processed_data_pt=base_path.format('val')), batch_size=args.batch_size)
        test_loader = DataLoader(CPI3DDataset(processed_data_pt=base_path.format('test')), batch_size=args.batch_size)

    return train_loader, val_loader, test_loader

def main(args):
    wandb.login()
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(2024)

    project_name = f"{ctime().replace(' ', '_').replace(':', '_')}_{args.model_name}_{args.task}_{args.result_name}"
    wandb.init(dir=args.wandb_dir, project=project_name, config=vars(args))

    save_dir = './checkpoint_dir'
    os.makedirs(save_dir, exist_ok=True)

    for fold in range(5):
        train_loader, val_loader, test_loader = prepare_data_loaders(args, args.task, fold)

        model = TensorProductModel_one_hot3(
            device=rank,
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
            use_second_order_repr='1',
            batch_norm=True,
            dropout=args.dropout,
            confidence_dropout=args.dropout,
            confidence_no_batchnorm=False,
            num_confidence_outputs=1,
            morgan_net=[int(x) for x in args.morgan_net.split(',')],
            task_ml=args.task_ml,
            interaction_net=[int(x) for x in args.interaction_net.split(',')],
        ).to(rank)

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        snapshot_path = os.path.join(save_dir, f"{args.task}_{fold}_best_checkpoint.pt")
        if os.path.isfile(snapshot_path):
            model, run_epochs = load_snapshot(model, snapshot_path, rank='0')
        else:
            run_epochs = 0

        optimizer = get_optimization(args, model, optimization='adam')
        scheduler = get_scheduler(args, optimizer, scheduler=args.scheduler)
        loss_fn = get_loss(args.task_ml)
        best_val_loss = float('inf')

        for epoch in tqdm(range(run_epochs, args.n_epochs)):
            model.train()
            train_losses = []

            for data in tqdm(train_loader):
                optimizer.zero_grad()
                outputs = model(data.to(rank))
                loss = loss_fn(outputs, data.y.to(rank).float())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)
            wandb.log({"train_loss": avg_train_loss, "lr": optimizer.param_groups[0]['lr']})

            if epoch % args.eva_epochs == 0:
                model.eval()
                val_losses = []

                with torch.no_grad():
                    for val_data in val_loader:
                        val_outputs = model(val_data.to(rank))
                        val_loss = loss_fn(val_outputs, val_data.y.to(rank).float())
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                wandb.log({"val_loss": avg_val_loss})

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    save_snapshot_single(epoch, model, snapshot_path)

                scheduler.step(avg_val_loss if args.scheduler == 'Plateau' else None)

        # Final evaluation
        model, _ = load_snapshot(model, snapshot_path, rank='0')
        model.eval()

        predictions, labels = [], []
        with torch.no_grad():
            for data in test_loader:
                preds = model(data.to(rank)).detach().cpu().numpy()
                targets = data.y.detach().cpu().numpy()
                predictions.extend(preds)
                labels.extend(targets)

        result_df = pd.DataFrame({'labels': labels, 'predictions': predictions})
        result_df.to_csv(f'{args.task}_{fold}_{args.result_name}.csv', index=False)

        metrics_output = metrics(labels, predictions, f'{args.result_name}_{args.task}_{fold}')
        pd.DataFrame({'result': [metrics_output]}).to_csv(f'metric_{args.task}_{fold}_{args.result_name}.csv', index=False)

if __name__ == '__main__':
    args = parse_train_args()
    main(args)