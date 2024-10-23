from argparse import ArgumentParser,FileType

def parse_train_args():

    # General arguments
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--result_name', type=str, default=None)
    parser.add_argument('--task_ml', type=str, default='classification')
    parser.add_argument('--snapshot_path', type=str, default='./bestmodel.pt',help='checkpoint')
    parser.add_argument('--eva_epochs', type=int, default=2, help='Eval model every eva_epochs')

    # # Training arguments
    parser.add_argument('--optimization', type=str, default='adam', help='optimization')
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=75, help='Batch size')
    parser.add_argument('--scheduler', type=str, default='ExponentialLR', help='LR scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience of the LR scheduler')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--w_decay', type=float, default=1e-2, help='Weight decay added to loss')
    parser.add_argument('--gamma', type=float, default=0.95, help='gamma for ExponentialLR')

    # # Model
    parser.add_argument('--num_conv_layers', type=int, default=3, help='Number of interaction layers')
    parser.add_argument('--lig_max_radius', type=float, default=5.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--rec_max_radius', type=float, default=30.0, help='Radius cutoff for geometric graph')
    parser.add_argument('--ns', type=int, default=16, help='Number of hidden features per node of order 0')
    parser.add_argument('--nv', type=int, default=4, help='Number of hidden features per node of order >0')
    parser.add_argument('--distance_embed_dim', type=int, default=16, help='Embedding size for the distance')
    parser.add_argument('--cross_distance_embed_dim', type=int, default=32, help='Embeddings size for the cross distance')
    parser.add_argument('--use_second_order_repr', type=str, default='1', help='Whether to use only up to first order representations or also second')
    parser.add_argument('--cross_max_distance', type=float, default=80, help='Maximum cross distance in case not dynamic')
    parser.add_argument('--dropout', type=float, default=0.9, help='MLP dropout')
    parser.add_argument('--morgan_net',type=str , default='2048,512,256', help='ECFP network')
    parser.add_argument('--interaction_net',type=str , default='256,256,256', help='interaction network')
    args = parser.parse_args()
    return args
