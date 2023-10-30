import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import Evaluator, PygLinkPropPredDataset
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_geometric.utils import (degree,
                                   negative_sampling)
from torch_sparse import SparseTensor
# from tqdm import tqdm

from logger import Logger
from models import GCN, MLP, SAGE, LinkPredictor, MPLP
from node_label import spmdiff_
from utils import ( get_dataset, data_summary,
                   set_random_seeds, str2bool, get_data_split, initial_embedding)


def train(encoder, predictor, data, split_edge, optimizer, batch_size, 
        mask_target, dataset_name):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(device)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    if dataset_name.startswith("ogbl"):
        neg_edge_epoch = torch.randint(0, data.adj_t.size(0), data.edge_index.size(), dtype=torch.long,
                             device=device)
    else:
        neg_edge_epoch = negative_sampling(data.edge_index, num_nodes=data.adj_t.size(0))
    # for perm in (pbar := tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
    #                        shuffle=True)) ):
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        edge = pos_train_edge[perm].t()
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
        else:
            adj_t = data.adj_t


        h = encoder(data.x, adj_t)

        neg_edge = neg_edge_epoch[:,perm]
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(device)
        out = predictor(h, adj_t, train_edges).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            torch.nn.utils.clip_grad_norm_(data.x, 1.0)
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)
    
    return total_loss / total_examples


@torch.no_grad()
def test(encoder, predictor, data, split_edge, evaluator, 
         batch_size, use_valedges_as_input, fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    # pos_train_edge = split_edge['train']['edge'].to(device)
    pos_valid_edge = split_edge['valid']['edge'].to(device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(device)
    pos_test_edge = split_edge['test']['edge'].to(device)
    neg_test_edge = split_edge['test']['edge_neg'].to(device)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        out = predictor(h, adj_t, edge)
        pos_valid_preds += [out.squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h, adj_t, edge).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    start_time = time.perf_counter()
    if use_valedges_as_input:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode='use'
    else:
        cache_mode=None
    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        out = predictor(h, adj_t, edge, cache_mode=cache_mode)
        pos_test_preds += [out.squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, adj_t, edge, cache_mode=cache_mode).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')
    
    results = {}
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)


    valid_result = torch.cat((torch.ones(pos_valid_pred.size()), torch.zeros(neg_valid_pred.size())), dim=0)
    valid_pred = torch.cat((pos_valid_pred, neg_valid_pred), dim=0)

    test_result = torch.cat((torch.ones(pos_test_pred.size()), torch.zeros(neg_test_pred.size())), dim=0)
    test_pred = torch.cat((pos_test_pred, neg_test_pred), dim=0)

    results['AUC'] = (roc_auc_score(valid_result.cpu().numpy(),valid_pred.cpu().numpy()),roc_auc_score(test_result.cpu().numpy(),test_pred.cpu().numpy()))

    return results

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    # dataset setting
    parser.add_argument('--dataset', type=str, default='collab')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--use_valedges_as_input', type=str2bool, default='False', help='whether to use val edges as input')
    parser.add_argument('--year', type=int, default=-1)

    # MPLP settings
    parser.add_argument('--signature_dim', type=int, default=1024, help="the node signature dimension `F` in MPLP")
    parser.add_argument('--minimum_degree_onehot', type=int, default=-1, help='the minimum degree of hubs with onehot encoding to reduce variance')
    parser.add_argument('--mask_target', type=str2bool, default='True', help='whether to mask the target edges to remove the shortcut')
    parser.add_argument('--use_degree', type=str, default='none', choices=["none","mlp","AA","RA"], help="rescale vector norm to facilitate weighted count")
    
    # model setting
    parser.add_argument('--encoder', type=str, default='gcn')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--xdp', type=float, default=0.2)
    parser.add_argument('--feat_dropout', type=float, default=0.5)
    parser.add_argument('--label_dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--predictor', type=str, default='MPLP+combine', choices=["inner","mlp","ENL","MPLP+exact","MPLP+prop_only","MPLP+combine"])
    parser.add_argument('--use_feature', type=str2bool, default='True', help='whether to use node features as input')
    parser.add_argument('--feature_combine', type=str, default='hadamard', choices=['hadamard','plus_minus'], help='how to represent a link with two nodes features')
    parser.add_argument('--jk', type=str2bool, default='True', help='whether to use Jumping Knowledge')
    parser.add_argument('--batchnorm_affine', type=str2bool, default='True', help='whether to use Affine in BatchNorm')
    parser.add_argument('--use_embedding', type=str2bool, default='False', help='whether to train node embedding')
    # parser.add_argument('--dgcnn', type=str2bool, default='False', help='whether to use DGCNN as the target edge pooling')
    parser.add_argument('--torchhd_style', type=str2bool, default='True', help='whether to use torchhd to randomize vectors')
    parser.add_argument('--fast_inference', type=str2bool, default='False', help='whether to enable a faster inference by caching the node vectors')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--num_hops', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100, help='number of patience steps for early stopping')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--metric', type=str, default='Hits@50', help='main evaluation metric')

    # misc
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--data_split_only', type=str2bool, default='False')
    parser.add_argument('--print_summary', type=str, default='')

    args = parser.parse_args()
    # start time
    start_time = time.time()
    set_random_seeds(234)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    # device = torch.device('cpu')

    data, split_edge = get_dataset(args.dataset_dir, args.dataset, args.use_valedges_as_input, args.year)
    if data.x is None:
        args.use_feature = False

    if args.print_summary:
        data_summary(args.dataset, data, header='header' in args.print_summary, latex='latex' in args.print_summary);exit(0)
    else:
        print(args)
    final_log_path = Path(args.log_dir) / f"{args.dataset}_jobID_{os.getenv('JOB_ID','None')}_PID_{os.getpid()}_{int(time.time())}.log"
    final_log_path.mkdir(parents=True, exist_ok=True)
    with open(final_log_path, 'w') as f:
        print(args, file=f)
    
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(final_log_path, 'a') as f:
        f.write('\n' + cmd_input)
    
    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
        'AUC': Logger(args.runs, args),
    }

    val_max = 0.0
    for run in range(args.runs):
        if not args.dataset.startswith('ogbl-'):
            data, split_edge = get_data_split(args.dataset_dir, args.dataset, args.val_ratio, args.test_ratio, run=run)
            data = T.ToSparseTensor(remove_edge_index=False)(data)
            # Use training + validation edges for inference on test set.
            if args.use_valedges_as_input:
                val_edge_index = split_edge['valid']['edge'].t()
                full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
                data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, 
                                                        sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
                data.full_adj_t = data.full_adj_t.to_symmetric()
            else:
                data.full_adj_t = data.adj_t
            if args.data_split_only:
                if run == args.runs - 1:
                    exit(0)
                else:
                    continue
        
        if args.minimum_degree_onehot > 0:
            d_v = degree(data.edge_index[0],data.num_nodes)
            nodes_to_one_hot = d_v >= args.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            print(f"number of nodes to onehot: {int(one_hot_dim)}")
        data = data.to(device)
        if args.use_embedding:
            emb = initial_embedding(data, args.hidden_channels, device)
        else:
            emb = None
        if 'gcn' in args.encoder:
            encoder = GCN(data.num_features, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.feat_dropout, args.xdp, args.use_feature, args.jk, args.encoder, emb).to(device)
        elif args.encoder == 'sage':
            encoder = SAGE(data.num_features, args.hidden_channels,
                        args.hidden_channels, args.num_layers,
                        args.feat_dropout, args.xdp, args.use_feature, args.jk, emb).to(device)
        elif args.encoder == 'mlp':
            encoder = MLP(args.num_layers, data.num_features, 
                          args.hidden_channels, args.hidden_channels, args.dropout).to(device)

        predictor_in_dim = args.hidden_channels * int(args.use_feature or args.use_embedding)
                            # * (1 + args.jk * (args.num_layers - 1))
        if args.predictor in ['inner','mlp']:
            predictor = LinkPredictor(args.predictor, predictor_in_dim, args.hidden_channels, 1,
                                    args.num_layers, args.feat_dropout).to(device)
        # elif args.predictor == 'ENL':
        #     predictor = NaiveNodeLabelling(predictor_in_dim, args.hidden_channels,
        #                             args.num_layers, args.feat_dropout, args.num_hops, 
        #                             dgcnn=args.dgcnn, use_degree=args.use_degree).to(device)
        elif 'MPLP' in args.predictor:
            prop_type = args.predictor.split("+")[1]
            predictor = MPLP(predictor_in_dim, args.hidden_channels,
                                    args.num_layers, args.feat_dropout, args.label_dropout, args.num_hops, 
                                    prop_type=prop_type, torchhd_style=args.torchhd_style,
                                    use_degree=args.use_degree, signature_dim=args.signature_dim,
                                    minimum_degree_onehot=args.minimum_degree_onehot, batchnorm_affine=args.batchnorm_affine,
                                    feature_combine=args.feature_combine).to(device)

        encoder.reset_parameters()
        predictor.reset_parameters()
        parameters = list(encoder.parameters()) + list(predictor.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')

        cnt_wait = 0
        best_val = 0.0

        for epoch in range(1, 1 + args.epochs):
            loss = train(encoder, predictor, data, split_edge,
                         optimizer, args.batch_size, args.mask_target, args.dataset)

            results = test(encoder, predictor, data, split_edge,
                            evaluator, args.batch_size, args.use_valedges_as_input, args.fast_inference)

            if results[args.metric][0] >= best_val:
                best_val = results[args.metric][0]
                cnt_wait = 0
            else:
                cnt_wait +=1

            for key, result in results.items():
                loggers[key].add_result(run, result)

            if epoch % args.log_steps == 0:
                for key, result in results.items():
                    valid_hits, test_hits = result
                    to_print = (f'Run: {run + 1:02d}, ' +
                            f'Epoch: {epoch:02d}, '+
                            f'Loss: {loss:.4f}, '+
                            f'Valid: {100 * valid_hits:.2f}%, '+
                            f'Test: {100 * test_hits:.2f}%')
                    print(key)
                    print(to_print)
                    with open(final_log_path, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)
                print('---')

            if cnt_wait >= args.patience:
                break

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(final_log_path, 'a') as f:
                print(key,file=f)
                loggers[key].print_statistics(run=run, file=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(final_log_path, 'a') as f:
            print(key,file=f)
            loggers[key].print_statistics(file=f)
    # end time
    end_time = time.time()
    with open(final_log_path, 'a') as f:
        print(f"Total time: {end_time - start_time:.4f}s", file=f)

if __name__ == "__main__":
    main()
