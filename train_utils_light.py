import time
from typing import List, Tuple
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score

from ogb.linkproppred import Evaluator
from torch_geometric.utils import negative_sampling

from tqdm import tqdm
from logger import Logger

def __elem2spm(element: torch.Tensor, sizes: List[int], val: torch.Tensor=None) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    if val is None:
        sp_tensor =  SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
            element.device).fill_value_(1.0)
    else:
        sp_tensor =  SparseTensor(row=row, col=col, value=val, sparse_sizes=sizes).to_device(
            element.device)
    return sp_tensor


def __spm2elem(spm: SparseTensor) -> torch.Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    val = spm.storage.value()
    return elem, val

def __spmdiff(adj1: SparseTensor,
             adj2: SparseTensor, keep_val=False) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''

    
    element1, val1 = __spm2elem(adj1)
    element2, val2 = __spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]
    
    if keep_val and val1 is not None:
        retval1 = val1[maskelem1]
        return __elem2spm(retelem1, adj1.sizes(), retval1)
    else:
        return __elem2spm(retelem1, adj1.sizes())


def get_train_test(args):
    if args.dataset == "ogbl-citation2":
        evaluator = Evaluator(name='ogbl-citation2')
        loggers = {
            'MRR': Logger(args.runs, args),
            # 'AUC': Logger(args.runs, args),
        }
        return train_mrr, test_mrr, evaluator, loggers
    else:
        evaluator = Evaluator(name='ogbl-ddi')
        loggers = {
            'Hits@10': Logger(args.runs, args),
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
            'AUC': Logger(args.runs, args),
        }
        return train_hits, test_hits, evaluator, loggers

def train_hits(encoder, predictor, data, split_edge, optimizer, batch_size, 
        mask_target, dataset_name, num_neg):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(device)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    if dataset_name == "ogbl-vessel":
        neg_edge_epoch = split_edge['train']['edge_neg'].to(device).t()
    elif dataset_name.startswith("ogbl") and dataset_name != "ogbl-ddi": # use global negative sampling for ddi
        num_pos_max = max(data.adj_t.nnz()//2, pos_train_edge.size(0))
        neg_edge_epoch = torch.randint(0, data.adj_t.size(0), 
                                       size=(2, num_pos_max*num_neg),
                                        dtype=torch.long, device=device)
    else:
        neg_edge_epoch = negative_sampling(data.edge_index, num_nodes=data.adj_t.size(0),
                                           num_neg_samples=(data.adj_t.nnz()//2)*num_neg)
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True),desc='Train'):
        edge = pos_train_edge[perm].t()
        if mask_target:
            adj_t = data.adj_t
            undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            adj_t = __spmdiff(adj_t, target_adj, keep_val=True)
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
def test_hits(encoder, predictor, data, split_edge, evaluator, 
         batch_size, fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)

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
        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')

    start_time = time.perf_counter()
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode='use'
    else:
        cache_mode=None
    
    pos_test_pred, neg_test_pred = test_split('test', cache_mode=cache_mode)
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


def make_symmetric(sparse_tensor, reduce='sum'):
    # Extract COO format
    indices = sparse_tensor.coalesce().indices()
    row, col = indices[0], indices[1]
    value = sparse_tensor.coalesce().values()

    # Concatenate the original and transposed entries
    all_row = torch.cat([row, col])
    all_col = torch.cat([col, row])
    all_value = torch.cat([value, value])

    # Create a new COO matrix with these entries
    new_indices = torch.stack([all_row, all_col])
    new_value = all_value

    # Remove duplicates by summing the values for symmetric entries
    unique_indices, inverse_indices = torch.unique(new_indices, dim=1, return_inverse=True)
    #unique_value = torch.zeros(unique_indices.size(1), device=value.device).scatter_add_(0, inverse_indices, new_value)
    unique_value = torch.zeros(unique_indices.size(1), device=value.device).scatter_reduce_(0, inverse_indices, new_value, reduce="amax")

    # Create the symmetric sparse tensor
    symmetric_tensor = torch.sparse_coo_tensor(unique_indices, unique_value, sparse_tensor.size())

    return symmetric_tensor

# Example usage
indices = torch.tensor([[0, 1, 2, 1, 3], [1, 2, 0, 0, 3]])  # COO format indices
values = torch.tensor([3, 4, 5, 3, 42], dtype=torch.float32)  # COO format values
size = (4, 4)  # Size of the matrix

sparse_tensor = torch.sparse_coo_tensor(indices, values, size)


def train_mrr(encoder, predictor, data, split_edge, optimizer, batch_size, 
        mask_target, dataset_name, num_neg):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = BCEWithLogitsLoss(reduction='mean')
    source_edge = split_edge['train']['source_node'].to(device)
    target_edge = split_edge['train']['target_node'].to(device)
    adjmask = torch.ones_like(source_edge, dtype=torch.bool)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(source_edge.size(0)), batch_size,
                           shuffle=True),desc='Train'):
        if mask_target:
            adjmask[perm] = 0
            tei = torch.stack((source_edge[adjmask], target_edge[adjmask]), dim=0) # TODO: check if both direction is removed
            adj_t = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   source_edge.device, non_blocking=True)
            adjmask[perm] = 1
    
            # TODO compare my make_sym against to_sym: sparse_tensor_coo vs adj_t.to_symmetric()
            sparse_tensor_coo  = torch.sparse_coo_tensor(tei, torch.ones((tei.size(1))), (data.num_nodes, data.num_nodes))
            sprase_tensor_coo = make_symmetric(sparse_tensor)

            adj_t = adj_t.to_symmetric()
        else:
            adj_t = data.adj_t


        h = encoder(data.x, adj_t)
        dst_neg = torch.randint(0, data.num_nodes, perm.size()*num_neg,
                                dtype=torch.long, device=device)

        edge = torch.stack((source_edge[perm], target_edge[perm]), dim=0)
        neg_edge = torch.stack((source_edge[perm].repeat(num_neg), dst_neg), dim=0)
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
def test_mrr(encoder, predictor, data, split_edge, evaluator, 
         batch_size, fast_inference):
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split, cache_mode=None):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h, adj_t, torch.stack((src, dst)), cache_mode=cache_mode).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h, adj_t, torch.stack((src, dst_neg)), cache_mode=cache_mode).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return pos_pred, neg_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')

    start_time = time.perf_counter()
    if fast_inference:
        # caching
        predictor(h, adj_t, None, cache_mode='build')
        cache_mode='use'
    else:
        cache_mode=None
    
    pos_test_pred, neg_test_pred = test_split('test', cache_mode=cache_mode)
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    if fast_inference:
        # delete cache
        predictor(h, adj_t, None, cache_mode='delete')
    
    valid_mrr = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })['mrr_list'].mean().item()
    test_mrr = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })['mrr_list'].mean().item()

    results = {
        "MRR": (valid_mrr, test_mrr),
    }

    return results