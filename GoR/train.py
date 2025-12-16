import argparse

import torch.nn as nn
import dgl
from dgl.nn.pytorch.conv import GATConv
from dgl.nn.pytorch import JumpingKnowledge

from src.helper import *


class GAT(nn.Module):
    def __init__(self, in_dim, h_feats, dropout, attn_drop, n_head=4, num_layer=2):
        super(GAT, self).__init__()
        self.num_layer = num_layer
        self.n_head = n_head
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.act_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_dim, h_feats, num_heads=n_head, feat_drop=dropout, attn_drop=attn_drop, residual=False,
                    activation=None, allow_zero_in_degree=False))
        self.norm_layers.append(nn.BatchNorm1d(h_feats * n_head))
        self.act_layers.append(nn.PReLU(h_feats * n_head))
        for _ in range(num_layer - 1):
            self.gat_layers.append(
                GATConv(h_feats * n_head, h_feats, num_heads=n_head, feat_drop=dropout, attn_drop=attn_drop,
                        residual=False, activation=None, allow_zero_in_degree=False))
            self.norm_layers.append(nn.BatchNorm1d(h_feats * n_head))
            self.act_layers.append(nn.PReLU(h_feats * n_head))

        self.JKN = JumpingKnowledge(mode='max')

    def forward(self, g, in_feat):
        h = in_feat
        hidden_list = []
        for l in range(self.num_layer):
            h = self.gat_layers[l](g, h).reshape(in_feat.shape[0], -1)
            h = self.norm_layers[l](h)
            h = self.act_layers[l](h)
            hidden_list.append(torch.mean(h.reshape(in_feat.shape[0], self.n_head, -1), dim=1))

        ret = self.JKN(hidden_list)

        return ret


class GoR(nn.Module):
    def __init__(
            self,
            in_dim: int = 768,
            num_hidden: int = 768,
            num_layer: int = 2,
            n_head: int = 4,
            feat_drop: float = 0.2,
            attn_drop: float = 0.1,
    ):
        super(GoR, self).__init__()
        self.encoder = GAT(in_dim=in_dim, h_feats=num_hidden, dropout=feat_drop, attn_drop=attn_drop, n_head=n_head,
                           num_layer=num_layer)

    def lambda_mrr_loss(self, y_pred, y_true, padded_value_indicator=-1, reduction="mean"):
        """
        y_pred: FloatTensor [bz, topk]
        y_true: FloatTensor [bz, topk]
        """
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")
        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-50, max=50)
        scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0.)
        scores_diffs_exp = torch.exp(-scores_diffs)
        losses = torch.log(1. + scores_diffs_exp)

        if reduction == "sum":
            loss = torch.sum(losses[padded_pairs_mask])
        elif reduction == "mean":
            loss = torch.mean(losses[padded_pairs_mask])
        else:
            raise ValueError("Reduction method can be either sum or mean")

        return loss

    def forward(self, g, x, query_embedding_list, bert_score_list):
        node_rep = self.encoder(g, x)
        node_rep = torch.split(node_rep, g.batch_num_nodes().cpu().numpy().tolist(), dim=0)

        cl_loss_all = 0
        ranking_loss_all = 0
        entropy_all = 0
        """
        Note: We use a for loop to process each graph to avoid OOM. In GoR's training pipeline, there are actually two 
        "batch sizes", one is graph-level batch and the other is query-level batch. If the following for loop is 
        parallelized, the equivalent batch size is the product of the above two batch sizes, which is large and will 
        cause OOM on our computing devices. Nevertheless, if you have enough GPU Memory, you can parallelize it to 
        enable faster training.
        """
        for ind, (single_rep, query_embedding, bert_score) in enumerate(
                zip(node_rep, query_embedding_list, bert_score_list)):
            bert_score = bert_score.to(x.device)
            q = query_embedding.to(x.device)
            _, bert_sorted_idx = bert_score.sort(dim=-1, descending=True)
            p = single_rep[bert_sorted_idx[:, :1]]
            n = single_rep[bert_sorted_idx[:, 1:]]
            in_batch_neg_rep = torch.concat(node_rep[:ind] + node_rep[ind + 1:], dim=0).unsqueeze(0).repeat(p.shape[0],
                                                                                                            1, 1)
            n = torch.concat([n, in_batch_neg_rep], dim=1)
            q = q.unsqueeze(1)
            p_sim = torch.matmul(q, p.transpose(1, 2)).squeeze(1)
            n_sim = torch.matmul(q, n.transpose(1, 2)).squeeze(1)
            ranking_list = torch.concat([p_sim, n_sim], dim=-1)
            rank_score_prediction = ranking_list[:, :bert_sorted_idx.shape[-1]]
            rank_gt = 1 / torch.arange(1, 1 + rank_score_prediction.shape[-1]).view(1, -1).repeat(
                rank_score_prediction.shape[0], 1).to(x.device)
            ranking_loss_all += self.lambda_mrr_loss(rank_score_prediction, rank_gt)
            p_sim = torch.exp(p_sim / 1.0).sum(dim=-1)
            n_sim = torch.exp(n_sim / 1.0).sum(dim=-1)
            loss_cl = -torch.log(p_sim / (p_sim + n_sim))
            loss_cl = loss_cl.mean()
            cl_loss_all += loss_cl
            entropy_all += torch.distributions.Categorical(
                torch.softmax(torch.matmul(q.squeeze(1), single_rep.T), dim=-1)).entropy().mean()

        cl_loss_all /= len(query_embedding_list)
        ranking_loss_all /= len(query_embedding_list)
        entropy_all /= len(query_embedding_list)

        return cl_loss_all, ranking_loss_all, entropy_all


def train_gor(train_dataloader):
    model = GoR(in_dim=IN_DIM, num_hidden=HIDDEN_DIM, num_layer=NUM_LAYER, n_head=N_HEAD, feat_drop=DROPOUT)
    model.to(DEVICE)
    num_steps = len(train_dataloader) * MAX_EPOCH
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = lambda step: (1 + np.cos((step) * np.pi / num_steps)) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    for e in range(MAX_EPOCH):
        model.train()
        epoch_loss = 0
        entropy_loss = 0
        for batch_id, (g, query_embedding_l, bert_score_l) in enumerate(train_dataloader):
            g = g.to(DEVICE)
            cl_loss, ranking_loss, entropy = model(g, g.ndata['feat'], query_embedding_l, bert_score_l)
            loss = cl_loss + COE * ranking_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.detach().cpu()
            entropy_loss += entropy.detach().cpu()
        print('{} In epoch {}, lr: {:.5f}, loss: {:.4f}, entropy: {:.4f}'.format(show_time(), e,
                                                                                 optimizer.param_groups[0]['lr'],
                                                                                 float(epoch_loss / len(
                                                                                     train_dataloader)), float(
                entropy_loss / len(train_dataloader))))

    check_path("./weights")
    torch.save(model.state_dict(), "./weights/{}.pth".format(DATASET))


class GraphDataloader(dgl.data.DGLDataset):
    def __init__(self, query_embedding_list, gs_list, bert_score_list):
        self.query_embedding_list = query_embedding_list
        self.gs_list = gs_list
        self.bert_score_list = bert_score_list
        super(GraphDataloader, self).__init__(name="GraphDataloader")

    def process(self):
        pass

    def __getitem__(self, index):
        return self.gs_list[index], self.query_embedding_list[index], self.bert_score_list[index]

    def __len__(self):
        return int(len(self.gs_list))


def mix_collate_fn(batch):
    graph_data, query_embedding, bert_score = list(zip(*batch))
    graph_data = np.array(graph_data).flatten()
    graph_data = [dgl.add_self_loop(i) for i in graph_data]
    graph_data = dgl.batch(graph_data)

    query_embedding = [torch.vstack(q) for q in query_embedding]
    bert_score = [torch.from_numpy(bs) for bs in bert_score]

    return graph_data, query_embedding, bert_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epoch", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--in_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--drop", type=float, default=-1)
    parser.add_argument("--coe", type=float, default=-1)
    opt = parser.parse_args()
    DATASET = opt.dataset
    SEED = opt.seed
    DEVICE = opt.device
    DROPOUT = opt.drop
    COE = opt.coe
    BATCH_SIZE = opt.batch_size
    MAX_EPOCH = opt.max_epoch
    LR = opt.lr
    IN_DIM = opt.in_dim
    HIDDEN_DIM = opt.hidden_dim
    NUM_LAYER = opt.num_layer
    N_HEAD = opt.n_head

    if DEVICE == 'mps':
        print("DGL does not support MPS, using CPU instead")
        DEVICE = 'cpu'
        
    hyper_configuration = {
        "qmsum": {"dropout": 0.2, "coe": 0.9},
    }

    DROPOUT = hyper_configuration[DATASET]["dropout"] if DROPOUT == -1 else DROPOUT
    COE = hyper_configuration[DATASET]["coe"] if COE == -1 else COE

    set_seed(int(SEED))
    

    gs_list, _ = dgl.load_graphs("./training_data/{}_gs.dgl".format(DATASET))
    query_embedding_list = read_from_pkl(output_file="./training_data/{}_qe.pkl".format(DATASET))
    bert_score_list = read_from_pkl(output_file="./training_data/{}_bs.pkl".format(DATASET))

    train_dataset = GraphDataloader(query_embedding_list=query_embedding_list, gs_list=gs_list,
                                    bert_score_list=bert_score_list)
    train_dataloader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                       collate_fn=mix_collate_fn, num_workers=0, pin_memory=True)
    train_gor(train_dataloader=train_dataloader)