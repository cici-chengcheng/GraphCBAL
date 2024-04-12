import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.nn.parameter import Parameter
import torch.nn.init as init


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, batchsize, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batchsize = batchsize
        self.weight = nn.Parameter(torch.FloatTensor(batchsize, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(batchsize * out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # input(节点数, batchsize, 维数)  weight(batchsize, 输入维数, 输出维数)  adj(节点数, 节点数)
        support = torch.einsum("jik,ikp->jip", input, self.weight)  # XW  (节点数, batchsize, nhid)
        support = torch.reshape(support, [support.size(0), -1])  # (节点数, batchsize * nhid)
        # support = torch.mm(input, self.weight)
        if self.bias is not None:
            support = support + self.bias
        output = torch.spmm(adj, support)  # AXW  (节点数, batchsize * nhid)
        output = torch.reshape(output, [output.size(0), self.batchsize, -1])  # (节点数, batchsize, 维数)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, batchsize, dropout=0.5, softmax=True, bias=False):
        super(GCN, self).__init__()
        self.dropout_rate = dropout
        self.softmax = softmax
        self.batchsize = batchsize
        self.gc1 = GraphConvolution(nfeat, nhid, batchsize, bias=bias)
        self.gc2 = GraphConvolution(nhid, nclass, batchsize, bias=bias)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x, adj):
        x = x.expand([self.batchsize] + list(x.size())).transpose(0, 1)  # (节点数, batchsize, 特征数)
        x = self.dropout(x)
        x = F.relu(self.gc1(x, adj))  # (节点数, batchsize, nhid)
        x = self.dropout(x)
        x = self.gc2(x, adj)  # (节点数, batchsize, 类别数)
        x = x.transpose(0, 1).transpose(1, 2)  # (batchsize, 类别数, 节点数)
        if self.softmax:
            return F.log_softmax(x, dim=1), x  # (batchsize, 类别数, 节点数)
        else:
            return x.squeeze()

    def reset(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


class MLP_NORM(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta, norm_func_id, norm_layers,
                 orders, orders_func_id, cuda):
        super(MLP_NORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nnodes, nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.gamma = torch.tensor(gamma)
        self.delta = torch.tensor(delta)
        self.norm_layers = norm_layers
        self.orders = orders
        self.class_eye = torch.eye(nclass)
        self.nodes_eye = torch.eye(nnodes)
        self.cuda = cuda

        self.nnodes_ = nnodes
        self.nfeat_ = nfeat
        self.nhid_ = nhid
        self.nclass_ = nclass
        self.dropout_ = dropout
        self.alpha_ = alpha
        self.beta_ = beta
        self.gamma_ = gamma
        self.delta_ = delta
        self.norm_func_id_ = norm_func_id
        self.norm_layers_ = norm_layers
        self.orders_ = orders
        self.orders_func_id_ = orders_func_id
        self.cuda_ = cuda

        device = torch.device("cuda:" + str(cuda))
        #self.net = self.net.to(device)

        if 1:
            device = torch.device("cuda:" + str(0))
            print("here1")
            self.orders_weight = Parameter(
                torch.ones(self.orders, 1) / self.orders, requires_grad=True
            ).to(device)
            # use kaiming_normal to initialize the weight matrix in Orders3
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(self.nclass, self.orders), requires_grad=True
            ).to(device)
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(self.orders, self.orders), requires_grad=True
            ).to(device)
            # use diag matirx to initialize the second norm layer
            self.diag_weight = Parameter(
                torch.ones(self.nclass, 1) / self.nclass, requires_grad=True
            ).cuda()
            self.alpha = self.alpha.to(device)
            self.beta = self.beta.to(device)
            self.gamma = self.gamma.to(device)
            self.delta = self.delta.to(device)
            self.class_eye = self.class_eye.to(device)
            self.nodes_eye = self.nodes_eye.to(device)
        else:
            print("here2")
            self.orders_weight = Parameter(
                torch.ones(self.orders, 1) / self.orders, requires_grad=True
            )
            # use kaiming_normal to initialize the weight matrix in Orders3
            self.orders_weight_matrix = Parameter(
                torch.DoubleTensor(self.nclass, self.orders), requires_grad=True
            )
            self.orders_weight_matrix2 = Parameter(
                torch.DoubleTensor(self.orders, self.orders), requires_grad=True
            )
            # use diag matirx to initialize the second norm layer
            self.diag_weight = Parameter(
                torch.ones(self.nclass, 1) / self.nclass, requires_grad=True
            )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.elu = torch.nn.ELU()

        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def forward(self, x, adj):
        # x = x.expand([1]+list(x.size())).transpose(0,1)  # (node_num, batchsize, feat)
        xX = F.dropout(x, self.dropout, training=self.training)
        xX = self.fc1(x)  # (node_num, batchsize, nhid)
        xA = self.fc3(adj)  # (node_num, nhid)
        # xA = xA.expand([1]+list(xA.size())).transpose(0,1) # (node_num, batchsize, nhid)
        x = F.relu(self.delta * xX + (1 - self.delta) * xA)  # (node_num, batchsize, nhid)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)  # (node_num, batchsize, nclass)
        h0 = x
        for _ in range(self.norm_layers):
            # adj_drop = F.dropout(adj, self.dropout, training=self.training)
            x = self.norm(x, h0, adj)

        softmax = F.log_softmax(x, dim=1).unsqueeze(0)  # (1, node_num, nclass)
        return softmax.transpose(1, 2), x.unsqueeze(0).transpose(1, 2)  # (batchsize, 类别数, 节点数)
        # return x

    def reset(self):
        print("here3")
        new_net = MLP_NORM(nnodes=self.nnodes_, nfeat=self.nfeat_, nhid=self.nhid_, nclass=self.nclass_,
                           dropout=self.dropout_, alpha=self.alpha_, beta=self.beta_, gamma=self.gamma_,
                           delta=self.delta_,
                           norm_func_id=self.norm_func_id_, norm_layers=self.norm_layers_, orders=self.orders_,
                           orders_func_id=self.orders_func_id_,
                           cuda=self.cuda_)
        return new_net

        # self.orders_weight = Parameter(
        #     torch.ones(self.orders, 1) / self.orders, requires_grad=True
        # ).cuda()
        # # use kaiming_normal to initialize the weight matrix in Orders3
        # self.orders_weight_matrix = Parameter(
        #     torch.DoubleTensor(self.nclass, self.orders), requires_grad=True
        # ).cuda()
        # self.orders_weight_matrix2 = Parameter(
        #     torch.DoubleTensor(self.orders, self.orders), requires_grad=True
        # ).cuda()
        # # use diag matirx to initialize the second norm layer
        # self.diag_weight = Parameter(
        #     torch.ones(self.nclass, 1) / self.nclass, requires_grad=True
        # ).cuda()

        # init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        # init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        # self.elu = torch.nn.ELU()

    def norm_func1(self, x, h0, adj):
        # print('norm_func1 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
              self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        # print('norm_func2 run')
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        # u = torch.cholesky(coe2 * coe2 * torch.eye(self.nclass) + coe * res)
        # inv = torch.cholesky_inverse(u)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
              self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0

        # calculate z
        xx = torch.mm(x, x.t())
        hx = torch.mm(h0, x.t())
        # print('adj', adj.shape)
        # print('orders_weight', self.orders_weight[0].shape)
        adj = adj.to_dense()
        adjk = adj
        a_sum = adjk * self.orders_weight[0]
        for i in range(1, self.orders):
            adjk = torch.mm(adjk, adj)
            a_sum += adjk * self.orders_weight[i]
        z = torch.mm(coe1 * xx + self.beta * a_sum - self.gamma * coe1 * hx,
                     torch.inverse(coe1 * coe1 * xx + (self.alpha + self.beta) * self.nodes_eye))
        # print(z.shape)
        # print(z)
        return res

    def order_func1(self, x, res, adj):
        # Orders1
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # Orders2
        tmp_orders = torch.spmm(adj, res)
        #print('tmp_orders', tmp_orders.shape)
        #print('orders_weight', self.orders_weight[0].shape)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        # Orders3
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix.float())),
                               self.orders_weight_matrix2.float())
        # orders_para = torch.mm(x, self.orders_weight_matrix)
        orders_para = torch.transpose(orders_para, 0, 1)
        tmp_orders = torch.spmm(adj, res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            tmp_orders = torch.spmm(adj, tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders



