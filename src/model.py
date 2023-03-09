import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, knn_graph
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import torch.nn.parallel
import torch.utils.data
from spatial_utils import *

class LayerNorm(nn.Module):
    """
    layer normalization
    Simple layer norm object optionally used with the convolutional encoder.
    """

    def __init__(self, feature_dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones((feature_dim,)))
        self.register_parameter("gamma", self.gamma)
        self.beta = nn.Parameter(torch.zeros((feature_dim,)))
        self.register_parameter("beta", self.beta)
        self.eps = eps

    def forward(self, x):
        # x: [batch_size, embed_dim]
        # normalize for each embedding
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # output shape is the same as x
        # Type not match for self.gamma and self.beta??????????????????????
        # output: [batch_size, embed_dim]
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

def get_activation_function(activation, context_str):
    if activation == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise Exception("{} activation not recognized.".format(context_str))


class SingleFeedForwardNN(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                    output_dim,
                    dropout_rate=None,
                    activation="sigmoid",
                    use_layernormalize=False,
                    skip_connection = False,
                    context_str = ''):
        '''
        Args:
            input_dim (int32): the input embedding dim
            output_dim (int32): dimension of the output of the network.
            dropout_rate (scalar tensor or float): Dropout keep prob.
            activation (string): tanh or relu or leakyrelu or sigmoid
            use_layernormalize (bool): do layer normalization or not
            skip_connection (bool): do skip connection or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(SingleFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if dropout_rate is not None:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = None

        self.act = get_activation_function(activation, context_str)

        if use_layernormalize:
            # the layer normalization is only used in the hidden layer, not the last layer
            self.layernorm = nn.LayerNorm(self.output_dim)
        else:
            self.layernorm = None

        # the skip connection is only possible, if the input and out dimention is the same
        if self.input_dim == self.output_dim:
            self.skip_connection = skip_connection
        else:
            self.skip_connection = False
        
        self.linear = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform(self.linear.weight)
        




    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size,..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        # Linear layer
        output = self.linear(input_tensor)
        # non-linearity
        output = self.act(output)
        # dropout
        if self.dropout is not None:
            output = self.dropout(output)

        # skip connection
        if self.skip_connection:
            output = output + input_tensor

        # layer normalization
        if self.layernorm is not None:
            output = self.layernorm(output)

        return output

class MultiLayerFeedForwardNN(nn.Module):
    """
        Creates a fully connected feed forward neural network.
        N fully connected feed forward NN, each hidden layer will use non-linearity, layer normalization, dropout
        The last layer do not have any of these
    """

    def __init__(self, input_dim,
                    output_dim,
                    num_hidden_layers=0,
                    dropout_rate=0.5,
                    hidden_dim=-1,
                    activation="relu",
                    use_layernormalize=True,
                    skip_connection = False,
                    context_str = None):
        '''
        Args:
            input_dim (int32): the input embedding dim
            num_hidden_layers (int32): number of hidden layers in the network, set to 0 for a linear network.
            output_dim (int32): dimension of the output of the network.
            dropout (scalar tensor or float): Dropout keep prob.
            hidden_dim (int32): size of the hidden layers
            activation (string): tanh or relu
            use_layernormalize (bool): do layer normalization or not
            context_str (string): indicate which spatial relation encoder is using the current FFN
        '''
        super(MultiLayerFeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_layernormalize = use_layernormalize
        self.skip_connection = skip_connection
        self.context_str = context_str

        self.layers = nn.ModuleList()
        if self.num_hidden_layers <= 0:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))
        else:
            self.layers.append( SingleFeedForwardNN(input_dim = self.input_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            for i in range(self.num_hidden_layers-1):
                self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.hidden_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = self.use_layernormalize,
                                                    skip_connection = self.skip_connection,
                                                    context_str = self.context_str))

            self.layers.append( SingleFeedForwardNN(input_dim = self.hidden_dim,
                                                    output_dim = self.output_dim,
                                                    dropout_rate = self.dropout_rate,
                                                    activation = self.activation,
                                                    use_layernormalize = False,
                                                    skip_connection = False,
                                                    context_str = self.context_str))

        

    def forward(self, input_tensor):
        '''
        Args:
            input_tensor: shape [batch_size, ..., input_dim]
        Returns:
            tensor of shape [batch_size, ..., output_dim]
            note there is no non-linearity applied to the output.
        Raises:
            Exception: If given activation or normalizer not supported.
        '''
        assert input_tensor.size()[-1] == self.input_dim
        output = input_tensor
        for i in range(len(self.layers)):
            output = self.layers[i](output)

        return output

def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) / (frequency_num*1.0 - 1))
        timescales = min_radius * np.exp(np.arange(frequency_num).astype(float) * log_timescale_increment)
        freq_list = 1.0/timescales
    return freq_list

class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """
    def __init__(self, spa_embed_dim, coord_dim = 2, frequency_num = 16, 
            max_radius =0.01, min_radius = 0.00001,
            freq_init = "geometric",
            ffn=None):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim 
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.ffn = ffn
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()
        self.input_embed_dim = self.cal_input_dim()

        if self.ffn is not None:
          self.ffn = MultiLayerFeedForwardNN(2 * frequency_num * 2, spa_embed_dim)

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)


    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis = 1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis = 1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")
        
        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis = 4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis = 3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis = 4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat
        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1
        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))
        return spr_embeds

    def forward(self, coords):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            coords: a python list with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """   
        spr_embeds = self.make_input_embeds(coords)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)
        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:
            return spr_embeds

class GCN(nn.Module):
    """
        GCN
    """
    def __init__(self, num_features_in=3, num_features_out=1, k=20, MAT=False):
        super(GCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k = k
        self.MAT = MAT
        self.conv1 = GCNConv(num_features_in, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)
        if MAT:
          self.fc_morans = nn.Linear(32, num_features_out)
    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
          edge_index = ei
          edge_weight = ew
        else:
          edge_index = knn_graph(c, k=self.k).to(self.device)
          edge_weight = makeEdgeWeight(c, edge_index).to(self.device)
        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        if self.MAT:
          morans_output = self.fc_morans(h2)
          return output, morans_output
        else:
          return output

class PEGCN(nn.Module):
    """
        GCN with positional encoder and auxiliary tasks
    """
    def __init__(self, num_features_in=3, num_features_out=1, emb_hidden_dim=128, emb_dim=16, k = 20, MAT=False):
        super(PEGCN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb_hidden_dim = emb_hidden_dim
        self.emb_dim = emb_dim
        self.k = k
        self.MAT = MAT
        self.spenc = GridCellSpatialRelationEncoder(spa_embed_dim=emb_hidden_dim,ffn=True,min_radius=1e-06,max_radius=360)
        self.dec = nn.Sequential(
            nn.Linear(emb_hidden_dim, emb_hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 2, emb_hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(emb_hidden_dim // 4, emb_dim)
        )
        self.conv1 = GCNConv(num_features_in + emb_dim, 32)
        self.conv2 = GCNConv(32, 32)
        self.fc = nn.Linear(32, num_features_out)
        if MAT:
          self.fc_morans = nn.Linear(32, num_features_out)
    def forward(self, x, c, ei, ew):
        x = x.float()
        c = c.float()
        if torch.is_tensor(ei) & torch.is_tensor(ew):
          edge_index = ei
          edge_weight = ew
        else:
          edge_index = knn_graph(c, k=self.k).to(self.device)
          edge_weight = makeEdgeWeight(c, edge_index).to(self.device)

        c = c.reshape(1, c.shape[0], c.shape[1])
        emb = self.spenc(c.detach().cpu().numpy())
        emb = emb.reshape(emb.shape[1],emb.shape[2])
        emb = self.dec(emb).float()
        x = torch.cat((x,emb),dim=1)

        h1 = F.relu(self.conv1(x, edge_index, edge_weight))
        h1 = F.dropout(h1, training=self.training)
        h2 = F.relu(self.conv2(h1, edge_index, edge_weight))
        h2 = F.dropout(h2, training=self.training)
        output = self.fc(h2)
        if self.MAT:
          morans_output = self.fc_morans(h2)
          return output, morans_output
        else:
          return output

class LossWrapper(nn.Module):
    def __init__(self, model, task_num=1, loss='mse', uw=True, lamb=0.5, k=20, batch_size=2048):
        super(LossWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.task_num = task_num
        self.uw = uw
        self.lamb = lamb
        self.k = k
        self.batch_size = batch_size
        if task_num > 1:
          self.log_vars = nn.Parameter(torch.zeros((task_num)))
        if loss=="mse":
          self.criterion = nn.MSELoss()
        elif loss=="l1":
          self.criterion = nn.L1Loss()

    def forward(self, input, targets, coords, edge_index, edge_weight, morans_input):

        if self.task_num==1:
          outputs = self.model(input, coords, edge_index, edge_weight)
          loss = self.criterion(targets.float().reshape(-1),outputs.float().reshape(-1))
          return loss

        else:
          outputs1, outputs2 = self.model(input, coords, edge_index, edge_weight)
          if torch.is_tensor(morans_input):
            targets2 = morans_input
          else:
            moran_weight_matrix = knn_to_adj(knn_graph(coords, k=self.k), self.batch_size) 
            with torch.enable_grad():
              targets2 = lw_tensor_local_moran(targets, sparse.csr_matrix(moran_weight_matrix)).to(self.device)
          if self.uw:
            precision1 = 0.5 * torch.exp(-self.log_vars[0])
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss1 = torch.sum(precision1 * loss1 + self.log_vars[0], -1)

            precision2 = 0.5 * torch.exp(-self.log_vars[1])
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss2 = torch.sum(precision2 * loss2 + self.log_vars[1], -1)

            loss = loss1 + loss2
            loss = torch.mean(loss)
            return loss, self.log_vars.data.tolist()
          else:
            loss1 = self.criterion(targets.float().reshape(-1),outputs1.float().reshape(-1))
            loss2 = self.criterion(targets2.float().reshape(-1),outputs2.float().reshape(-1))
            loss = loss1 + self.lamb * loss2
            return loss        