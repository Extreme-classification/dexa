import math
import torch
import numpy as np
from xclib.utils.sparse import normalize
from xclib.utils.clustering import cluster_balance, b_kmeans_dense


class AuxLayer(torch.nn.Module):
    """

    """
    def __init__(self, input_size, output_size,
                 output_size_org=None, mapping=None, device="cpu"):
        super(AuxLayer, self).__init__()
        assert math.log2(output_size).is_integer(), \
            "number of aux vectos must be power of 2"
        self.padding_idx = 0
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(
            torch.Tensor(output_size, input_size))
        self.device = device
        if mapping is not None:
            self.register_buffer('mapping', torch.LongTensor(mapping))
        else:
            self.register_buffer('mapping', torch.zeros(output_size_org).long())
        self.initialize()

    def cluster_and_set_mapping(self, X, num_threads=6):
        _, mapping = cluster_balance(
            X=normalize(X.astype('float32'), copy=True),
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=self.output_size,
            splitter=b_kmeans_dense,
            num_threads=num_threads,
            verbose=True)
        self.set_mapping(mapping)

    def set_mapping(self, mapping):
        self.mapping.copy_(
            torch.LongTensor(mapping).to(self.mapping.get_device()))

    def encode(self, x, ind):
        return x + self.weight[self.mapping[ind]].squeeze()
    
    def forward(self, x):
        return self.encode(*x)

    @property
    def repr_dims(self):
        return self.encoder.repr_dims

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.weight.data)

    def __repr__(self):
        s = '{name}(input_size: {input_size}, output_size: {output_size}, {device}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)
