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

    def _power_two_check(self, n):
        assert math.log2(n).is_integer(), \
            "number of aux vectos must be power of 2"

    def _cluster_and_set_mapping_vanilla(self, X, num_threads=6):
        self._power_two_check(len(X))
        _, mapping = cluster_balance(
            X=normalize(X.astype('float32'), copy=True),
            clusters=[np.arange(len(X), dtype='int')],
            num_clusters=self.output_size,
            splitter=b_kmeans_dense,
            num_threads=num_threads,
            verbose=True)
        self.set_mapping(mapping)

    def _cluster_and_set_mapping_hlp(self, X, freq, num_hlpv, num_threads=6):
        mapping = np.full(len(X), fill_value=-1, dtype='int')
        indices = np.argsort(freq)

        vanilla_indices = indices[:-num_hlpv]
        hlp_indices = indices[-num_hlpv:]
        
        # assign first free vectors to head labels
        for m, ind in enumerate(hlp_indices):
            mapping[ind] = m

        # cluster only vanilla labels
        self._power_two_check(self.output_size - num_hlpv)
        _, vanilla_mapping = cluster_balance(
            X=normalize(X[vanilla_indices].astype('float32'), copy=True),
            clusters=[np.arange(len(vanilla_indices), dtype='int')],
            num_clusters=self.output_size - num_hlpv,
            splitter=b_kmeans_dense,
            num_threads=num_threads,
            verbose=True)

        # assign shared free vectors to clustered labels
        for i, m in zip(vanilla_indices, vanilla_mapping):
            # first few free vectors are reserved for head (hence +m)
            mapping[i] = num_hlpv + m
        self.set_mapping(mapping)

    def cluster_and_set_mapping(self, X, freq, num_hlpv, num_threads=6):
        if num_hlpv > 0:
            self._cluster_and_set_mapping_hlp(X, freq, num_hlpv, num_threads)
        else:
            self._cluster_and_set_mapping_vanilla(X, num_threads)


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
