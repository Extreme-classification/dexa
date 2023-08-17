import torch


class AuxLayer(torch.nn.Module):
    """

    """
    def __init__(self, input_size, output_size, mapping=None, device="cpu"):
        super(AuxLayer, self).__init__()
        self.padding_idx = 0
        self.input_size = input_size
        self.output_size = output_size
        self.weight = torch.nn.Parameter(
            torch.Tensor(output_size, input_size))
        self.device = device
        if mapping is not None:
            self.mapping = torch.arange(output_size, dtype=torch.int64)
            # self.mapping = torch.LongTensor(mapping)
        else:
            self.mapping = None
        self.initialize()

    def set_mapping(self, mapping):
        self.mapping = torch.LongTensor(mapping)

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
