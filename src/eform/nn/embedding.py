import torch

from eform.nn.basis import bessel
from functools import partial
from torch_geometric.data import Data


class InitialEmbedding(torch.nn.Module):
    def __init__(self, num_species, cutoff, embed_dim=8, num_basis=16):
        super().__init__()
        self.embed_node_x = torch.nn.Embedding(num_species, embed_dim)
        self.embed_node_z = torch.nn.Embedding(num_species, embed_dim)
        self.embed_edge = partial(bessel, start=0.0, end=cutoff,
                                  num_basis=num_basis)

    def forward(self, data: Data):
        # Embed node
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))

        return data
