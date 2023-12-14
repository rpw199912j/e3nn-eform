import torch
import torch_geometric
import torch_scatter

from e3nn.nn.models.v2103.gate_points_networks import SimpleNetwork
from typing import Union, Dict


class SimplePeriodicNetwork(SimpleNetwork):
    def __init__(self, **kwargs):
        """The keyword `pool_nodes` is used by SimpleNetwork to determine
        whether we sum over all atom contributions per example. In this
        example, we want use a mean operations instead,
        so we will override this behavior.
        """
        self.pool = False
        if kwargs['pool_nodes'] is True:
            kwargs['pool_nodes'] = False
            kwargs['num_nodes'] = 1.
            self.pool = True
        super().__init__(**kwargs)

    # Overwriting preprocess method of
    # SimpleNetwork to adapt for periodic boundary data
    def preprocess(self,
                   data: Union[torch_geometric.data.Data,
                               Dict[str, torch.Tensor]]) -> torch.Tensor:
        if 'batch' in data:
            batch = data['batch']
        else:
            batch = data['pos'].new_zeros(data['pos'].shape[0],
                                          dtype=torch.long)

        edge_src = data['edge_index'][0]  # Edge source
        edge_dst = data['edge_index'][1]  # Edge destination

        # We need to compute this in the computation graph to backprop to
        # positions. We are computing the relative distances + unit cell
        # shifts from periodic boundaries
        edge_vec = data["edge_attr"].double()

        return batch, data['x'], edge_src, edge_dst, edge_vec

    def forward(self,
                data: Union[torch_geometric.data.Data,
                            Dict[str, torch.Tensor]]) -> torch.Tensor:
        # if pool_nodes was set to True, use scatter_mean to aggregate
        output = super().forward(data)
        if self.pool is True:
            # Take mean over atoms per example
            return torch_scatter.scatter_mean(output, data.batch, dim=0)
        else:
            return output
