import numpy as np
import plotly.graph_objs as go
import os
import torch
import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')
import ovito._extensions.pyscript

from glob import glob
from torch_geometric.data import InMemoryDataset, Data
from ovito.data import CutoffNeighborFinder, DataCollection
from ovito.io import import_file
from ovito.modifiers import AffineTransformationModifier
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


class PeriodicCrystal(InMemoryDataset):
    def __init__(
        self, root=None,
        transform=None, pre_transform=None, pre_filter=None, log=False,
        nn_cutoff=3, tasks=["reg"],
        file_ids=[0],
        col_reshape: bool = False
    ):
        self.nn_cutoff = nn_cutoff
        self.tasks = tasks
        self.file_ids = file_ids
        self.col_reshape = col_reshape
        super().__init__(root, transform, pre_transform, pre_filter, log)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            f"crystals/Fe2O3.data"
        ]

    @property
    def processed_file_names(self):
        return ["crystals/crystal_processed.pt"]

    @staticmethod
    def plotly_vis(data: Data, center_atom_idx: int = -1):
        """Visualize the dislocation within the simulation cell"""
        atoms = data.pos.numpy()
        # generate atom indices for hover labels
        num_atoms = atoms.shape[0]
        atom_ids = torch.arange(0, num_atoms, dtype=torch.int32)
        # generate the plotly trace for all atoms
        atoms_trace = go.Scatter3d(
            x=atoms[:, 0], y=atoms[:, 1], z=atoms[:, 2],
            name="Perfect FCC [0, 1]", mode="markers",
            customdata=atom_ids,
            hovertemplate="<b>Atom %{customdata:d}<br>" +
                          "x: %{x:.3f}<br>" +
                          "y: %{y:.3f}<br>" +
                          "z: %{z:.3f}<br>",
            marker=dict(
                symbol='circle',
                size=6,
                color='rgb(244,244,244)',
                line=dict(
                    color='rgb(50,50,50)',
                    width=0.5)
            )
        )

        # Visualize NN for a selected atom
        # Set up an adjacency matrix
        adj_mat = torch.zeros(size=(num_atoms, num_atoms), dtype=torch.int32)
        src, dst = data.edge_index[0, :], data.edge_index[1, :]
        adj_mat[src, dst] = 1
        # find the nearest neighbors within the cutoff
        neighbor_ids = torch.nonzero(adj_mat[center_atom_idx, :]).flatten()
        highlight_ids = torch.cat([torch.tensor([center_atom_idx]),
                                   neighbor_ids])
        num_neighbors = neighbor_ids.shape[0]
        highlight_trace = go.Scatter3d(
            x=atoms[highlight_ids, 0],
            y=atoms[highlight_ids, 1],
            z=atoms[highlight_ids, 2],
            name="Highlighted Atoms", mode="markers",
            customdata=highlight_ids,
            hovertemplate="<b>Atom %{customdata:d}<br>" +
                          "x: %{x:.3f}<br>" +
                          "y: %{y:.3f}<br>" +
                          "z: %{z:.3f}<br>",
            marker=dict(
                symbol='circle',
                size=6,
                color=['rgb(255,128,0)'] + ['rgb(0,153,0)'] * num_neighbors,
                line=dict(
                    color='rgb(50,50,50)',
                    width=0.5)
            )
        )

        # set the plot aesthetics
        axis = dict(
            showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
        )
        # Setup the layout
        layout = go.Layout(
            title=dict(text="Input atoms"),
            showlegend=True,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
                camera=dict(
                    projection=dict(
                        type="orthographic"
                    )
                )
            ),
            hovermode='closest',

        )
        traces = [
            atoms_trace,
            highlight_trace,
        ]
        fig = go.Figure(data=traces, layout=layout)
        # TODO: Display source file name?
        return fig

    @staticmethod
    def permute_particles(frame: int, data: DataCollection):
        """Randomly permute the ordering of particles without
        changing the geometry
        """
        # Get the particles and their unique identifiers
        particles = data.particles_
        particle_positions = particles.positions_
        particle_identifiers = particles.identifiers_
        # Get a random number generator to shuffle atoms
        from numpy.random import default_rng
        rng = default_rng()
        shuffle_ids = rng.permutation(x=particle_positions.shape[0])
        particle_positions[:] = particle_positions.array[shuffle_ids]
        particle_identifiers[:] = particle_identifiers.array[shuffle_ids]

    def process_dump(self, dump_file: str,
                     file_idx: int,
                     tasks=["reg"],
                     trans_mat=[[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0]],
                     permute: bool = False):
        """Processing pipeline for a single dump file"""
        # TODO: need to find correpsonding energy
        eform_per_atom = 0
        pipeline = import_file(dump_file)
        # Apply transformation matrix to the input simulation cell
        # Note: the default trans_mat leave the input cell untouched
        cell_modifier = AffineTransformationModifier(
            operate_on={'particles', 'cell'},
            transformation=trans_mat
        )
        pipeline.modifiers.append(cell_modifier)
        # If needed, permute the ordering of the atoms
        if permute:
            pipeline.modifiers.append(self.permute_particles)
        # Run OVITO analysis pipeline
        data = pipeline.compute()

        # Get the particle positions
        atom_pos = data.particles.positions.array

        # find all atoms within a cutoff distance (in units of Angstrom)
        nn_finder = CutoffNeighborFinder(cutoff=self.nn_cutoff,
                                         data_collection=data)
        nn_ids, nn_vecs = nn_finder.find_all()

        num_nodes = atom_pos.shape[0]
        # Store atom positions
        atom_pos = torch.tensor(atom_pos, dtype=torch.float32)
        # Store the relative distance between atoms as edges
        edges = torch.tensor(nn_ids, dtype=torch.long).t().contiguous()
        shift_vecs = torch.tensor(nn_vecs, dtype=torch.float32)
        # Use the node features to store element type
        # For now, we don't distinguish element types
        x = OneHotEncoder().fit_transform(
            np.zeros(shape=num_nodes, dtype=int)
        )
        x = torch.tensor(x).long()
        if self.col_reshape:
            x = x.reshape(-1, 1).double()

        return Data(
            x=x,
            y=eform_per_atom,
            pos=atom_pos,
            edge_index=edges,
            edge_attr=shift_vecs
        )

    def process(self):
        data_list = []
        pbar = tqdm(self.raw_paths)
        for i, fpath in enumerate(pbar):
            fname = fpath.split("/")[-1]
            pbar.set_description("Processing %s" % fname)
            try:
                data_list.append(self.process_dump(fpath, i, tasks=self.tasks))
            except RuntimeError as e:
                print(e)
                print(f"Error encounter in: {fpath}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    data_folder = "../../../data"
    try:
        processed_name = "crystal_processed.pt"
        os.remove(f"{data_folder}/processed/train/{processed_name}")
        print("Old processed data removed")
    except FileNotFoundError:
        print("Processed data not found")
    dn = PeriodicCrystal(root=data_folder)
    print(f"number of graphs: {len(dn)}")
