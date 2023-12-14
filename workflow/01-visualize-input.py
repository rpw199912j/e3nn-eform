import os
from eform.dataset import PeriodicCrystal


xstal_net = PeriodicCrystal(
    root="../data/"
)


fig = xstal_net.plotly_vis(xstal_net[0], center_atom_idx=1205)
fig.write_html("../figures/01-vis-input.html")