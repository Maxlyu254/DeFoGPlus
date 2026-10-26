"""Conditional product priors and the same DeFoG rate construction per pair."""
import torch
from torch.nn import functional as F

from flow_matching.rate_matrix import RateMatrixDesigner


class PriorBank(torch.nn.Module):
    def __init__(self, statistics, base_nodes, base_edges):
        super().__init__()
        for key in ["counts", "offsets", "references", "q", "mu"]:
            self.register_buffer(key, statistics[key].clone(), persistent=False)
        self.register_buffer("base_nodes", base_nodes.clone(), persistent=False)
        self.register_buffer("base_edges", base_edges.clone(), persistent=False)
        self.has_missing_sizes = bool((statistics["counts"][1:] == 0).any())

    def reference(self, mask):
        sizes = mask.sum(-1)
        count = self.counts[sizes]
        index = self.offsets[sizes] + (torch.rand(len(sizes), device=mask.device)*count).long()
        index = index.clamp_max(self.references.shape[0]-1)
        labels = self.references[index, :mask.shape[1]].long()
        # Uniformly assign a sampled composition to vertex positions.
        order = torch.rand(mask.shape, device=mask.device).masked_fill(~mask, 2).argsort(-1)
        labels = labels.gather(-1, order)
        if self.has_missing_sizes:
            fallback = self.base_nodes.expand(*mask.shape, -1).reshape(-1, len(self.base_nodes))
            fallback = fallback.multinomial(1).reshape(mask.shape)
            labels = torch.where((count == 0)[:, None], fallback, labels)
        return F.one_hot(labels, num_classes=len(self.base_nodes)).float()*mask[..., None]

    def probabilities(self, mask, atoms_mode, edge_mode, reference=None):
        b, n = mask.shape
        if atoms_mode == "marginal":
            x = self.base_nodes.expand(b, n, -1)
        else:
            if reference is None:
                raise ValueError("Fixed/reference atom prior requires reference atom identities")
            x = reference.clone()
            x = torch.where(mask[..., None], x, self.base_nodes)
        if edge_mode == "marginal":
            e = self.base_edges.expand(b, n, n, -1)
        else:
            if edge_mode == "size":
                q = self.q[mask.sum(-1)][:, None, None].expand(b, n, n)
            elif edge_mode == "degree":
                labels = reference.argmax(-1)
                weights = self.mu[mask.sum(-1)[:, None], labels]*mask
                q = (weights[:, :, None]*weights[:, None, :] /
                     weights.sum(-1)[:, None, None].clamp_min(1e-12)).clamp(0, 1)
            else:
                raise ValueError(edge_mode)
            bond_types = self.base_edges[1:] / self.base_edges[1:].sum()
            e = torch.cat(((1-q)[..., None], q[..., None]*bond_types), dim=-1)
        return x, e


def interpolate(endpoint, prior, t):
    scale = t.reshape(t.shape[0], *([1]*(endpoint.ndim-1)))
    return scale*endpoint + (1-scale)*prior


class ConditionalRates(RateMatrixDesigner):
    def bind(self, nodes, edges, fixed_atoms=False):
        self.nodes, self.edges, self.fixed_atoms = nodes, edges, fixed_atoms

    def compute_dfm_variables(self, t, xt, et, x1, e1):
        endpoint_x = F.one_hot(x1, self.num_classes_X).float()
        endpoint_e = F.one_hot(e1, self.num_classes_E).float()
        px = interpolate(endpoint_x, self.nodes, t)
        pe = interpolate(endpoint_e, self.edges, t)
        dx, de = endpoint_x-self.nodes, endpoint_e-self.edges
        return dict(pt_vals_X=px, pt_vals_E=pe,
                    pt_vals_at_Xt=px.gather(-1, xt).squeeze(-1),
                    pt_vals_at_Et=pe.gather(-1, et).squeeze(-1),
                    dt_p_vals_X=dx, dt_p_vals_E=de,
                    dt_p_vals_at_Xt=dx.gather(-1, xt).squeeze(-1),
                    dt_p_vals_at_Et=de.gather(-1, et).squeeze(-1),
                    Z_t_X=(px>0).sum(-1), Z_t_E=(pe>0).sum(-1))

    def compute_Rstar(self, values):
        output = []
        for name in ["X", "E"]:
            numerator = F.relu(values[f"dt_p_vals_{name}"]-
                               values[f"dt_p_vals_at_{name}t"][..., None])
            denominator = values[f"Z_t_{name}"]*values[f"pt_vals_at_{name}t"]
            rates = numerator/denominator.clamp_min(1e-30)[..., None]
            rates = torch.where(denominator[..., None]>0, rates, 0)
            rates = torch.where(values[f"pt_vals_{name}"] > 0, rates, 0)
            output.append(rates)
        if self.fixed_atoms:
            output[0] = torch.zeros_like(output[0])
        return tuple(output)
