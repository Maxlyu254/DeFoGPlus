"""Small full-model invariants used before allocating time to a long run."""
import torch
from torch_geometric.loader import DataLoader

from graph_discrete_flow_model import GraphDiscreteFlowModel
from src import utils
from experiments.batch1.storage import isolated_rng


@torch.no_grad()
def smoke_invariants(model, data):
    was_training = model.training
    model.eval()
    try:
        batch = next(iter(DataLoader(data.train_dataset, batch_size=2)))
        dense, mask = utils.to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        dense = dense.mask(mask)
        time = torch.full((2, 1), .4)
        with isolated_rng(31):
            actual = model.apply_noise(dense.X, dense.E, batch.y, mask, t=time)
        if model.variant == "B0":
            with isolated_rng(31):
                expected = GraphDiscreteFlowModel.apply_noise(model, dense.X, dense.E, batch.y, mask, t=time)
            for key in ["X_t", "E_t", "y_t", "t", "node_mask"]:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        if model.atoms_mode == "fixed":
            torch.testing.assert_close(actual["X_t"], dense.X, rtol=0, atol=0)
        sizes = torch.tensor([3, 4], device=model.device)
        with isolated_rng(41):
            samples, initial, _ = model.sample_conditioned(sizes)
        if model.variant == "B0":
            with isolated_rng(41):
                original, _ = GraphDiscreteFlowModel.sample_batch(
                    model, batch_id=0, batch_size=2, keep_chain=0,
                    number_chain_steps=2, save_final=0, num_nodes=sizes, save_visualization=False)
            for sample, reference in zip(samples, original):
                for left, right in zip(sample, reference):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        if model.atoms_mode == "fixed":
            for (atoms, _), reference in zip(samples, initial):
                torch.testing.assert_close(atoms, reference, rtol=0, atol=0)
        assert model._sampling_reference is None
    finally:
        model.train(was_training)
