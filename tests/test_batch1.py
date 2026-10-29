"""Math, symmetry, metric identities, and baseline parity without dataset access."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src")]

import torch
from torch.nn import functional as F
from torch_geometric.data import Data

from src.utils import PlaceHolder
from flow_matching.rate_matrix import RateMatrixDesigner
from experiments.batch1.priors import PriorBank, ConditionalRates, interpolate
from experiments.batch1.statistics import fit_statistics, save_statistics, load_statistics
from experiments.batch1.metrics import evaluate


def graph(labels, pairs, idx=0):
    both = [(i, j) for i, j in pairs for i, j in [(i, j), (j, i)]]
    return Data(x=F.one_hot(torch.tensor(labels), 4).float(),
                edge_index=torch.tensor(both).T.long(), idx=torch.tensor(idx))


def fixture():
    return fit_statistics([graph([0, 0, 2], [(0, 1), (1, 2)], 1),
                           graph([0, 2, 2], [(0, 1), (1, 2), (0, 2)], 2),
                           graph([0, 3], [(0, 1)], 3)], 4, 3)


class PriorsTest(unittest.TestCase):
    def setUp(self):
        self.stats = fixture()
        self.x = torch.tensor([.7, .1, .15, .05])
        self.e = torch.tensor([.8, .1, .05, .03, .02])
        self.bank = PriorBank(self.stats, self.x, self.e)

    def test_train_statistics_and_cache_identity(self):
        self.assertEqual(self.stats["counts"].tolist(), [0, 0, 1, 2])
        self.assertEqual(self.stats["split_ids"].tolist(), [1, 2, 3])
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"stats.pt"
            metadata = save_statistics(self.stats, path, "training-only")
            self.assertEqual(metadata["training_graphs"], 3)
            self.assertEqual(load_statistics(path, "training-only")["schema"], 1)
            with self.assertRaises(ValueError):
                load_statistics(path, "wrong-split")

    def test_empirical_reference_and_padding(self):
        mask = torch.tensor([[1, 1, 0]]*32+[[1, 1, 1]]*32, dtype=torch.bool)
        reference = self.bank.reference(mask)
        self.assertTrue(torch.equal(reference.sum(-1), mask.float()))
        for row in reference[:32, :2].argmax(-1):
            self.assertEqual(sorted(row.tolist()), [0, 3])
        allowed = [[0, 0, 2], [0, 2, 2]]
        for row in reference[32:].argmax(-1):
            self.assertIn(sorted(row.tolist()), allowed)

    def test_probabilities_normalize_and_permute(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        reference = F.one_hot(torch.tensor([[0, 2, 3]]), 4).float()
        permutation = torch.tensor([2, 0, 1])
        for atoms, edges in [("marginal", "size"), ("fixed", "degree"), ("reference", "degree")]:
            px, pe = self.bank.probabilities(mask, atoms, edges, reference)
            qx, qe = self.bank.probabilities(mask[:, permutation], atoms, edges, reference[:, permutation])
            torch.testing.assert_close(px.sum(-1), torch.ones(1, 3))
            torch.testing.assert_close(pe.sum(-1), torch.ones(1, 3, 3))
            torch.testing.assert_close(pe, pe.transpose(1, 2))
            torch.testing.assert_close(qx, px[:, permutation])
            torch.testing.assert_close(qe, pe[:, permutation][:, :, permutation])

    def test_path_endpoints(self):
        prior = self.x.expand(2, 3, 4)
        endpoint = F.one_hot(torch.tensor([[0, 1, 2], [1, 2, 3]]), 4).float()
        torch.testing.assert_close(interpolate(endpoint, prior, torch.zeros(2, 1)), prior)
        torch.testing.assert_close(interpolate(endpoint, prior, torch.ones(2, 1)), endpoint)

    def test_rate_parity_with_original_global_prior(self):
        limit = PlaceHolder(X=self.x.clone(), E=self.e.clone(), y=torch.empty(0))
        original = RateMatrixDesigner("general", "dummy", 0, 0, limit)
        conditional = ConditionalRates("general", "dummy", 0, 0, limit)
        conditional.bind(self.x.expand(2, 3, 4), self.e.expand(2, 3, 3, 5))
        mask = torch.ones(2, 3, dtype=torch.bool)
        state = (F.one_hot(torch.zeros(2, 3, dtype=torch.long), 4).float(),
                 F.one_hot(torch.zeros(2, 3, 3, dtype=torch.long), 5).float())
        predictions = (torch.ones(2, 3, 4)/4, torch.ones(2, 3, 3, 5)/5)
        t = torch.full((2, 1), .4)
        torch.manual_seed(77)
        a = original.compute_graph_rate_matrix(t, mask, state, predictions)
        torch.manual_seed(77)
        b = conditional.compute_graph_rate_matrix(t, mask, state, predictions)
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right)

    def test_conditional_rates_satisfy_kolmogorov_and_zero_support(self):
        for prior in [torch.tensor([.2, .3, .5]), torch.tensor([1., 0., 0.])]:
            limit = PlaceHolder(X=prior.clone(), E=prior.clone(), y=torch.empty(0))
            rate = ConditionalRates("general", "dummy", 0, 0, limit)
            rate.bind(prior.expand(1, 1, 3), prior.expand(1, 1, 1, 3))
            for target in range(3):
                for time in [.2, .7]:
                    endpoint = F.one_hot(torch.tensor(target), 3).float()
                    p = time*endpoint+(1-time)*prior
                    rows = []
                    for current in range(3):
                        variables = rate.compute_dfm_variables(
                            torch.tensor([[time]]), torch.tensor([[[current]]]),
                            torch.tensor([[[[current]]]]), torch.tensor([[target]]),
                            torch.tensor([[[target]]]))
                        row = rate.compute_Rstar(variables)[0][0, 0].clone()
                        self.assertTrue(torch.isfinite(row).all())
                        row[current] = -row.sum()
                        rows.append(row)
                    torch.testing.assert_close(p @ torch.stack(rows), endpoint-prior)

    def test_fixed_node_rates_zero(self):
        mask = torch.ones(1, 3, dtype=torch.bool)
        reference = F.one_hot(torch.tensor([[0, 2, 3]]), 4).float()
        px, pe = self.bank.probabilities(mask, "fixed", "degree", reference)
        rate = ConditionalRates("general", "dummy", 0, 0,
                                PlaceHolder(X=self.x, E=self.e, y=torch.empty(0)))
        rate.bind(px, pe, fixed_atoms=True)
        state = (reference, F.one_hot(torch.zeros(1, 3, 3, dtype=torch.long), 5).float())
        r = rate.compute_graph_rate_matrix(torch.tensor([[.5]]), mask, state,
                                          (torch.ones(1, 3, 4)/4, torch.ones(1, 3, 3, 5)/5))
        self.assertEqual(float(r[0].abs().sum()), 0)
        self.assertTrue(torch.isfinite(r[1]).all())


class MetricsTest(unittest.TestCase):
    def test_known_duplicates_novel_invalid_and_disconnected(self):
        def molecule(n, bonds):
            e = torch.zeros(n, n, dtype=torch.long)
            for i, j in bonds:
                e[i, j] = e[j, i] = 1
            return [torch.zeros(n, dtype=torch.long), e]
        samples = [molecule(1, []), molecule(1, []), molecule(2, [(0, 1)]),
                   molecule(2, []), molecule(6, [(0, i) for i in range(1, 6)])]
        result, diagnostics, groups, unique = evaluate(samples, ["C"], {"C"},
                                                       [s[0].clone() for s in samples])
        self.assertEqual((result["valid_count"], result["unique_count"], result["novel_count"]), (3, 2, 1))
        self.assertAlmostEqual(result["vun"], .2)
        self.assertAlmostEqual(result["validity"]*result["uniqueness"]*result["novelty"], .2)
        self.assertEqual(diagnostics["atom_change_fraction"], 0)
        self.assertEqual(diagnostics["composition_change_fraction"], 0)


class FrozenEndpointTest(unittest.TestCase):
    def test_original_endpoint_replacement_cannot_change_fixed_atoms(self):
        from experiments.batch1.model import Batch1Model
        from graph_discrete_flow_model import GraphDiscreteFlowModel
        model = Batch1Model.__new__(Batch1Model)
        torch.nn.Module.__init__(model)
        model.atoms_mode = "fixed"
        x = F.one_hot(torch.tensor([[0, 2, 0]]), 4).float()
        mask = torch.tensor([[True, True, False]])
        x *= mask[..., None]
        wrong = F.one_hot(torch.tensor([[1, 3, 1]]), 4).float()
        proposal = PlaceHolder(X=wrong, E=torch.empty(0), y=torch.empty(0))
        collapsed = PlaceHolder(X=wrong.argmax(-1).float(), E=torch.empty(0), y=torch.empty(0))
        with patch.object(GraphDiscreteFlowModel, "sample_p_zs_given_zt", return_value=(proposal, collapsed)):
            out, labels = model.sample_p_zs_given_zt(torch.tensor([[.9]]), torch.ones(1, 1),
                                                   x, torch.empty(0), torch.empty(0), mask)
        torch.testing.assert_close(out.X, x)
        self.assertEqual(labels.X.tolist(), [[0, 2, -1]])


class Batch2ProtocolTest(unittest.TestCase):
    def test_fixed_endpoint_and_matched_baseline_settings(self):
        from types import SimpleNamespace
        from experiments.batch1.run import make_config
        args = SimpleNamespace(protocol="batch2", data_dir="/tmp/qm9", seed=2,
                               smoke=False, wandb_mode=None)
        cfg = make_config(args, "M1")
        self.assertEqual(cfg.train.n_epochs, 1000)
        self.assertEqual(cfg.train.seed, 2)
        self.assertFalse(cfg.batch1.select_best_vun)
        self.assertEqual(cfg.batch1.late_curve_samples, 10000)
        self.assertEqual(cfg.batch1.wandb_project, "defog-qm9-batch2")
        args.protocol = "batch1"
        original = make_config(args, "M1")
        self.assertEqual(cfg.model, original.model)
        self.assertEqual(cfg.train, original.train)
        self.assertEqual(cfg.sample, original.sample)


class OfflineTelemetryTest(unittest.TestCase):
    def test_scalar_session_stays_offline_and_records_local_rows(self):
        from omegaconf import OmegaConf
        from experiments.batch1.storage import RunLog
        cfg = OmegaConf.create(dict(
            augmentation=dict(id="B0"), train=dict(seed=0),
            batch1=dict(wandb_mode="offline", wandb_entity=None,
                        wandb_project="defog-batch1-offline-selftest")))
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"WANDB_MODE": "disabled"}):
            log = RunLog(root, cfg, dict(logical_run_id="offline-selftest"))
            self.assertTrue(log.wandb.settings._offline)
            log.write("train", completed_epoch=1, optimizer_step=1, node_ce=.5)
            log.finish(dict(status="passed"))
            sessions = list((Path(root)/"wandb").glob("offline-run-*/*.wandb"))
            self.assertEqual(len(sessions), 1)
            self.assertGreater(sessions[0].stat().st_size, 0)
            row = json.loads((Path(root)/"metrics"/"train.jsonl").read_text())
            self.assertEqual((row["completed_epoch"], row["node_ce"]), (1, .5))


if __name__ == "__main__":
    unittest.main()
