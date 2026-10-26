"""Six configurations sharing the original transformer, loss, and CTMC sampler."""
import torch
from torch.nn import functional as F

from graph_discrete_flow_model import GraphDiscreteFlowModel
from flow_matching import flow_matching_utils as sampling
from src import utils
from experiments.batch1.priors import ConditionalRates, PriorBank, interpolate
from experiments.batch1.storage import rng_state


class Batch1Model(GraphDiscreteFlowModel):
    def __init__(self, cfg, statistics, **kwargs):
        if cfg.augmentation.atoms == "reference":
            kwargs["dataset_infos"].input_dims["X"] += kwargs["dataset_infos"].output_dims["X"]
        super().__init__(cfg=cfg, **kwargs)
        self.hparams.pop("statistics", None)
        self.variant = cfg.augmentation.id
        self.atoms_mode, self.edge_mode = cfg.augmentation.atoms, cfg.augmentation.edges
        self.prior_bank = PriorBank(statistics, self.limit_dist.X, self.limit_dist.E)
        self._sampling_reference = None
        self.batch1_completed_epochs = 0
        self.batch1_signature = None
        self.batch1_loaded = None
        if self.variant != "B0":
            self.rate_matrix_designer = ConditionalRates(
                cfg.sample.rdb, cfg.sample.rdb_crit, cfg.sample.eta,
                cfg.sample.omega, self.limit_dist)
        if cfg.sample.eta != 0 or cfg.sample.omega != 0:
            raise ValueError("Batch 1 fixes eta=omega=0; sampler searches are outside this study")

    def on_fit_start(self):
        # Logging is owned by the run callback; avoid the original wildcard uploads.
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def on_save_checkpoint(self, checkpoint):
        checkpoint["batch1"] = dict(schema=1, completed_epochs=self.batch1_completed_epochs,
                                    signature=self.batch1_signature, rng=rng_state(),
                                    sampler_epoch=self.batch1_completed_epochs)

    def on_load_checkpoint(self, checkpoint):
        record = checkpoint.get("batch1")
        if record is None or record["signature"] != self.batch1_signature:
            raise ValueError("Checkpoint configuration/data/code signature mismatch")
        self.batch1_loaded = record
        self.batch1_completed_epochs = record["completed_epochs"]

    def on_train_epoch_end(self):
        if self.atoms_mode != "fixed":
            super().on_train_epoch_end()

    def apply_noise(self, X, E, y, node_mask, t=None):
        if self.variant == "B0":
            return super().apply_noise(X, E, y, node_mask, t)
        if t is None:
            t = self.time_distorter.train_ft(X.shape[0], X.device)
        reference = (X if self.atoms_mode == "fixed" else
                     self.prior_bank.reference(node_mask) if self.atoms_mode == "reference" else None)
        px, pe = self.prior_bank.probabilities(node_mask, self.atoms_mode, self.edge_mode, reference)
        probs_x, probs_e = interpolate(X, px, t), interpolate(E, pe, t)
        noisy = sampling.sample_discrete_features(probs_x, probs_e, node_mask)
        xt = F.one_hot(noisy.X, X.shape[-1]).float()
        et = F.one_hot(noisy.E, E.shape[-1]).float()
        dense = utils.PlaceHolder(X=xt, E=et, y=y).mask(node_mask)
        return dict(t=t, X_t=dense.X, E_t=dense.E, y_t=y,
                    node_mask=node_mask, reference=reference)

    def forward(self, noisy_data, extra_data, node_mask):
        if self.atoms_mode != "reference":
            return super().forward(noisy_data, extra_data, node_mask)
        reference = noisy_data.get("reference")
        if reference is None:
            reference = self._sampling_reference
        if reference is None:
            raise ValueError("M0/M1 require permanent reference atom features")
        x = torch.cat((noisy_data["X_t"], extra_data.X, reference), dim=-1).float()
        e = torch.cat((noisy_data["E_t"], extra_data.E), dim=-1).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(x, e, y, node_mask)

    def training_step(self, data, batch_idx):
        if self.atoms_mode != "fixed":
            return super().training_step(data, batch_idx)
        if data.edge_index.numel() == 0:
            return None  # identical no-bond-batch policy to the reference model
        dense, mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense = dense.mask(mask)
        noisy = self.apply_noise(dense.X, dense.E, data.y, mask)
        pred = self.forward(noisy, self.compute_extra_data(noisy), mask)
        valid = dense.E.any(-1)
        # Reuse the original edge CE accumulator and scaling. Passing a
        # zero-width node tensor to the original joint loss has an ambiguous
        # reshape; its node accumulator must instead remain untouched.
        edge_loss = self.train_loss.edge_loss(pred.E[valid], dense.E[valid])
        return {"loss": self.train_loss.lambda_train[0]*edge_loss}

    @torch.no_grad()
    def reconstruct_loss(self, data, time):
        dense, mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense = dense.mask(mask)
        t = torch.full((dense.X.shape[0], 1), time, device=dense.X.device)
        noisy = self.apply_noise(dense.X, dense.E, data.y, mask, t=t)
        prediction = self.forward(noisy, self.compute_extra_data(noisy), mask)
        valid_edges = dense.E.any(-1)
        edge = F.cross_entropy(prediction.E[valid_edges], dense.E.argmax(-1)[valid_edges])
        node = (None if self.atoms_mode == "fixed" else
                F.cross_entropy(prediction.X[mask], dense.X.argmax(-1)[mask]).item())
        return {"edge_ce": edge.item(), "node_ce": node}

    @staticmethod
    def trajectory_summary(x, e, initial, mask, time):
        # Small, aggregate diagnostics only. No full chain tensors are retained.
        labels = x.argmax(-1)
        initial_labels = initial.argmax(-1)
        adjacency = e.argmax(-1)>0
        pairs = mask[:, :, None] & mask[:, None, :]
        pairs &= ~torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)[None]
        current_counts = (x*mask[..., None]).sum(1)
        initial_counts = (initial*mask[..., None]).sum(1)
        return dict(time=float(time), atoms_changed=float(((labels != initial_labels)&mask).sum()/mask.sum()),
                    composition_distance=float((current_counts-initial_counts).abs().sum(-1).div(2*mask.sum(-1)).mean()),
                    edge_density=float((adjacency & pairs).sum()/pairs.sum().clamp_min(1)))

    def sample_p_zs_given_zt(self, t, s, X_t, E_t, y_t, node_mask):
        one_hot, discrete = super().sample_p_zs_given_zt(t, s, X_t, E_t, y_t, node_mask)
        if self.atoms_mode == "fixed":
            # The reference sampler replaces BOTH categorical proposals with
            # network predictions at s=1. Zero node rates alone do not freeze
            # atoms at that endpoint, so enforce the condition after every step.
            one_hot.X = X_t.clone()
            discrete.X = X_t.argmax(-1).masked_fill(~node_mask, -1).type_as(discrete.X)
        return one_hot, discrete

    @torch.no_grad()
    def sample_conditioned(self, sizes, reference=None, trace=False):
        b, n = len(sizes), int(sizes.max())
        mask = torch.arange(n, device=self.device)[None] < sizes[:, None]
        if self.atoms_mode != "marginal" and reference is None:
            reference = self.prior_bank.reference(mask)
        if reference is not None:
            reference = reference[:, :n].to(self.device)
        if self.variant == "B0":
            z = sampling.sample_discrete_feature_noise(self.noise_dist.get_limit_dist(), mask)
        else:
            px, pe = self.prior_bank.probabilities(mask, self.atoms_mode, self.edge_mode, reference)
            self.rate_matrix_designer.bind(px, pe, self.atoms_mode == "fixed")
            draw = sampling.sample_discrete_features(px.clone(), pe.clone(), mask)
            z = utils.PlaceHolder(X=F.one_hot(draw.X, px.shape[-1]).float(),
                                 E=F.one_hot(draw.E, pe.shape[-1]).float(),
                                 y=torch.zeros(b, 0, device=self.device)).mask(mask)
        initial = z.X.clone()
        self._sampling_reference = reference
        traces = []
        trace_n = min(b, self.cfg.batch1.diagnostic_samples)
        targets = list(self.cfg.batch1.trajectory_times) if trace else []
        def record(time):
            traces.append(self.trajectory_summary(z.X[:trace_n], z.E[:trace_n],
                          initial[:trace_n], mask[:trace_n], time))
        if targets and targets[0] == 0:
            record(0)
            targets.pop(0)
        try:
            for step in range(self.cfg.sample.sample_steps):
                t = torch.full((b, 1), step/self.cfg.sample.sample_steps, device=self.device)
                s = torch.full_like(t, (step+1)/self.cfg.sample.sample_steps)
                # Match DeFoG's absorbing-prior boundary treatment. At t=0 a
                # zero-support source can otherwise give an incorrect support count.
                if step == 0 and self.variant != "B0":
                    t = t+1e-6
                t = self.time_distorter.sample_ft(t, self.cfg.sample.time_distortion)
                s = self.time_distorter.sample_ft(s, self.cfg.sample.time_distortion)
                z, discrete = self.sample_p_zs_given_zt(t, s, z.X, z.E, z.y, mask)
                while targets and float(s[0, 0]) >= targets[0]-1e-6:
                    record(float(s[0, 0]))
                    targets.pop(0)
            labels_x, labels_e = z.X.argmax(-1), z.E.argmax(-1)
            samples = [[labels_x[i, :size].cpu(), labels_e[i, :size, :size].cpu()]
                       for i, size in enumerate(sizes.tolist())]
            initials = [initial[i, :size].argmax(-1).cpu() for i, size in enumerate(sizes.tolist())]
            return samples, initials, traces
        finally:
            self._sampling_reference = None
