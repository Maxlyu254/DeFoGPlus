"""Hydrogen completeness and canonical identities must agree for novelty."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'src')]
import torch
from rdkit import Chem, RDLogger
from hydra import compose, initialize_config_dir
from experiments.batch1.h_metrics import evaluate_h
from experiments.batch1.metrics import canonical_reference
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qm9_h_profile import training_probe_steps

DECODER=['H','C','N','O','F']


def graph(smiles):
    mol=Chem.AddHs(Chem.MolFromSmiles(smiles))
    atoms=torch.tensor([DECODER.index(a.GetSymbol()) for a in mol.GetAtoms()])
    edges=torch.zeros(len(atoms),len(atoms),dtype=torch.long)
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        edges[i,j]=edges[j,i]=int(bond.GetBondTypeAsDouble())
    return atoms,edges


class HydrogenMetricsTest(unittest.TestCase):
    def setUp(self): RDLogger.DisableLog('rdApp.*')

    def test_explicit_methane_matches_implicit_reference(self):
        sample=graph('C')
        p=torch.tensor([4,2,0,3,1])
        reordered=(sample[0][p],sample[1][p][:,p])
        result=evaluate_h([sample,reordered],DECODER,canonical_reference(['C']))
        self.assertEqual(result['valid_count'],2)
        self.assertEqual(result['unique_count'],1)
        self.assertEqual(result['novel_count'],0)

    def test_missing_h_cannot_be_completed_implicitly(self):
        atoms,edges=graph('C')
        result=evaluate_h([(atoms[:-1],edges[:-1,:-1])],DECODER,set())
        self.assertEqual(result['valid_count'],0)
        self.assertEqual(result['implicit_h_completion_count']+result['radical_count'],1)

    def test_disconnected_and_overvalent_are_invalid(self):
        atoms,edges=graph('C')
        disconnected=edges.clone();disconnected[0,1]=disconnected[1,0]=0
        overvalent=edges.clone();overvalent[0,1]=overvalent[1,0]=2
        result=evaluate_h([(atoms,disconnected),(atoms,overvalent)],DECODER,set())
        self.assertEqual(result['valid_count'],0)
        self.assertEqual(result['disconnected_count'],1)
        self.assertEqual(result['invalid_count'],1)

    def test_novel_identity(self):
        result=evaluate_h([graph('CC')],DECODER,canonical_reference(['C']))
        self.assertEqual(result['validity'],1)
        self.assertEqual(result['novelty'],1)

    def test_config_preserves_backbone_sampler(self):
        with initialize_config_dir(version_base='1.3',config_dir=str(Path(__file__).resolve().parents[1]/'configs')):
            cfg=compose(config_name='h_profile',overrides=['augmentation=M1'])
        self.assertFalse(cfg.dataset.remove_h)
        self.assertEqual(cfg.model.n_layers,9)
        self.assertEqual(cfg.sample.sample_steps,500)
        self.assertEqual((cfg.sample.eta,cfg.sample.omega),(0,0))
        self.assertEqual(cfg.augmentation.id,'M1')


class ManualProbeTest(unittest.TestCase):
    class TinyModel(torch.nn.Linear):
        def __init__(self): super().__init__(2,1,bias=False)
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(),lr=.01,amsgrad=True)
        def training_step(self,batch,index):
            return {'loss':((self(batch)-3)**2).mean()}

    def test_disabled_clipping_still_initializes_two_optimizer_steps(self):
        for value in [None,0]:
            with self.subTest(clip_grad=value):
                model=self.TinyModel()
                before=model.weight.detach().clone()
                with patch('torch.nn.utils.clip_grad_norm_') as clip:
                    optimizer=training_probe_steps(model,torch.ones(2,2),value)
                clip.assert_not_called()
                self.assertFalse(torch.equal(before,model.weight))
                state=optimizer.state[model.weight]
                self.assertEqual(int(state['step']),2)
                self.assertIn('max_exp_avg_sq',state)

    def test_enabled_clipping_is_applied_to_both_steps(self):
        model=self.TinyModel()
        original=torch.nn.utils.clip_grad_norm_
        with patch('torch.nn.utils.clip_grad_norm_',wraps=original) as clip:
            training_probe_steps(model,torch.ones(2,2),.1)
        self.assertEqual(clip.call_count,2)
        self.assertLessEqual(float(model.weight.grad.norm()),.100001)


if __name__=='__main__': unittest.main()
