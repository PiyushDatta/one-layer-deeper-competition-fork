from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F
from torch import nn

from benchmark import TokenLossBatch
from benchmark.manifest import load_manifest
from benchmark.runner import _loss_and_accuracy


ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 17


class TinyTokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2, 3, VOCAB_SIZE))
        self.config = SimpleNamespace(vocab_size=VOCAB_SIZE)
        self.auxiliary = {"source": "tiny-token-model"}

    def forward(self, input_ids, attention_mask=None):
        return self.logits, self.auxiliary


class TokenTrainingLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.token_manifest = load_manifest(
            ROOT / "benchmark/manifests/smoke_cpu.json"
        )

    @staticmethod
    def separate_output_batch():
        return {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.tensor([[3, 5], [7, -100]], dtype=torch.long),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
            "target_positions": torch.tensor([[1, 2], [2, -1]], dtype=torch.long),
        }

    def test_structured_loss_preserves_variable_length_boundaries(self) -> None:
        model = TinyTokenModel()
        received = []

        def sequence_loss(batch: TokenLossBatch) -> torch.Tensor:
            received.append(batch)
            row_losses = [
                F.cross_entropy(
                    batch.logits[row][batch.valid_mask[row]],
                    batch.labels[row][batch.valid_mask[row]],
                )
                for row in range(batch.labels.shape[0])
            ]
            return torch.stack(row_losses).mean()

        loss, accuracy, example_count, loss_weight = _loss_and_accuracy(
            model,
            self.separate_output_batch(),
            self.token_manifest,
            torch.device("cpu"),
            token_training_loss=sequence_loss,
        )

        self.assertEqual(len(received), 1)
        batch = received[0]
        self.assertIsInstance(batch, TokenLossBatch)
        self.assertTrue(
            torch.equal(
                batch.labels,
                torch.tensor([[3, 5], [7, -100]], dtype=torch.long),
            )
        )
        self.assertEqual(batch.valid_mask.sum(dim=1).tolist(), [2, 1])
        self.assertEqual(tuple(batch.logits.shape), (2, 2, VOCAB_SIZE))
        self.assertTrue(
            torch.equal(
                batch.target_positions,
                torch.tensor([[1, 2], [2, -1]], dtype=torch.long),
            )
        )
        self.assertIs(batch.auxiliary, model.auxiliary)
        self.assertEqual(accuracy, 0.0)
        self.assertEqual(example_count, 2)
        self.assertEqual(loss_weight, 3)

        loss.backward()
        self.assertGreater(model.logits.grad[0, 1].abs().sum().item(), 0.0)
        self.assertGreater(model.logits.grad[0, 2].abs().sum().item(), 0.0)
        self.assertGreater(model.logits.grad[1, 2].abs().sum().item(), 0.0)
        self.assertEqual(model.logits.grad[1, 0].abs().sum().item(), 0.0)

    def test_legacy_loss_remains_flattened_and_matches_default(self) -> None:
        legacy_model = TinyTokenModel()
        received = {}

        def legacy_loss(logits, labels, auxiliary):
            received["logits"] = logits
            received["labels"] = labels
            received["auxiliary"] = auxiliary
            return F.cross_entropy(logits, labels)

        legacy, _, _, _ = _loss_and_accuracy(
            legacy_model,
            self.separate_output_batch(),
            self.token_manifest,
            torch.device("cpu"),
            training_loss=legacy_loss,
        )
        default, _, _, _ = _loss_and_accuracy(
            TinyTokenModel(),
            self.separate_output_batch(),
            self.token_manifest,
            torch.device("cpu"),
        )

        self.assertEqual(tuple(received["logits"].shape), (3, VOCAB_SIZE))
        self.assertEqual(received["labels"].tolist(), [3, 5, 7])
        self.assertIs(received["auxiliary"], legacy_model.auxiliary)
        self.assertEqual(legacy.item(), default.item())

    def test_structured_loss_preserves_causal_rows(self) -> None:
        model = TinyTokenModel()
        received = []
        batch = {
            "input_ids": torch.ones((2, 3), dtype=torch.long),
            "labels": torch.tensor(
                [[-100, 3, 5], [-100, 7, -100]], dtype=torch.long
            ),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        }

        def sequence_loss(loss_batch):
            received.append(loss_batch)
            return F.cross_entropy(
                loss_batch.logits[loss_batch.valid_mask],
                loss_batch.labels[loss_batch.valid_mask],
            )

        _loss_and_accuracy(
            model,
            batch,
            self.token_manifest,
            torch.device("cpu"),
            token_training_loss=sequence_loss,
        )

        self.assertEqual(tuple(received[0].logits.shape), (2, 2, VOCAB_SIZE))
        self.assertEqual(received[0].labels.tolist(), [[3, 5], [7, -100]])
        self.assertEqual(received[0].valid_mask.sum(dim=1).tolist(), [2, 1])
        self.assertIsNone(received[0].target_positions)

    def test_structured_loss_return_contract_is_enforced(self) -> None:
        with self.assertRaisesRegex(TypeError, "one scalar tensor"):
            _loss_and_accuracy(
                TinyTokenModel(),
                self.separate_output_batch(),
                self.token_manifest,
                torch.device("cpu"),
                token_training_loss=lambda batch: batch.logits,
            )

        with self.assertRaisesRegex(ValueError, "must be differentiable"):
            _loss_and_accuracy(
                TinyTokenModel(),
                self.separate_output_batch(),
                self.token_manifest,
                torch.device("cpu"),
                token_training_loss=lambda batch: torch.tensor(1.0),
            )


if __name__ == "__main__":
    unittest.main()
