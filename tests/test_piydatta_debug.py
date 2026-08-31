from __future__ import annotations

import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark import (
    BackwardPassContext,
    count_model_state_elements,
    ModelSpec,
    OptimizerSpec,
    TokenLossBatch,
)
from benchmark.runner import _load_submission_file


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = (
    ROOT / "submissions" / "piydatta_submission" / "submission.py"
)


class PiydattaDebugBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.submission = _load_submission_file(SUBMISSION_PATH)
        self.namespace = self.submission.build_model.__globals__
        self.original_debug_setting = self.namespace["DBUG"]
        # The shipped width costs ~2 GiB per model, and these tests cover wiring.
        self.shipped_width = {
            name: self.namespace[name]
            for name in ("D_MODEL", "NUM_HEADS", "D_FF")
        }
        self.namespace["D_MODEL"] = 32
        self.namespace["NUM_HEADS"] = 4
        self.namespace["D_FF"] = 128
        self.model_class = self.namespace["Model"]
        self.spec = ModelSpec(17, 4, 1_000_000)
        self.input_ids = torch.tensor([[1, 2, 3, 0]])
        self.attention_mask = torch.tensor([[True, True, True, False]])

    def tearDown(self) -> None:
        self.namespace["DBUG"] = self.original_debug_setting
        self.namespace.update(self.shipped_width)

    def test_final_path_only_keeps_training_hypothesis_tensors(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec, use_act=True)
        self.assertFalse(hasattr(model, "collect_act_diagnostics"))
        self.assertFalse(hasattr(model, "collect_model_diagnostics"))
        self.assertFalse(hasattr(model, "collect_training_diagnostics"))
        self.assertFalse(hasattr(model, "_debug_initial_parameter_buffers"))

        model.eval()
        logits, auxiliary = model(self.input_ids, self.attention_mask)

        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertTrue(torch.is_tensor(auxiliary))
        self.assertEqual(auxiliary.ndim, 0)

        fixed_model = self.model_class(self.spec, use_act=False)
        fixed_model.train()
        _, training_auxiliary = fixed_model(
            self.input_ids, self.attention_mask
        )
        self.assertFalse(hasattr(fixed_model, "collect_act_diagnostics"))
        self.assertFalse(hasattr(fixed_model, "collect_model_diagnostics"))
        self.assertFalse(hasattr(fixed_model, "collect_training_diagnostics"))
        self.assertIsInstance(training_auxiliary, dict)
        train_loops = self.namespace["TRAIN_LOOPS"]
        self.assertEqual(
            training_auxiliary["hypothesis_logits"].shape,
            (1, train_loops, 4, 17),
        )
        # One prior row per example now, not one shared vector.
        self.assertEqual(
            training_auxiliary["hypothesis_log_prior"].shape,
            (1, train_loops),
        )

        fixed_model.eval()
        _, fixed_auxiliary = fixed_model(
            self.input_ids, self.attention_mask
        )
        self.assertTrue(torch.is_tensor(fixed_auxiliary))
        self.assertEqual(fixed_auxiliary.ndim, 0)

    def test_debug_path_exposes_diagnostics_only_when_enabled(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=True)
        self.assertIs(model.collect_act_diagnostics, False)
        model.collect_act_diagnostics = True

        _, auxiliary = model(self.input_ids, self.attention_mask)

        self.assertIsInstance(auxiliary, dict)
        diagnostics = auxiliary["act"]
        self.assertIsNotNone(diagnostics)
        self.assertEqual(
            diagnostics["cap_forced_mask"].shape,
            self.attention_mask.shape,
        )
        self.assertEqual(
            diagnostics["tail_forced_mask"].shape,
            self.attention_mask.shape,
        )
        self.assertFalse(diagnostics["update_counts"].requires_grad)

    def test_debug_model_diagnostics_are_complete_and_noninvasive(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=False)
        model.eval()

        baseline_logits, baseline_auxiliary = model(
            self.input_ids,
            self.attention_mask,
        )
        self.assertIsNone(baseline_auxiliary["model_diagnostics"])

        model.collect_model_diagnostics = True
        diagnostic_logits, auxiliary = model(
            self.input_ids,
            self.attention_mask,
        )
        torch.testing.assert_close(diagnostic_logits, baseline_logits)

        diagnostics = auxiliary["model_diagnostics"]
        self.assertIsInstance(diagnostics, dict)
        self.assertEqual(
            diagnostics["segment_token_counts"].shape,
            (self.namespace["NUM_SEGMENTS"],),
        )
        self.assertEqual(len(diagnostics["layers"]), model.max_loops)
        self.assertEqual(len(diagnostics["stage_logits"]), model.max_loops + 1)
        self.assertEqual(
            set(diagnostics["segment_counterfactual_logits"]),
            {"zero", "permuted", "zero_nx", "zero_t", "swap_n_x"},
        )
        for counterfactual_logits in diagnostics[
            "segment_counterfactual_logits"
        ].values():
            self.assertEqual(counterfactual_logits.shape, diagnostic_logits.shape)
            self.assertFalse(counterfactual_logits.requires_grad)
        for layer in diagnostics["layers"]:
            self.assertEqual(
                layer["attention_mass_by_segment"].shape,
                (
                    self.namespace["NUM_SEGMENTS"],
                    self.namespace["NUM_SEGMENTS"],
                ),
            )
            self.assertEqual(
                layer["attention_mass_by_stream"].shape,
                (self.namespace["NUM_SEGMENTS"], 3),
            )
            self.assertFalse(layer["input_rms"].requires_grad)
            self.assertTrue(
                torch.isfinite(layer["attention_mass_by_segment"]).all()
            )

        segment_stats = diagnostics["segment_embedding"]
        self.assertTrue(torch.equal(segment_stats["delta_norms"], torch.zeros(5)))
        persistent_state = model.state_dict()
        self.assertFalse(
            any(name.startswith("_debug_initial_parameter_") for name in persistent_state)
        )

    def test_act_model_diagnostic_stages_end_at_weighted_output(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=True)
        model.eval()
        model.collect_act_diagnostics = True
        model.collect_model_diagnostics = True

        logits, auxiliary = model(self.input_ids, self.attention_mask)

        model_diagnostics = auxiliary["model_diagnostics"]
        self.assertIsNotNone(auxiliary["act"])
        torch.testing.assert_close(
            model_diagnostics["stage_logits"][-1]["logits"],
            logits,
        )
        for layer in model_diagnostics["layers"]:
            populated = layer["segment_query_counts"] > 0
            stream_mass = layer["attention_mass_by_stream"]
            torch.testing.assert_close(
                stream_mass[populated, 2],
                torch.ones_like(stream_mass[populated, 2]),
            )

    def test_model_diagnostics_do_not_change_training_gradients(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=False)
        model.eval()

        baseline_logits, _ = model(self.input_ids, self.attention_mask)
        baseline_logits[self.attention_mask].square().mean().backward()
        baseline_gradient = model.processor.block.qkv.weight.grad.detach().clone()

        model.zero_grad(set_to_none=True)
        model.collect_model_diagnostics = True
        diagnostic_logits, _ = model(self.input_ids, self.attention_mask)
        diagnostic_logits[self.attention_mask].square().mean().backward()

        torch.testing.assert_close(
            model.processor.block.qkv.weight.grad,
            baseline_gradient,
        )

    def test_sampled_training_diagnostics_trace_gradient_credit(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=False)
        model.train()
        model.collect_training_diagnostics = True

        logits, auxiliary = model(self.input_ids, self.attention_mask)
        loss = self.namespace["token_training_loss"](
            TokenLossBatch(
                logits=logits[:, :3],
                labels=torch.tensor([[1, 2, 3]]),
                valid_mask=torch.ones(1, 3, dtype=torch.bool),
                target_positions=torch.tensor([[0, 1, 2]]),
                auxiliary=auxiliary,
            )
        )
        loss.backward()
        diagnostics = model.consume_training_grad_diagnostics()

        self.assertEqual(
            len(diagnostics["stages"]), self.namespace["TRAIN_LOOPS"] + 1
        )
        self.assertEqual(
            diagnostics["segment_signal_grad_rms_by_segment"].shape,
            (self.namespace["NUM_SEGMENTS"],),
        )
        self.assertEqual(
            diagnostics["segment_token_counts"].sum().item(),
            self.attention_mask.sum().item(),
        )
        for stage in diagnostics["stages"]:
            self.assertTrue(torch.isfinite(stage["state_grad_rms"]))
            self.assertTrue(torch.isfinite(stage["relative_to_final"]))
            self.assertIsNotNone(stage["control_grad_rms"])
        self.assertAlmostEqual(
            diagnostics["stages"][-1]["relative_to_final"].item(),
            1.0,
        )
        with self.assertRaisesRegex(RuntimeError, "no sampled training"):
            model.consume_training_grad_diagnostics()

    def test_act_training_diagnostics_trace_executed_stages(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=True)
        model.train()
        model.collect_act_diagnostics = True
        model.collect_training_diagnostics = True

        logits, auxiliary = model(self.input_ids, self.attention_mask)
        logits[self.attention_mask].square().mean().backward()
        diagnostics = model.consume_training_grad_diagnostics()

        self.assertEqual(
            len(diagnostics["stages"]),
            auxiliary["act"]["global_iterations"] + 1,
        )
        self.assertTrue(
            all(stage["control_grad_rms"] is None for stage in diagnostics["stages"])
        )

    def test_synchronized_processor_keeps_prompt_memory_immutable(self) -> None:
        width = self.namespace["D_MODEL"]
        processor_class = self.namespace["SynchronizedProcessor"]
        prompt_len = 2

        class RecordingBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_inputs = []
                self.masks = []

            def forward(self, x, attention_mask):
                self.prompt_inputs.append(
                    x[:, :prompt_len].detach().clone()
                )
                self.masks.append(attention_mask.detach().clone())
                candidate = x.clone()
                candidate[:, :prompt_len] = (
                    candidate[:, :prompt_len] + 100.0
                )
                candidate[:, prompt_len:] = (
                    candidate[:, prompt_len:] + 1.0
                )
                return candidate

        block = RecordingBlock()
        scratchpad_tokens = 2
        processor = processor_class(
            block,
            num_loops=2,
            num_scratchpad_tokens=scratchpad_tokens,
        )
        prompt_memory = torch.randn(1, prompt_len, width)
        work_state = torch.zeros(
            1,
            prompt_len + 1 + scratchpad_tokens,
            width,
        )
        attention_mask = torch.tensor([[True, False]])
        stage_states = []
        hypothesis_states = []

        result = processor(
            prompt_memory,
            work_state,
            attention_mask,
            stage_states=stage_states,
            hypothesis_states=hypothesis_states,
        )

        self.assertEqual(len(block.prompt_inputs), 2)
        torch.testing.assert_close(block.prompt_inputs[0], prompt_memory)
        torch.testing.assert_close(block.prompt_inputs[1], prompt_memory)
        torch.testing.assert_close(result, torch.full_like(result, 2.0))
        self.assertEqual(
            [state.shape for state in stage_states],
            [(1, prompt_len, width)] * 3,
        )
        self.assertEqual(
            [state.shape for state in hypothesis_states],
            [(1, prompt_len, width)] * 2,
        )
        expected_joint_mask = torch.tensor(
            [[True, False, True, True, False, True, True]]
        )
        torch.testing.assert_close(block.masks[0], expected_joint_mask)
        torch.testing.assert_close(block.masks[1], expected_joint_mask)

    def test_synchronized_processor_preserves_prompt_gradient_path(self) -> None:
        width = self.namespace["D_MODEL"]
        processor = self.namespace["SynchronizedProcessor"](
            self.namespace["Block"](),
            num_loops=1,
        )
        prompt_memory = torch.randn(1, 3, width, requires_grad=True)
        work_state = torch.randn(1, 4, width, requires_grad=True)

        result = processor(
            prompt_memory,
            work_state,
            torch.ones(1, 3, dtype=torch.bool),
        )
        result.square().mean().backward()

        self.assertGreater(prompt_memory.grad.abs().sum().item(), 0.0)
        self.assertGreater(work_state.grad.abs().sum().item(), 0.0)

    def test_synchronized_processor_reads_scratch_on_the_next_loop(self) -> None:
        width = self.namespace["D_MODEL"]
        processor_class = self.namespace["SynchronizedProcessor"]
        prompt_len = 1

        class ScratchReadWriteBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.call_count = 0

            def forward(self, x, attention_mask):
                del attention_mask
                candidate = x.clone()
                aligned_index = prompt_len + 1
                scratch_index = aligned_index + prompt_len
                if self.call_count == 0:
                    candidate[:, scratch_index, 0] = x[:, 0, 0]
                else:
                    candidate[:, aligned_index, 0] = x[:, scratch_index, 0]
                self.call_count += 1
                return candidate

        block = ScratchReadWriteBlock()
        processor = processor_class(
            block,
            num_loops=2,
            num_scratchpad_tokens=1,
        )
        prompt_memory = torch.zeros(1, prompt_len, width)
        prompt_memory[:, 0, 0] = 7.0
        work_state = torch.zeros(1, 1 + prompt_len + 1, width)

        result = processor(prompt_memory, work_state)

        self.assertEqual(block.call_count, 2)
        self.assertEqual(result[0, 1, 0].item(), 7.0)
        self.assertEqual(result[0, 2, 0].item(), 7.0)

    def test_train_and_eval_run_different_synchronized_depths(self) -> None:
        self.namespace["DBUG"] = False
        train_loops = self.namespace["TRAIN_LOOPS"]
        eval_loops = self.namespace["EVAL_LOOPS"]
        model = self.model_class(self.spec)

        self.assertFalse(model.use_act)
        self.assertEqual(model.max_loops, eval_loops)
        self.assertIsInstance(
            model.processor,
            self.namespace["SynchronizedProcessor"],
        )

        model.train()
        self.assertEqual(model.active_loops(), train_loops)
        logits, auxiliary = model(self.input_ids, self.attention_mask)
        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertIsInstance(auxiliary, dict)
        self.assertEqual(auxiliary["ponder_cost"].item(), 0.0)
        self.assertEqual(
            auxiliary["hypothesis_logits"].shape,
            (1, train_loops, 4, 17),
        )

        model.eval()
        self.assertEqual(model.active_loops(), eval_loops)
        eval_logits, _ = model(self.input_ids, self.attention_mask)
        self.assertEqual(eval_logits.shape, (1, 4, 17))

    def test_act_and_fixed_modes_share_common_initialization(self) -> None:
        self.namespace["DBUG"] = False
        torch.manual_seed(1234)
        fixed_model = self.model_class(self.spec, use_act=False)
        torch.manual_seed(1234)
        act_model = self.model_class(self.spec, use_act=True)

        torch.testing.assert_close(
            fixed_model.token_embedding.weight,
            act_model.token_embedding.weight,
        )
        torch.testing.assert_close(
            fixed_model.position_embedding.weight,
            act_model.position_embedding.weight,
        )
        torch.testing.assert_close(
            fixed_model.segment_embedding.weight,
            act_model.segment_embedding.weight,
        )
        torch.testing.assert_close(
            fixed_model.final_norm.weight,
            act_model.final_norm.weight,
        )
        for fixed_parameter, act_parameter in zip(
            fixed_model.processor.block.parameters(),
            act_model.processor.block.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(fixed_parameter, act_parameter)

    def test_workspace_starts_from_aligned_prompt_representation(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec)

        class RecordingProcessor(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_memory = None
                self.work_state = None

            def forward(
                self,
                prompt_memory,
                work_state,
                attention_mask,
                *,
                hypothesis_states=None,
                **kwargs,
            ):
                self.prompt_memory = prompt_memory.detach().clone()
                self.work_state = work_state.detach().clone()
                if hypothesis_states is not None:
                    output_end = 1 + prompt_memory.shape[1]
                    hypothesis_states.extend(
                        [work_state[:, 1:output_end]] * model.active_loops()
                    )
                return work_state

        processor = RecordingProcessor()
        model.processor = processor
        model(self.input_ids, self.attention_mask)

        positions = torch.arange(self.input_ids.shape[1])
        segment_ids = torch.tensor([[0, 1, 2, 0]])
        expected_prompt = (
            model.token_embedding(self.input_ids)
            + model.position_embedding(positions)
            + model.segment_embedding(segment_ids)
        )
        expected_workspace = (
            expected_prompt + model.workspace_token.view(1, 1, -1)
        )
        expected_control = model.control_token.view(1, 1, -1).expand(
            self.input_ids.shape[0], -1, -1
        )

        torch.testing.assert_close(processor.prompt_memory, expected_prompt)
        torch.testing.assert_close(
            processor.work_state[:, :1],
            expected_control,
        )
        torch.testing.assert_close(
            processor.work_state[:, 1 : 1 + self.input_ids.shape[1]],
            expected_workspace,
        )
        expected_scratchpad = model.scratchpad_embedding.weight.unsqueeze(0)
        torch.testing.assert_close(
            processor.work_state[:, 1 + self.input_ids.shape[1] :],
            expected_scratchpad,
        )

    def test_segment_embeddings_follow_prompt_fields(self) -> None:
        self.namespace["DBUG"] = False
        spec = ModelSpec(17, 11, 1_000_000)
        model = self.model_class(spec)
        captured_segment_ids = []

        def capture_segment_ids(module, inputs):
            del module
            captured_segment_ids.append(inputs[0].detach().clone())

        handle = model.segment_embedding.register_forward_pre_hook(
            capture_segment_ids
        )
        try:
            prompt_only = torch.tensor(
                [[2, 10, 9, 10, 3, 9, 7, 4, 8, 0, 0]]
            )
            prompt_mask = prompt_only.ne(0)
            model(prompt_only, prompt_mask)
            torch.testing.assert_close(
                captured_segment_ids[-1],
                torch.tensor([[1, 1, 1, 1, 2, 2, 2, 3, 3, 0, 0]]),
            )

            causal_style = torch.tensor(
                [[1, 2, 10, 3, 9, 4, 8, 5, 14, 6, 0]]
            )
            causal_mask = causal_style.ne(0)
            model(causal_style, causal_mask)
            torch.testing.assert_close(
                captured_segment_ids[-1],
                torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 0]]),
            )
        finally:
            handle.remove()

    def test_synchronized_model_ignores_padded_token_ids(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec)
        model.eval()
        changed_padding = self.input_ids.clone()
        changed_padding[0, -1] = 16

        original_logits, _ = model(self.input_ids, self.attention_mask)
        changed_logits, _ = model(changed_padding, self.attention_mask)

        torch.testing.assert_close(
            original_logits[:, :-1],
            changed_logits[:, :-1],
        )

    def test_synchronized_debug_loss_reaches_all_state_components(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec)
        logits, auxiliary = model(self.input_ids, self.attention_mask)

        self.assertIsInstance(auxiliary, dict)
        self.assertIsNone(auxiliary["act"])
        self.assertEqual(auxiliary["ponder_cost"].item(), 0.0)
        valid_logits = logits[self.attention_mask]
        labels = torch.tensor([1, 2, 3])
        loss = self.namespace["training_loss"](
            valid_logits,
            labels,
            auxiliary,
        )
        loss.backward()

        self.assertGreater(model.control_token.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.workspace_token.grad.abs().sum().item(), 0.0)
        self.assertGreater(
            model.scratchpad_embedding.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            model.token_embedding.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            model.segment_embedding.weight.grad.abs().sum().item(),
            0.0,
        )
        self.assertGreater(
            model.processor.block.qkv.weight.grad.abs().sum().item(),
            0.0,
        )

    def test_synchronized_scratchpad_handles_an_all_padding_prompt(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec)
        model.eval()
        input_ids = torch.zeros(1, 4, dtype=torch.long)
        attention_mask = torch.zeros(1, 4, dtype=torch.bool)

        logits, _ = model(input_ids, attention_mask)

        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertTrue(torch.isfinite(logits).all())

    def test_tail_cutoff_is_counted_per_example(self) -> None:
        self.namespace["DBUG"] = True
        width = self.namespace["D_MODEL"]
        processor_class = self.namespace["UniversalProcessor"]

        class IdentityBlock(torch.nn.Module):
            def forward(self, x, attention_mask):
                return x

        class FirstChannelHaltingUnit(torch.nn.Module):
            def forward(self, x):
                return x[..., :1]

        time_embedding = torch.nn.Embedding(2, width)
        torch.nn.init.zeros_(time_embedding.weight)
        processor = processor_class(
            IdentityBlock(),
            time_embedding,
            use_act=True,
            max_loops=2,
            halting_unit=FirstChannelHaltingUnit(),
            tail_halt_fraction=0.89,
        )
        x = torch.zeros(2, 10, width)
        x[0, :9, 0] = 10.0
        x[0, 9, 0] = -10.0
        x[1, :7, 0] = 10.0
        x[1, 7, 0] = -10.0
        attention_mask = torch.tensor(
            [
                [True] * 10,
                [True] * 8 + [False] * 2,
            ]
        )

        _, _, diagnostics = processor(
            x,
            torch.zeros(10, width),
            attention_mask,
            collect_act_diagnostics=True,
        )

        assert diagnostics is not None
        self.assertTrue(diagnostics["tail_forced_mask"][0, 9])
        self.assertFalse(diagnostics["tail_forced_mask"][1].any())
        self.assertTrue(diagnostics["cap_forced_mask"][1, 7])
        self.assertEqual(diagnostics["global_iterations"], 2)

        one_step_embedding = torch.nn.Embedding(1, width)
        torch.nn.init.zeros_(one_step_embedding.weight)
        one_step_processor = processor_class(
            IdentityBlock(),
            one_step_embedding,
            use_act=True,
            max_loops=1,
            halting_unit=FirstChannelHaltingUnit(),
            tail_halt_fraction=0.89,
        )
        _, _, final_step_diagnostics = one_step_processor(
            x[:1],
            torch.zeros(10, width),
            attention_mask[:1],
            collect_act_diagnostics=True,
        )
        assert final_step_diagnostics is not None
        self.assertFalse(final_step_diagnostics["tail_forced_mask"].any())
        self.assertTrue(final_step_diagnostics["cap_forced_mask"][0, 9])

    def test_cosine_decay_scheduler_follows_the_wall_clock_budget(self) -> None:
        first_parameter = torch.nn.Parameter(torch.tensor(1.0))
        second_parameter = torch.nn.Parameter(torch.tensor(1.0))
        first_optimizer = torch.optim.SGD([first_parameter], lr=0.1)
        second_optimizer = torch.optim.AdamW([second_parameter], lr=0.01)
        optimizer = self.namespace["CombinedOptimizer"](
            [first_optimizer, second_optimizer]
        )
        now = [0.0]
        scheduler = self.namespace["CosineDecayScheduler"](
            optimizer,
            total_seconds=10.0,
            hold_fraction=0.5,
            min_factor=0.0,
            started_at=0.0,
            clock=lambda: now[0],
        )

        for elapsed in (0.0, 2.0, 5.0):
            now[0] = elapsed
            scheduler.step()
            self.assertEqual(
                [group["lr"] for group in optimizer.param_groups],
                [0.1, 0.01],
            )

        now[0] = 7.5
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.005)

        now[0] = 10.0
        scheduler.step()
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.0, 0.0],
        )

        # Overshooting the budget stays clamped at the floor.
        now[0] = 99.0
        scheduler.step()
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.0, 0.0],
        )

    def test_scheduler_floor_is_positive_and_keyed_to_the_seed_budget(self) -> None:
        self.namespace["DBUG"] = False
        min_factor = self.namespace["LR_DECAY_MIN_FACTOR"]
        self.assertGreater(min_factor, 0.0)

        model = self.namespace["build_model"](self.spec)
        bundle = self.submission.build_optimizer(
            model,
            OptimizerSpec(training_time_seconds=60.0, device_type="cpu"),
        )
        scheduler = bundle.scheduler

        self.assertEqual(scheduler.total_seconds, 60.0)
        self.assertEqual(
            scheduler.started_at,
            self.namespace["_SEED_STARTED_AT"],
        )
        # The floor is reached at the deadline, not at a fixed step number.
        scheduler.clock = lambda: scheduler.started_at + 60.0
        scheduler.step()
        for group, base_lr in zip(
            scheduler.optimizer.param_groups,
            scheduler.base_lrs,
            strict=True,
        ):
            self.assertAlmostEqual(group["lr"], base_lr * min_factor)
            self.assertGreater(group["lr"], 0.0)

    def test_build_model_materializes_on_the_accelerator(self) -> None:
        self.namespace["DBUG"] = False
        build_model = self.namespace["build_model"]

        expected = self.namespace["_construction_device"]().type
        model = build_model(self.spec)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertEqual(parameter.device.type, expected)

        original = self.namespace["BUILD_ON_ACCELERATOR"]
        self.namespace["BUILD_ON_ACCELERATOR"] = False
        try:
            self.assertEqual(self.namespace["_construction_device"]().type, "cpu")
            cpu_model = build_model(self.spec)
            for name, parameter in cpu_model.named_parameters():
                with self.subTest(parameter=name):
                    self.assertEqual(parameter.device.type, "cpu")
        finally:
            self.namespace["BUILD_ON_ACCELERATOR"] = original

    def test_shipped_width_stays_under_the_model_state_ceiling(self) -> None:
        self.namespace.update(self.shipped_width)
        self.namespace["DBUG"] = False
        width = self.shipped_width["D_MODEL"]
        self.assertEqual(width % self.shipped_width["NUM_HEADS"], 0)
        self.assertEqual(self.shipped_width["D_FF"], 4 * width)

        ceiling = 500_000_000
        # The meta device counts the real shipped model without allocating it.
        for max_seq_len in (13, 64):
            with self.subTest(max_seq_len=max_seq_len), torch.device("meta"):
                model = self.model_class(ModelSpec(17, max_seq_len, ceiling))
            elements = count_model_state_elements(model)
            self.assertLessEqual(elements, ceiling)
            # Width is traded against loop count, so the floor only guards
            # against an accidental collapse to a toy model.
            self.assertGreater(elements, 1_000_000)

    def test_latent_hypothesis_loss_uses_only_valid_target_positions(self) -> None:
        candidate_logits = torch.zeros(1, 2, 4, 3)
        candidate_logits[0, 0, 1, 0] = 4.0
        candidate_logits[0, 0, 3, 1] = 4.0
        candidate_logits[0, 1, 1, 2] = 4.0
        candidate_logits[0, 1, 3, 2] = 4.0
        candidate_logits.requires_grad_()
        hypothesis_prior = torch.tensor(
            [0.2, -0.2],
            requires_grad=True,
        )
        batch = TokenLossBatch(
            logits=torch.zeros(1, 3, 3, requires_grad=True),
            labels=torch.tensor([[0, 1, -100]]),
            valid_mask=torch.tensor([[True, True, False]]),
            target_positions=torch.tensor([[1, 3, -1]]),
            auxiliary={
                "ponder_cost": torch.zeros(()),
                "hypothesis_logits": candidate_logits,
                "hypothesis_log_prior": torch.log_softmax(
                    hypothesis_prior,
                    dim=0,
                ),
            },
        )

        loss = self.namespace["token_training_loss"](batch)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(candidate_logits.grad[:, :, 1].abs().sum(), 0)
        self.assertGreater(candidate_logits.grad[:, :, 3].abs().sum(), 0)
        self.assertEqual(candidate_logits.grad[:, :, 0].abs().sum(), 0)
        self.assertEqual(candidate_logits.grad[:, :, 2].abs().sum(), 0)
        self.assertGreater(hypothesis_prior.grad.abs().sum(), 0)
        self.assertIsNone(batch.logits.grad)

    def test_sam_perturbs_then_restores_before_updating(self) -> None:
        self.namespace["DBUG"] = False
        self.assertGreater(self.namespace["SAM_RHO"], 0.0)
        # Not build_model, which materialises on the accelerator while the
        # fixture inputs are on the CPU.
        model = self.model_class(self.spec, use_act=False)
        bundle = self.submission.build_optimizer(
            model,
            OptimizerSpec(training_time_seconds=60.0, device_type="cpu"),
        )
        self.assertEqual(bundle.backward_passes_per_step, 2)
        self.assertIsNotNone(bundle.between_backward_passes)

        logits, _ = model(self.input_ids, self.attention_mask)
        logits[self.attention_mask].square().mean().backward()
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        with torch.no_grad():
            bundle.between_backward_passes(
                BackwardPassContext(completed_steps=0, pass_index=1, total_passes=2)
            )
        moved = [
            name
            for name, parameter in model.named_parameters()
            if not torch.equal(parameter, before[name])
        ]
        self.assertGreater(len(moved), 0)

        # The perturbation must be exactly rho in global norm.
        squared = sum(
            float((parameter - before[name]).pow(2).sum())
            for name, parameter in model.named_parameters()
        )
        self.assertAlmostEqual(squared**0.5, self.namespace["SAM_RHO"], places=4)

        # Freezing the learning rate isolates restoration from the update.
        for group in bundle.optimizer.param_groups:
            group["lr"] = 0.0
        bundle.optimizer.step()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                torch.testing.assert_close(parameter, before[name])

    def test_sam_can_be_ablated_to_a_single_pass(self) -> None:
        self.namespace["DBUG"] = False
        original = self.namespace["SAM_RHO"]
        self.namespace["SAM_RHO"] = 0.0
        try:
            model = self.namespace["build_model"](self.spec)
            bundle = self.submission.build_optimizer(
                model,
                OptimizerSpec(training_time_seconds=60.0, device_type="cpu"),
            )
        finally:
            self.namespace["SAM_RHO"] = original
        self.assertEqual(bundle.backward_passes_per_step, 1)
        self.assertIsNone(bundle.between_backward_passes)
        self.assertIsNone(bundle.optimizer.restore)

    def test_gates_are_exempt_from_weight_decay(self) -> None:
        self.namespace["DBUG"] = False
        model = self.namespace["build_model"](self.spec)
        bundle = self.submission.build_optimizer(
            model,
            OptimizerSpec(training_time_seconds=60.0, device_type="cpu"),
        )
        gate_names = set(self.namespace["GATE_PARAMETER_NAMES"])
        gates = {
            id(parameter)
            for name, parameter in model.named_parameters()
            if name in gate_names
        }
        self.assertGreater(len(gates), 0)

        seen = set()
        for group in bundle.optimizer.param_groups:
            decay = group.get("weight_decay")
            for parameter in group["params"]:
                if id(parameter) in gates:
                    seen.add(id(parameter))
                    self.assertEqual(decay, self.namespace["GATE_WEIGHT_DECAY"])
                elif decay is not None:
                    self.assertGreater(decay, 0.0)
        self.assertEqual(seen, gates)

    def test_straight_through_commits_to_one_token(self) -> None:
        self.namespace["DBUG"] = False
        self.assertTrue(self.namespace["RETOKENIZE_STRAIGHT_THROUGH"])
        model = self.model_class(self.spec, use_act=False)
        workspace = torch.randn(2, 3, self.namespace["D_MODEL"], requires_grad=True)

        with torch.no_grad():
            # Open the gate fully so the token view is what comes back.
            model.retokenize_gate.fill_(20.0)
        blended = model._retokenize(workspace)

        # Forward is a single embedding row, so it must match one exactly.
        probabilities = F.softmax(
            model.head(model.final_norm(workspace))
            / self.namespace["RETOKENIZE_TEMPERATURE"],
            dim=-1,
        )
        chosen = model.token_embedding.weight[probabilities.argmax(-1)]
        direction = F.normalize(blended.detach(), dim=-1)
        torch.testing.assert_close(
            direction,
            F.normalize(chosen, dim=-1),
            atol=1e-4,
            rtol=1e-4,
        )
        # Backward still flows, which is the point of straight-through.
        blended.sum().backward()
        self.assertIsNotNone(workspace.grad)
        self.assertGreater(float(workspace.grad.abs().sum()), 0.0)

    def test_submission_pins_both_batch_sizes(self) -> None:
        self.assertEqual(
            self.submission.batch_size, self.namespace["TRAIN_BATCH_SIZE"]
        )
        # Unset, eval would inherit the training batch and slow the ladder.
        self.assertEqual(
            self.submission.eval_batch_size, self.namespace["EVAL_BATCH_SIZE"]
        )
        self.assertGreater(
            self.submission.eval_batch_size, self.submission.batch_size
        )

    def test_action_history_is_scaffolding_not_answer(self) -> None:
        self.namespace["DBUG"] = False
        self.assertTrue(self.namespace["USE_ACTION_HISTORY"])
        model = self.model_class(self.spec, use_act=False)
        model.train()

        recorded = []
        original = model.processor.forward

        def spy(prompt_memory, work_state, attention_mask=None, **kwargs):
            hypothesis_states = kwargs.get("hypothesis_states")
            result = original(prompt_memory, work_state, attention_mask, **kwargs)
            if hypothesis_states is not None:
                recorded.extend(state.detach().clone() for state in hypothesis_states)
            return result

        model.processor.forward = spy
        _, auxiliary = model(self.input_ids, self.attention_mask)

        # Exit k's readout must be the block output, with no history term added.
        head, norm = model.head, model.final_norm
        for exit_index, state in enumerate(recorded):
            with self.subTest(exit=exit_index):
                torch.testing.assert_close(
                    head(norm(state)),
                    auxiliary["hypothesis_logits"][:, exit_index],
                )

    def test_action_history_can_be_disabled(self) -> None:
        self.namespace["DBUG"] = False
        torch.manual_seed(7)
        with_history = self.model_class(self.spec, use_act=False)
        with_history.eval()
        baseline, _ = with_history(self.input_ids, self.attention_mask)

        # A closed gate is not the same as no signal, so check the hard switch.
        original = self.namespace["USE_ACTION_HISTORY"]
        self.namespace["USE_ACTION_HISTORY"] = False
        try:
            torch.manual_seed(7)
            without = self.model_class(self.spec, use_act=False)
            without.eval()
            self.assertFalse(hasattr(without, "history_gate"))
            disabled, _ = without(self.input_ids, self.attention_mask)
        finally:
            self.namespace["USE_ACTION_HISTORY"] = original

        self.assertFalse(torch.allclose(baseline, disabled))

    def test_block_cannot_see_the_t_digits(self) -> None:
        self.namespace["DBUG"] = False
        self.assertTrue(self.namespace["HIDE_T_FROM_BLOCK"])
        model = self.model_class(self.spec, use_act=False)
        model.train()

        # Same N and x, different T digit. Every recurrent state must match.
        input_ids = torch.tensor([[2, 10, 4, 8], [2, 10, 4, 9]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        captured = []

        def capture(_module, _args, output):
            captured.append(output.detach().clone())

        handle = model.processor.block.register_forward_hook(capture)
        try:
            _, auxiliary = model(input_ids, mask)
        finally:
            handle.remove()

        # prompt_memory still holds the T tokens, they are just unreachable as
        # attention keys, so the invariant is on what the block produces for
        # the work stream rather than on its raw input.
        prompt_len = input_ids.shape[1]
        self.assertGreater(len(captured), 0)
        for loop, output in enumerate(captured):
            with self.subTest(loop=loop):
                work = output[:, prompt_len:]
                torch.testing.assert_close(work[0], work[1])

        # Every exit therefore reads the same, T changes nothing downstream.
        candidates = auxiliary["hypothesis_logits"]
        torch.testing.assert_close(candidates[0], candidates[1])

        # The exit selector is the one place T is still allowed through.
        prior = auxiliary["hypothesis_log_prior"]
        self.assertFalse(torch.allclose(prior[0], prior[1]))

    def test_hiding_t_leaves_the_answer_slots_writable(self) -> None:
        # target_positions sit on the T segment, so masking the prompt keys
        # must not also blank the workspace the answer is read from.
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec, use_act=False)
        model.eval()
        input_ids = torch.tensor([[2, 10, 4, 8]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        logits, _ = model(input_ids, mask)

        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertTrue(torch.isfinite(logits).all())
        # Positions 2 and 3 are the T segment and carry the answer.
        self.assertGreater(float(logits[0, 2].abs().sum()), 0.0)
        self.assertGreater(float(logits[0, 3].abs().sum()), 0.0)

    def test_eval_selects_an_exit_per_row_from_the_t_segment(self) -> None:
        self.namespace["DBUG"] = True
        model = self.model_class(self.spec, use_act=False)
        model.eval()

        # Two prompts identical except for the digit in the T segment.
        input_ids = torch.tensor([[2, 10, 4, 8], [2, 10, 4, 9]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        logits, auxiliary = model(input_ids, mask)

        prior = auxiliary["hypothesis_log_prior"]
        self.assertEqual(prior.shape, (2, self.namespace["EVAL_LOOPS"]))
        self.assertFalse(torch.allclose(prior[0], prior[1]))

        chosen = prior.argmax(dim=-1)
        for row in range(2):
            torch.testing.assert_close(
                logits[row],
                auxiliary["hypothesis_logits"][row, chosen[row]],
            )

    def test_row_label_routes_gradient_to_the_matching_exit(self) -> None:
        # Exit 0 predicts token 2, exit 1 predicts token 0, and the label is 0.
        candidate_logits = torch.zeros(1, 2, 1, 3)
        candidate_logits[0, 0, 0, 2] = 8.0
        candidate_logits[0, 1, 0, 0] = 8.0
        candidate_logits.requires_grad_()
        hypothesis_prior = torch.zeros(2, requires_grad=True)
        batch = TokenLossBatch(
            logits=torch.zeros(1, 1, 3),
            labels=torch.tensor([[0]]),
            valid_mask=torch.tensor([[True]]),
            target_positions=torch.tensor([[0]]),
            auxiliary={
                "ponder_cost": torch.zeros(()),
                "hypothesis_logits": candidate_logits,
                "hypothesis_log_prior": torch.log_softmax(hypothesis_prior, dim=0),
            },
        )

        loss = self.namespace["token_training_loss"](batch)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        matching = candidate_logits.grad[0, 1].abs().sum()
        mismatched = candidate_logits.grad[0, 0].abs().sum()
        self.assertGreater(matching, mismatched)
        self.assertLess(hypothesis_prior.grad[1].item(), hypothesis_prior.grad[0].item())

    def test_selector_loss_pulls_the_prior_toward_the_explaining_exit(self) -> None:
        self.assertGreater(self.namespace["SELECTOR_LOSS_WEIGHT"], 0.0)
        # Exit 0 predicts token 2, exit 1 predicts token 0, the label is 0.
        candidate_logits = torch.zeros(1, 2, 1, 3)
        candidate_logits[0, 0, 0, 2] = 8.0
        candidate_logits[0, 1, 0, 0] = 8.0
        hypothesis_prior = torch.zeros(2, requires_grad=True)

        def build():
            return TokenLossBatch(
                logits=torch.zeros(1, 1, 3),
                labels=torch.tensor([[0]]),
                valid_mask=torch.tensor([[True]]),
                target_positions=torch.tensor([[0]]),
                auxiliary={
                    "ponder_cost": torch.zeros(()),
                    "hypothesis_logits": candidate_logits,
                    "hypothesis_log_prior": torch.log_softmax(
                        hypothesis_prior, dim=0
                    ),
                },
            )

        self.namespace["token_training_loss"](build()).backward()
        with_selector = hypothesis_prior.grad.clone()

        hypothesis_prior.grad = None
        original_weight = self.namespace["SELECTOR_LOSS_WEIGHT"]
        self.namespace["SELECTOR_LOSS_WEIGHT"] = 0.0
        try:
            self.namespace["token_training_loss"](build()).backward()
        finally:
            self.namespace["SELECTOR_LOSS_WEIGHT"] = original_weight
        without_selector = hypothesis_prior.grad.clone()

        # Both push the prior toward exit 1, the term makes the push stronger.
        self.assertLess(with_selector[1].item(), 0.0)
        self.assertLess(with_selector[1].item(), without_selector[1].item())

    def test_row_loss_is_a_soft_min_bounded_by_the_exit_losses(self) -> None:
        # Equal exits must reduce to plain cross entropy, and unequal exits
        # must land between the best exit and their mean. Isolate the soft-min
        # from the selector term, which adds the prior/posterior cross entropy.
        original_weight = self.namespace["SELECTOR_LOSS_WEIGHT"]
        self.namespace["SELECTOR_LOSS_WEIGHT"] = 0.0
        self.addCleanup(
            self.namespace.__setitem__, "SELECTOR_LOSS_WEIGHT", original_weight
        )

        def row_loss(second_exit_logit: float) -> float:
            candidate_logits = torch.zeros(1, 2, 1, 3)
            candidate_logits[0, 0, 0, 0] = 4.0
            candidate_logits[0, 1, 0, 0] = second_exit_logit
            batch = TokenLossBatch(
                logits=torch.zeros(1, 1, 3),
                labels=torch.tensor([[0]]),
                valid_mask=torch.tensor([[True]]),
                target_positions=torch.tensor([[0]]),
                auxiliary={
                    "ponder_cost": torch.zeros(()),
                    "hypothesis_logits": candidate_logits,
                    "hypothesis_log_prior": torch.log_softmax(
                        torch.zeros(2), dim=0
                    ),
                },
            )
            return float(self.namespace["token_training_loss"](batch))

        identical = row_loss(4.0)
        plain = float(
            torch.nn.functional.cross_entropy(
                torch.tensor([[4.0, 0.0, 0.0]]), torch.tensor([0])
            )
        )
        self.assertAlmostEqual(identical, plain, places=5)

        best = plain
        worst = float(
            torch.nn.functional.cross_entropy(
                torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([0])
            )
        )
        mixed = row_loss(0.0)
        self.assertGreater(mixed, best)
        self.assertLess(mixed, 0.5 * (best + worst))


if __name__ == "__main__":
    unittest.main()
