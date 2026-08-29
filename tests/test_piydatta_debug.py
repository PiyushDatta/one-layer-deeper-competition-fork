from __future__ import annotations

import unittest
from pathlib import Path

import torch

from benchmark import ModelSpec, TokenLossBatch
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
        self.model_class = self.namespace["Model"]
        self.spec = ModelSpec(17, 4, 1_000_000)
        self.input_ids = torch.tensor([[1, 2, 3, 0]])
        self.attention_mask = torch.tensor([[True, True, True, False]])

    def tearDown(self) -> None:
        self.namespace["DBUG"] = self.original_debug_setting

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
        self.assertEqual(
            training_auxiliary["hypothesis_logits"].shape,
            (1, fixed_model.max_loops, 4, 17),
        )
        self.assertEqual(
            training_auxiliary["hypothesis_log_prior"].shape,
            (fixed_model.max_loops,),
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

        self.assertEqual(len(diagnostics["stages"]), model.max_loops + 1)
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

    def test_default_model_uses_two_synchronized_loops(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec)

        self.assertFalse(model.use_act)
        self.assertEqual(model.max_loops, 2)
        self.assertIsInstance(
            model.processor,
            self.namespace["SynchronizedProcessor"],
        )
        logits, auxiliary = model(self.input_ids, self.attention_mask)
        self.assertEqual(logits.shape, (1, 4, 17))
        self.assertIsInstance(auxiliary, dict)
        self.assertEqual(auxiliary["ponder_cost"].item(), 0.0)
        self.assertEqual(
            auxiliary["hypothesis_logits"].shape,
            (1, model.max_loops, 4, 17),
        )

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
                        [work_state[:, 1:output_end]] * model.max_loops
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

    def test_cosine_decay_scheduler_updates_every_optimizer_group(self) -> None:
        first_parameter = torch.nn.Parameter(torch.tensor(1.0))
        second_parameter = torch.nn.Parameter(torch.tensor(1.0))
        first_optimizer = torch.optim.SGD([first_parameter], lr=0.1)
        second_optimizer = torch.optim.AdamW([second_parameter], lr=0.01)
        optimizer = self.namespace["CombinedOptimizer"](
            [first_optimizer, second_optimizer]
        )
        scheduler = self.namespace["CosineDecayScheduler"](
            optimizer,
            start_step=2,
            end_step=4,
            min_factor=0.0,
        )

        scheduler.step()
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.1, 0.01],
        )
        scheduler.step()
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.1, 0.01],
        )
        scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.005)
        scheduler.step()
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [0.0, 0.0],
        )

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

    def test_eval_uses_the_globally_selected_hypothesis(self) -> None:
        self.namespace["DBUG"] = False
        model = self.model_class(self.spec, use_act=False)
        with torch.no_grad():
            model.hypothesis_prior.copy_(torch.tensor([10.0, -10.0]))

        model.train()
        _, training_auxiliary = model(
            self.input_ids,
            self.attention_mask,
        )
        expected_logits = training_auxiliary["hypothesis_logits"][:, 0]

        model.eval()
        evaluation_logits, evaluation_auxiliary = model(
            self.input_ids,
            self.attention_mask,
        )

        torch.testing.assert_close(evaluation_logits, expected_logits)
        self.assertTrue(torch.is_tensor(evaluation_auxiliary))
        self.assertEqual(evaluation_auxiliary.ndim, 0)

    def test_hypothesis_depth_penalty_prefers_the_earlier_exit(self) -> None:
        candidate_logits = torch.zeros(
            1,
            2,
            2,
            3,
            requires_grad=True,
        )
        hypothesis_prior = torch.zeros(2, requires_grad=True)
        batch = TokenLossBatch(
            logits=torch.zeros(1, 1, 3),
            labels=torch.tensor([[0]]),
            valid_mask=torch.tensor([[True]]),
            target_positions=torch.tensor([[1]]),
            auxiliary={
                "ponder_cost": torch.zeros(()),
                "hypothesis_logits": candidate_logits,
                "hypothesis_log_prior": torch.log_softmax(
                    hypothesis_prior,
                    dim=0,
                ),
            },
        )

        self.namespace["token_training_loss"](batch).backward()

        self.assertLess(hypothesis_prior.grad[0].item(), 0.0)
        self.assertGreater(hypothesis_prior.grad[1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
