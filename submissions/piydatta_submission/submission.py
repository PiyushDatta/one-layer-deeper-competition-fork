from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from benchmark import (
    assert_model_state,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
)
from torch import nn, Tensor

D_MODEL = 288
NUM_HEADS = 9
D_FF = 4 * D_MODEL
PONDER_WEIGHT = 0.005
USE_ACT = False
FIXED_LOOPS = 2
ACT_MAX_LOOPS = 16
ACT_TAIL_HALT_FRACTION = None
USE_MUON = True
MUON_LR = 1e-3
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.1
MUON_ADJUST_LR_FN = "match_rms_adamw"
LR_DECAY_START_STEP = 200
LR_DECAY_END_STEP = 400
LR_DECAY_MIN_FACTOR = 0.0
USE_LATENT_HYPOTHESES = True
HYPOTHESIS_TEMPERATURE = 1.0
HYPOTHESIS_ALL_LOSS_WEIGHT = 0.25
HYPOTHESIS_DEPTH_PENALTY = 0.01
NUM_SCRATCHPAD_TOKENS = 4
DBUG = False

PAD_TOKEN_ID = 0
N_TOKEN_ID = 2
X_TOKEN_ID = 3
T_TOKEN_ID = 4
ANS_TOKEN_ID = 5
NUM_SEGMENTS = 5

def training_loss(
    logits: Tensor,
    labels: Tensor,
    auxiliary: object,
) -> Tensor:
    task_loss = F.cross_entropy(logits, labels)
    ponder_cost = (
        auxiliary["ponder_cost"]
        if isinstance(auxiliary, dict)
        else auxiliary
    )
    return task_loss + PONDER_WEIGHT * ponder_cost


def _candidate_target_logits(
    candidate_logits: Tensor,
    batch: TokenLossBatch,
) -> Tensor:
    batch_size, candidate_count, _, vocab_size = candidate_logits.shape
    if batch.target_positions is None:
        candidate_target_logits = candidate_logits[:, :, :-1, :]
        if candidate_target_logits.shape[2] != batch.labels.shape[1]:
            raise ValueError("causal candidate logits do not match target length")
        return candidate_target_logits

    positions = batch.target_positions.clamp_min(0)
    gather_positions = positions[:, None, :, None].expand(
        batch_size,
        candidate_count,
        positions.shape[1],
        vocab_size,
    )
    return candidate_logits.gather(2, gather_positions)


def token_training_loss(batch: TokenLossBatch) -> Tensor:
    if not isinstance(batch.auxiliary, dict):
        raise TypeError("latent hypothesis training requires auxiliary tensors")

    ponder_cost = batch.auxiliary["ponder_cost"]
    candidate_logits = batch.auxiliary.get("hypothesis_logits")
    hypothesis_log_prior = batch.auxiliary.get("hypothesis_log_prior")
    if candidate_logits is None or hypothesis_log_prior is None:
        task_loss = F.cross_entropy(
            batch.logits[batch.valid_mask],
            batch.labels[batch.valid_mask],
        )
        return task_loss + PONDER_WEIGHT * ponder_cost

    candidate_target_logits = _candidate_target_logits(
        candidate_logits,
        batch,
    )
    batch_size, candidate_count, target_length, vocab_size = (
        candidate_target_logits.shape
    )
    candidate_labels = batch.labels[:, None, :].expand(
        batch_size,
        candidate_count,
        target_length,
    )
    token_losses = F.cross_entropy(
        candidate_target_logits.reshape(-1, vocab_size),
        candidate_labels.reshape(-1),
        reduction="none",
    ).view(batch_size, candidate_count, target_length)

    candidate_valid_mask = batch.valid_mask[:, None, :]
    target_counts = candidate_valid_mask.sum(dim=-1).clamp_min(1)
    sequence_losses = (
        (token_losses * candidate_valid_mask).sum(dim=-1)
        / target_counts
    )
    rows_with_targets = batch.valid_mask.any(dim=-1)
    candidate_evidence = sequence_losses[rows_with_targets].mean(dim=0)
    depth_cost = HYPOTHESIS_DEPTH_PENALTY * torch.arange(
        candidate_count,
        device=candidate_evidence.device,
        dtype=candidate_evidence.dtype,
    )
    selection_evidence = candidate_evidence + depth_cost

    temperature = HYPOTHESIS_TEMPERATURE
    survivor_loss = -temperature * torch.logsumexp(
        hypothesis_log_prior - selection_evidence / temperature,
        dim=0,
    )
    all_candidate_loss = candidate_evidence.mean()
    task_loss = (
        (1.0 - HYPOTHESIS_ALL_LOSS_WEIGHT) * survivor_loss
        + HYPOTHESIS_ALL_LOSS_WEIGHT * all_candidate_loss
    )
    return task_loss + PONDER_WEIGHT * ponder_cost


class Config:
    def __init__(self, vocab_size: int, max_seq_len: int) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out = nn.Linear(D_MODEL, D_MODEL)
        self.mixer_norm = RMSNorm(D_MODEL)
        self.up = nn.Linear(D_MODEL, D_FF)
        self.down = nn.Linear(D_FF, D_MODEL)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor | None,
        *,
        layer_diagnostics: list[dict[str, object]] | None = None,
        segment_ids: Tensor | None = None,
        query_mask: Tensor | None = None,
        key_mask: Tensor | None = None,
        key_stream_ids: Tensor | None = None,
    ) -> Tensor:
        residual = x
        x = self.attention_norm(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        k = k.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        v = v.view(batch, length, NUM_HEADS, -1).transpose(1, 2)
        mask = None
        if attention_mask is not None:
            if attention_mask.shape == (batch, length):
                mask = attention_mask[:, None, None, :]
            elif attention_mask.shape == (batch, length, length):
                mask = attention_mask[:, None, :, :]
            else:
                raise ValueError("invalid attention_mask shape")
            mask = mask.to(device=x.device, dtype=torch.bool)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).contiguous().view(batch, length, D_MODEL)
        attention_update = self.out(x)
        post_attention = residual + attention_update
        mlp_update = self.down(
            F.gelu(self.up(self.mixer_norm(post_attention)))
        )
        output = post_attention + mlp_update

        if DBUG and layer_diagnostics is not None:
            self._append_diagnostics(
                layer_diagnostics,
                input_state=residual,
                attention_update=attention_update,
                post_attention=post_attention,
                mlp_update=mlp_update,
                output_state=output,
                q=q,
                k=k,
                attention_mask=mask,
                segment_ids=segment_ids,
                query_mask=query_mask,
                key_mask=key_mask,
                key_stream_ids=key_stream_ids,
            )
        return output

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        mask_float = mask.to(dtype=values.dtype)
        return (values * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    @classmethod
    def _masked_rms(cls, values: Tensor, mask: Tensor) -> Tensor:
        token_mean_square = values.float().square().mean(dim=-1)
        return cls._masked_mean(token_mean_square, mask).clamp_min(0.0).sqrt()

    @classmethod
    def _rms_by_segment(
        cls,
        values: Tensor,
        segment_ids: Tensor,
        mask: Tensor,
    ) -> Tensor:
        return torch.stack(
            [
                cls._masked_rms(
                    values,
                    mask & segment_ids.eq(segment),
                )
                for segment in range(NUM_SEGMENTS)
            ]
        )

    @classmethod
    def _append_diagnostics(
        cls,
        diagnostics: list[dict[str, object]],
        *,
        input_state: Tensor,
        attention_update: Tensor,
        post_attention: Tensor,
        mlp_update: Tensor,
        output_state: Tensor,
        q: Tensor,
        k: Tensor,
        attention_mask: Tensor | None,
        segment_ids: Tensor | None,
        query_mask: Tensor | None,
        key_mask: Tensor | None,
        key_stream_ids: Tensor | None,
    ) -> None:
        batch_size, sequence_length, _ = input_state.shape
        device = input_state.device
        if segment_ids is None:
            segment_ids = torch.zeros(
                batch_size,
                sequence_length,
                dtype=torch.long,
                device=device,
            )
        if segment_ids.shape != (batch_size, sequence_length):
            raise ValueError("diagnostic segment_ids must match the block sequence")
        if query_mask is None:
            query_mask = torch.ones(
                batch_size,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        else:
            query_mask = query_mask.bool()
        if key_mask is None:
            key_mask = torch.ones_like(query_mask)
        else:
            key_mask = key_mask.bool()
        if query_mask.shape != (batch_size, sequence_length):
            raise ValueError("diagnostic query_mask must match the block sequence")
        if key_mask.shape != (batch_size, sequence_length):
            raise ValueError("diagnostic key_mask must match the block sequence")
        if key_stream_ids is None:
            key_stream_ids = torch.full(
                (batch_size, sequence_length),
                2,
                dtype=torch.long,
                device=device,
            )
        if key_stream_ids.shape != (batch_size, sequence_length):
            raise ValueError(
                "diagnostic key_stream_ids must match the block sequence"
            )

        input_value = input_state.detach().float()
        attention_value = attention_update.detach().float()
        post_attention_value = post_attention.detach().float()
        mlp_value = mlp_update.detach().float()
        output_value = output_state.detach().float()
        segment_ids = segment_ids.detach()
        query_mask = query_mask.detach()
        key_mask = key_mask.detach()
        key_stream_ids = key_stream_ids.detach()

        input_rms = cls._masked_rms(input_value, query_mask)
        attention_rms = cls._masked_rms(attention_value, query_mask)
        post_attention_rms = cls._masked_rms(
            post_attention_value,
            query_mask,
        )
        mlp_rms = cls._masked_rms(mlp_value, query_mask)
        output_rms = cls._masked_rms(output_value, query_mask)
        input_output_cosines = F.cosine_similarity(
            input_value,
            output_value,
            dim=-1,
        )
        input_output_cosine = cls._masked_mean(
            input_output_cosines,
            query_mask,
        )

        q_value = q.detach().float()
        k_value = k.detach().float()
        attention_scores = torch.matmul(
            q_value,
            k_value.transpose(-2, -1),
        ) / math.sqrt(q_value.shape[-1])
        if attention_mask is not None:
            diagnostic_mask = attention_mask.detach().bool()
            attention_scores = attention_scores.masked_fill(
                ~diagnostic_mask,
                torch.finfo(attention_scores.dtype).min,
            )
        else:
            diagnostic_mask = None
        attention_probabilities = attention_scores.softmax(dim=-1)
        if diagnostic_mask is not None:
            attention_probabilities = attention_probabilities.masked_fill(
                ~diagnostic_mask,
                0.0,
            )
        attention_probabilities = torch.nan_to_num(attention_probabilities)
        entropy_per_head = -(
            attention_probabilities
            * attention_probabilities.clamp_min(1e-12).log()
        ).sum(dim=-1)
        attention_entropy = cls._masked_mean(
            entropy_per_head.mean(dim=1),
            query_mask,
        )
        effective_attended_tokens = cls._masked_mean(
            entropy_per_head.exp().mean(dim=1),
            query_mask,
        )

        mean_attention = attention_probabilities.mean(dim=1)
        attention_mass_rows = []
        attention_stream_rows = []
        segment_query_counts = []
        for query_segment in range(NUM_SEGMENTS):
            selected_queries = query_mask & segment_ids.eq(query_segment)
            query_count = selected_queries.sum()
            segment_query_counts.append(query_count)
            key_masses = []
            for key_segment in range(NUM_SEGMENTS):
                selected_keys = key_mask & segment_ids.eq(key_segment)
                mass_per_query = (
                    mean_attention
                    * selected_keys[:, None, :].to(mean_attention.dtype)
                ).sum(dim=-1)
                key_masses.append(
                    cls._masked_mean(mass_per_query, selected_queries)
                )
            attention_mass_rows.append(torch.stack(key_masses))
            stream_masses = []
            for key_stream in range(3):
                selected_keys = key_mask & key_stream_ids.eq(key_stream)
                mass_per_query = (
                    mean_attention
                    * selected_keys[:, None, :].to(mean_attention.dtype)
                ).sum(dim=-1)
                stream_masses.append(
                    cls._masked_mean(mass_per_query, selected_queries)
                )
            attention_stream_rows.append(torch.stack(stream_masses))

        state_change = output_value - input_value
        diagnostics.append(
            {
                "step": len(diagnostics) + 1,
                "valid_query_count": query_mask.sum().detach(),
                "input_rms": input_rms.detach(),
                "attention_update_rms": attention_rms.detach(),
                "attention_update_ratio": (
                    attention_rms / input_rms.clamp_min(1e-12)
                ).detach(),
                "mlp_update_rms": mlp_rms.detach(),
                "mlp_update_ratio": (
                    mlp_rms / post_attention_rms.clamp_min(1e-12)
                ).detach(),
                "output_rms": output_rms.detach(),
                "input_output_cosine": input_output_cosine.detach(),
                "attention_entropy": attention_entropy.detach(),
                "effective_attended_tokens": effective_attended_tokens.detach(),
                "segment_query_counts": torch.stack(
                    segment_query_counts
                ).detach(),
                "attention_mass_by_segment": torch.stack(
                    attention_mass_rows
                ).detach(),
                "attention_mass_by_stream": torch.stack(
                    attention_stream_rows
                ).detach(),
                "state_change_rms_by_segment": cls._rms_by_segment(
                    state_change,
                    segment_ids,
                    query_mask,
                ).detach(),
            }
        )


class SynchronizedProcessor(nn.Module):
    def __init__(
        self,
        block: Block,
        *,
        num_loops: int,
        num_scratchpad_tokens: int = 0,
    ) -> None:
        super().__init__()
        if num_loops < 1:
            raise ValueError("num_loops must be positive")
        if num_scratchpad_tokens < 0:
            raise ValueError("num_scratchpad_tokens must be non-negative")
        self.block = block
        self.num_loops = num_loops
        self.num_scratchpad_tokens = num_scratchpad_tokens

    def forward(
        self,
        prompt_memory: Tensor,
        work_state: Tensor,
        attention_mask: Tensor | None = None,
        *,
        segment_ids: Tensor | None = None,
        layer_diagnostics: list[dict[str, object]] | None = None,
        stage_states: list[Tensor] | None = None,
        training_stage_states: list[Tensor] | None = None,
        hypothesis_states: list[Tensor] | None = None,
    ) -> Tensor:
        batch_size, prompt_len, _ = prompt_memory.shape
        output_end = 1 + prompt_len
        expected_work_len = output_end + self.num_scratchpad_tokens
        if work_state.shape[:2] != (batch_size, expected_work_len):
            raise ValueError(
                "work_state must contain one control token, one aligned work "
                "token per prompt position, and the configured scratchpad tokens"
            )

        prompt_mask = torch.ones(
            batch_size,
            prompt_len,
            dtype=torch.bool,
            device=prompt_memory.device,
        )
        joint_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, prompt_len):
                raise ValueError("synchronized processor requires a padding mask")
            prompt_mask = attention_mask.bool()
        control_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=prompt_memory.device,
        )
        scratchpad_mask = torch.ones(
            batch_size,
            self.num_scratchpad_tokens,
            dtype=torch.bool,
            device=prompt_memory.device,
        )
        work_mask = torch.cat(
            (control_mask, prompt_mask, scratchpad_mask),
            dim=1,
        )
        if attention_mask is not None:
            joint_mask = torch.cat((prompt_mask, work_mask), dim=1)

        joint_segment_ids = None
        joint_stream_ids = None
        if layer_diagnostics is not None:
            if segment_ids is None or segment_ids.shape != (
                batch_size,
                prompt_len,
            ):
                raise ValueError(
                    "segment_ids are required for synchronized diagnostics"
                )
            control_segments = torch.zeros(
                batch_size,
                1,
                dtype=segment_ids.dtype,
                device=segment_ids.device,
            )
            scratchpad_segments = torch.zeros(
                batch_size,
                self.num_scratchpad_tokens,
                dtype=segment_ids.dtype,
                device=segment_ids.device,
            )
            work_segments = torch.cat(
                (control_segments, segment_ids, scratchpad_segments),
                dim=1,
            )
            joint_segment_ids = torch.cat(
                (segment_ids, work_segments),
                dim=1,
            )
            prompt_streams = torch.zeros_like(segment_ids)
            control_stream = torch.ones_like(control_segments)
            work_streams = torch.full_like(segment_ids, 2)
            # The public debug schema has three streams. Scratchpad slots are
            # mutable work, so report them with the aligned workspace.
            scratchpad_streams = torch.full_like(scratchpad_segments, 2)
            joint_stream_ids = torch.cat(
                (
                    prompt_streams,
                    control_stream,
                    work_streams,
                    scratchpad_streams,
                ),
                dim=1,
            )
        if stage_states is not None:
            stage_states.append(work_state[:, 1:output_end].detach())
        if training_stage_states is not None:
            work_state.retain_grad()
            training_stage_states.append(work_state)

        for _ in range(self.num_loops):
            joint_state = torch.cat((prompt_memory, work_state), dim=1)
            if layer_diagnostics is None:
                candidate_state = self.block(joint_state, joint_mask)
            else:
                candidate_state = self.block(
                    joint_state,
                    joint_mask,
                    layer_diagnostics=layer_diagnostics,
                    segment_ids=joint_segment_ids,
                    query_mask=torch.cat(
                        (torch.zeros_like(prompt_mask), work_mask),
                        dim=1,
                    ),
                    key_mask=torch.cat((prompt_mask, work_mask), dim=1),
                    key_stream_ids=joint_stream_ids,
                )
            work_state = candidate_state[:, prompt_len:]
            if stage_states is not None:
                stage_states.append(work_state[:, 1:output_end].detach())
            if training_stage_states is not None:
                work_state.retain_grad()
                training_stage_states.append(work_state)
            if hypothesis_states is not None:
                hypothesis_states.append(work_state[:, 1:output_end])

        return work_state


class UniversalProcessor(nn.Module):
    def __init__(
        self,
        block: Block,
        time_embedding: nn.Embedding,
        *,
        use_act: bool,
        max_loops: int,
        halting_unit: nn.Linear | None = None,
        halting_prob_threshold: float = 0.01,
        tail_halt_fraction: float | None = None,
    ) -> None:
        super().__init__()
        if max_loops < 1:
            raise ValueError("max_loops must be positive")
        if use_act and halting_unit is None:
            raise ValueError("ACT requires a halting unit")
        if tail_halt_fraction is not None and not 0.0 < tail_halt_fraction <= 1.0:
            raise ValueError("tail_halt_fraction must be in (0, 1]")

        self.block = block
        self.time_embedding = time_embedding
        self.use_act = use_act
        self.max_loops = max_loops
        self.halting_unit = halting_unit
        self.halting_prob_threshold = halting_prob_threshold
        self.tail_halt_fraction = tail_halt_fraction

    def forward(
        self,
        x: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None = None,
        *,
        collect_act_diagnostics: bool = False,
        segment_ids: Tensor | None = None,
        layer_diagnostics: list[dict[str, object]] | None = None,
        stage_states: list[Tensor] | None = None,
        training_stage_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, object] | None]:
        if self.use_act:
            return self.forward_act(
                x,
                position_signal,
                attention_mask,
                collect_act_diagnostics=collect_act_diagnostics,
                segment_ids=segment_ids,
                layer_diagnostics=layer_diagnostics,
                stage_states=stage_states,
                training_stage_states=training_stage_states,
            )
        return self.forward_fixed(
            x,
            position_signal,
            attention_mask,
            segment_ids=segment_ids,
            layer_diagnostics=layer_diagnostics,
            stage_states=stage_states,
            training_stage_states=training_stage_states,
        )

    def forward_act(
        self,
        x: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None = None,
        *,
        collect_act_diagnostics: bool = False,
        segment_ids: Tensor | None = None,
        layer_diagnostics: list[dict[str, object]] | None = None,
        stage_states: list[Tensor] | None = None,
        training_stage_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, object] | None]:
        batch_size, seq_len, _ = x.shape
        curr_halt_prob = x.new_zeros(batch_size, seq_len, 1)
        update_counts = torch.zeros_like(curr_halt_prob)
        remainders = torch.zeros_like(curr_halt_prob)
        cap_forced_mask = None
        tail_forced_mask = None
        if DBUG and collect_act_diagnostics:
            cap_forced_mask = torch.zeros_like(curr_halt_prob, dtype=torch.bool)
            tail_forced_mask = torch.zeros_like(curr_halt_prob, dtype=torch.bool)
        weighted_output = torch.zeros_like(x)
        threshold = 1.0 - self.halting_prob_threshold
        if attention_mask is None:
            valid_tokens = torch.ones_like(curr_halt_prob, dtype=torch.bool)
        else:
            valid_tokens = attention_mask.unsqueeze(-1).bool()

        if stage_states is not None:
            stage_states.append(x.detach())
        if training_stage_states is not None:
            x.retain_grad()
            training_stage_states.append(x)
        for step in range(self.max_loops):
            was_running = valid_tokens & (curr_halt_prob < threshold)
            step_signal = position_signal + self.time_embedding.weight[step]
            block_input = torch.where(was_running, x + step_signal, x)
            if layer_diagnostics is None:
                candidate_x = self.block(block_input, attention_mask)
            else:
                candidate_x = self.block(
                    block_input,
                    attention_mask,
                    layer_diagnostics=layer_diagnostics,
                    segment_ids=segment_ids,
                    query_mask=was_running.squeeze(-1),
                    key_mask=valid_tokens.squeeze(-1),
                    key_stream_ids=torch.full(
                        x.shape[:2],
                        2,
                        dtype=torch.long,
                        device=x.device,
                    ),
                )
            x = torch.where(was_running, candidate_x, x)
            if training_stage_states is not None:
                x.retain_grad()
                training_stage_states.append(x)
            if self.halting_unit is None:
                raise RuntimeError("ACT processor has no halting unit")
            halting_logit = self.halting_unit(x)
            h = torch.sigmoid(halting_logit)
            naturally_halted = was_running & (curr_halt_prob + h >= threshold)
            tail_forced = torch.zeros_like(was_running)
            if self.tail_halt_fraction is not None and step < self.max_loops - 1:
                halted_after_natural = valid_tokens & (~was_running | naturally_halted)
                halted_counts = halted_after_natural.sum(dim=1, keepdim=True)
                valid_counts = valid_tokens.sum(dim=1, keepdim=True)
                tail_cutoff_reached = halted_counts >= (
                    self.tail_halt_fraction * valid_counts
                )
                tail_forced = was_running & ~naturally_halted & tail_cutoff_reached
                if DBUG and tail_forced_mask is not None:
                    tail_forced_mask = tail_forced_mask | tail_forced
            if step == self.max_loops - 1:
                if DBUG and cap_forced_mask is not None:
                    cap_forced_mask = was_running & ~naturally_halted
                newly_halted = was_running
            else:
                newly_halted = naturally_halted | tail_forced
            still_running = was_running & ~newly_halted
            update_counts = update_counts + was_running.to(dtype=x.dtype)
            remainder = 1.0 - curr_halt_prob
            remainders = torch.where(newly_halted, remainder, remainders)
            update_prob = torch.where(
                newly_halted,
                remainder,
                torch.where(still_running, h, torch.zeros_like(h)),
            )
            curr_halt_prob = curr_halt_prob + update_prob
            weighted_output = weighted_output + update_prob * x
            if stage_states is not None:
                provisional_output = weighted_output + (
                    1.0 - curr_halt_prob
                ) * valid_tokens.to(dtype=x.dtype) * x
                stage_states.append(provisional_output.detach())
            if not (valid_tokens & (curr_halt_prob < threshold)).any():
                break

        ponder_time = update_counts + remainders
        ponder_cost = ponder_time[valid_tokens].mean()
        act_diagnostics = None
        if DBUG and cap_forced_mask is not None:
            act_diagnostics = {
                "update_counts": update_counts.squeeze(-1).detach(),
                "remainders": remainders.squeeze(-1).detach(),
                "cap_forced_mask": cap_forced_mask.squeeze(-1).detach(),
                "tail_forced_mask": tail_forced_mask.squeeze(-1).detach(),
                "max_loops": self.max_loops,
                "global_iterations": step + 1,
                "tail_halt_fraction": self.tail_halt_fraction,
            }
        return weighted_output, ponder_cost, act_diagnostics

    def forward_fixed(
        self,
        x: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None = None,
        *,
        segment_ids: Tensor | None = None,
        layer_diagnostics: list[dict[str, object]] | None = None,
        stage_states: list[Tensor] | None = None,
        training_stage_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, None]:
        if attention_mask is None:
            valid_tokens = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        elif attention_mask.shape == x.shape[:2]:
            valid_tokens = attention_mask.bool()
        else:
            valid_tokens = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        if stage_states is not None:
            stage_states.append(x.detach())
        if training_stage_states is not None:
            x.retain_grad()
            training_stage_states.append(x)
        for step in range(self.max_loops):
            x = x + position_signal + self.time_embedding.weight[step]
            if layer_diagnostics is None:
                x = self.block(x, attention_mask)
            else:
                x = self.block(
                    x,
                    attention_mask,
                    layer_diagnostics=layer_diagnostics,
                    segment_ids=segment_ids,
                    query_mask=valid_tokens,
                    key_mask=valid_tokens,
                    key_stream_ids=torch.full(
                        x.shape[:2],
                        2,
                        dtype=torch.long,
                        device=x.device,
                    ),
                )
            if stage_states is not None:
                stage_states.append(x.detach())
            if training_stage_states is not None:
                x.retain_grad()
                training_stage_states.append(x)

        return x, x.new_zeros(()), None


# TODO(piydatta): Experiment with using this processor to turn stored gradients
# from the previous training step into temporary fast-weight updates. It is
# intentionally not instantiated or used by Model yet.
class GradientUpdateNetwork(nn.Module):
    def __init__(self, *, use_act: bool, max_loops: int) -> None:
        super().__init__()
        self.row_embedding = nn.Embedding(D_MODEL, D_MODEL)
        block = Block()
        time_embedding = nn.Embedding(max_loops, D_MODEL)
        halting_unit = nn.Linear(D_MODEL, 1) if use_act else None
        self.processor = UniversalProcessor(
            block,
            time_embedding,
            use_act=use_act,
            max_loops=max_loops,
            halting_unit=halting_unit,
        )
        self.final_norm = RMSNorm(D_MODEL)

        init_std = 0.02
        nn.init.normal_(self.row_embedding.weight, std=init_std)
        nn.init.normal_(time_embedding.weight, std=init_std)

    def forward(self, gradient_tokens: Tensor) -> tuple[Tensor, Tensor]:
        added_batch_dimension = gradient_tokens.ndim == 2
        if added_batch_dimension:
            gradient_tokens = gradient_tokens.unsqueeze(0)
        if gradient_tokens.ndim != 3 or gradient_tokens.shape[-1] != D_MODEL:
            raise ValueError(
                "gradient tokens must have shape (rows, D_MODEL) or "
                "(batch, rows, D_MODEL)"
            )

        row_count = gradient_tokens.shape[-2]
        if row_count > self.row_embedding.num_embeddings:
            raise ValueError("gradient token count exceeds row embedding size")
        row_positions = torch.arange(row_count, device=gradient_tokens.device)
        row_signal = self.row_embedding(row_positions)
        x, ponder_cost, _ = self.processor(gradient_tokens, row_signal)
        x = self.final_norm(x)
        if added_batch_dimension:
            x = x.squeeze(0)
        return x, ponder_cost


class Model(nn.Module):

    def __init__(self, spec: ModelSpec, use_act: bool = False) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.use_act = use_act
        self.max_loops = ACT_MAX_LOOPS if self.use_act else FIXED_LOOPS
        if DBUG:
            self.collect_act_diagnostics = False
            self.collect_model_diagnostics = False
            self.collect_training_diagnostics = False
            self._debug_training_context = None

        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        init_std = 0.02
        nn.init.normal_(self.token_embedding.weight, std=init_std)
        nn.init.normal_(self.position_embedding.weight, std=init_std)

        # Keep every shared parameter ahead of the ACT/fixed branch so both
        # modes receive identical shared initialization under the same seed.
        self.segment_embedding = nn.Embedding(
            NUM_SEGMENTS,
            D_MODEL,
            padding_idx=0,
        )
        nn.init.normal_(self.segment_embedding.weight, std=init_std)
        with torch.no_grad():
            self.segment_embedding.weight[0].zero_()

        if self.use_act:
            time_embedding = nn.Embedding(self.max_loops, D_MODEL)
            nn.init.normal_(time_embedding.weight, std=init_std)
            self.processor = UniversalProcessor(
                block,
                time_embedding,
                use_act=True,
                max_loops=self.max_loops,
                halting_unit=nn.Linear(D_MODEL, 1),
                tail_halt_fraction=ACT_TAIL_HALT_FRACTION,
            )
        else:
            self.control_token = nn.Parameter(torch.empty(D_MODEL))
            self.workspace_token = nn.Parameter(torch.empty(D_MODEL))
            nn.init.normal_(self.control_token, std=init_std)
            nn.init.normal_(self.workspace_token, std=init_std)
            self.processor = SynchronizedProcessor(
                block,
                num_loops=self.max_loops,
                num_scratchpad_tokens=NUM_SCRATCHPAD_TOKENS,
            )
            if USE_LATENT_HYPOTHESES:
                self.hypothesis_prior = nn.Parameter(
                    torch.empty(self.max_loops)
                )
                nn.init.normal_(self.hypothesis_prior, std=init_std)
            # Keep this after all existing fixed-path parameters so adding the
            # scratchpad does not shift their seeded initialization.
            self.scratchpad_embedding = nn.Embedding(
                NUM_SCRATCHPAD_TOKENS,
                D_MODEL,
            )
            nn.init.normal_(self.scratchpad_embedding.weight, std=init_std)

        if DBUG:
            self._debug_initial_parameter_buffers = {}
            with torch.no_grad():
                for index, (name, parameter) in enumerate(
                    self.named_parameters()
                ):
                    buffer_name = f"_debug_initial_parameter_{index}"
                    self.register_buffer(
                        buffer_name,
                        parameter.detach().clone(),
                        persistent=False,
                    )
                    self._debug_initial_parameter_buffers[name] = buffer_name

    def _run_processor(
        self,
        token_state: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None,
        *,
        segment_ids: Tensor | None = None,
        collect_act_diagnostics: bool = False,
        layer_diagnostics: list[dict[str, object]] | None = None,
        stage_states: list[Tensor] | None = None,
        training_stage_states: list[Tensor] | None = None,
        hypothesis_states: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, object] | None]:
        if self.use_act:
            processor_kwargs = {
                "collect_act_diagnostics": collect_act_diagnostics,
            }
            if (
                layer_diagnostics is not None
                or training_stage_states is not None
            ):
                processor_kwargs.update(
                    {
                        "segment_ids": segment_ids,
                        "layer_diagnostics": layer_diagnostics,
                        "stage_states": stage_states,
                        "training_stage_states": training_stage_states,
                    }
                )
            return self.processor(
                token_state,
                position_signal,
                attention_mask,
                **processor_kwargs,
            )

        batch_size = token_state.shape[0]
        prompt_memory = token_state + position_signal
        control_state = self.control_token.view(1, 1, -1).expand(
            batch_size,
            -1,
            -1,
        )
        workspace_state = prompt_memory + self.workspace_token.view(1, 1, -1)
        scratchpad_state = self.scratchpad_embedding.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        work_state = torch.cat(
            (control_state, workspace_state, scratchpad_state),
            dim=1,
        )
        if (
            layer_diagnostics is None
            and training_stage_states is None
            and hypothesis_states is None
        ):
            work_state = self.processor(
                prompt_memory,
                work_state,
                attention_mask,
            )
        else:
            work_state = self.processor(
                prompt_memory,
                work_state,
                attention_mask,
                segment_ids=segment_ids,
                layer_diagnostics=layer_diagnostics,
                stage_states=stage_states,
                training_stage_states=training_stage_states,
                hypothesis_states=hypothesis_states,
            )
        x = work_state[:, 1 : 1 + token_state.shape[1]]
        return x, x.new_zeros(()), None

    @staticmethod
    def _debug_segment_means(
        values: Tensor,
        segment_ids: Tensor,
        valid_tokens: Tensor,
    ) -> Tensor:
        return torch.stack(
            [
                Block._masked_mean(
                    values,
                    valid_tokens & segment_ids.eq(segment),
                )
                for segment in range(NUM_SEGMENTS)
            ]
        )

    def _debug_parameter_stats(self) -> dict[str, dict[str, Tensor]]:
        stats = {}
        for name, parameter in self.named_parameters():
            buffer_name = self._debug_initial_parameter_buffers[name]
            initial = getattr(self, buffer_name).detach().float()
            current = parameter.detach().float()
            initial_norm = initial.norm()
            delta_norm = (current - initial).norm()
            stats[name] = {
                "norm": current.norm().detach(),
                "delta_norm": delta_norm.detach(),
                "relative_delta": (
                    delta_norm / initial_norm.clamp_min(1e-12)
                ).detach(),
            }
        return stats

    def _debug_segment_embedding_stats(self) -> dict[str, Tensor]:
        buffer_name = self._debug_initial_parameter_buffers[
            "segment_embedding.weight"
        ]
        initial = getattr(self, buffer_name).detach().float()
        current = self.segment_embedding.weight.detach().float()
        initial_norms = initial.norm(dim=-1)
        current_norms = current.norm(dim=-1)
        delta_norms = (current - initial).norm(dim=-1)
        cosine_denominator = current_norms * initial_norms
        initial_cosines = torch.where(
            cosine_denominator > 0,
            (current * initial).sum(dim=-1)
            / cosine_denominator.clamp_min(1e-12),
            torch.zeros_like(cosine_denominator),
        )
        normalized = current / current_norms[:, None].clamp_min(1e-12)
        cosine_matrix = normalized @ normalized.transpose(0, 1)
        nonzero_rows = current_norms > 0
        cosine_matrix = cosine_matrix * (
            nonzero_rows[:, None] & nonzero_rows[None, :]
        ).to(cosine_matrix.dtype)
        return {
            "norms": current_norms.detach(),
            "delta_norms": delta_norms.detach(),
            "relative_deltas": (
                delta_norms / initial_norms.clamp_min(1e-12)
            ).detach(),
            "initial_cosines": initial_cosines.detach(),
            "cosine_matrix": cosine_matrix.detach(),
        }

    def consume_training_grad_diagnostics(self) -> dict[str, object]:
        if not DBUG or self._debug_training_context is None:
            raise RuntimeError("no sampled training diagnostics are available")
        context = self._debug_training_context
        self._debug_training_context = None

        segment_signal = context["segment_signal"]
        segment_ids = context["segment_ids"]
        valid_tokens = context["valid_tokens"]
        training_stage_states = context["training_stage_states"]
        synchronized = context["synchronized"]
        if segment_signal.grad is None:
            raise RuntimeError("segment signal did not receive a training gradient")

        segment_gradient = segment_signal.grad.detach().float()
        segment_counts = torch.stack(
            [
                (valid_tokens & segment_ids.eq(segment)).sum()
                for segment in range(NUM_SEGMENTS)
            ]
        ).detach()
        stages = []
        for step, state in enumerate(training_stage_states):
            if state.grad is None:
                raise RuntimeError(
                    f"recurrent training state at step {step} has no gradient"
                )
            state_gradient = state.grad.detach().float()
            control_gradient_rms = None
            if synchronized:
                control_gradient_rms = (
                    state_gradient[:, :1].square().mean().sqrt().detach()
                )
                # Existing segment diagnostics describe the output-aligned
                # workspace. Scratchpad slots are deliberately not assigned a
                # grammatical segment, so exclude them from these summaries.
                state_gradient = state_gradient[
                    :, 1 : 1 + valid_tokens.shape[1]
                ]
            stages.append(
                {
                    "step": step,
                    "state_grad_rms": Block._masked_rms(
                        state_gradient,
                        valid_tokens,
                    ).detach(),
                    "state_grad_rms_by_segment": Block._rms_by_segment(
                        state_gradient,
                        segment_ids,
                        valid_tokens,
                    ).detach(),
                    "control_grad_rms": control_gradient_rms,
                }
            )
        final_gradient_rms = stages[-1]["state_grad_rms"]
        for stage in stages:
            stage["relative_to_final"] = (
                stage["state_grad_rms"]
                / final_gradient_rms.clamp_min(1e-12)
            ).detach()

        return {
            "segment_token_counts": segment_counts,
            "segment_signal_grad_rms": Block._masked_rms(
                segment_gradient,
                valid_tokens,
            ).detach(),
            "segment_signal_grad_rms_by_segment": Block._rms_by_segment(
                segment_gradient,
                segment_ids,
                valid_tokens,
            ).detach(),
            "stages": stages,
        }

    def _build_model_diagnostics(
        self,
        *,
        x: Tensor,
        logits: Tensor,
        input_ids: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None,
        segment_ids: Tensor,
        valid_tokens: Tensor,
        layer_diagnostics: list[dict[str, object]],
        stage_states: list[Tensor],
    ) -> dict[str, object]:
        segment_token_counts = torch.stack(
            [
                (valid_tokens & segment_ids.eq(segment)).sum()
                for segment in range(NUM_SEGMENTS)
            ]
        ).detach()
        final_state_rms = Block._rms_by_segment(
            x.detach().float(),
            segment_ids,
            valid_tokens,
        ).detach()
        log_probabilities = logits.detach().float().log_softmax(dim=-1)
        probabilities = log_probabilities.exp()
        logit_entropy = -(probabilities * log_probabilities).sum(dim=-1)
        final_logit_entropy = self._debug_segment_means(
            logit_entropy,
            segment_ids,
            valid_tokens,
        ).detach()

        stage_logits = [
            {
                "step": step,
                "logits": self.head(self.final_norm(state)).detach(),
            }
            for step, state in enumerate(stage_states)
        ]

        base_token_state = self.token_embedding(input_ids)
        zero_segment_x, _, _ = self._run_processor(
            base_token_state,
            position_signal,
            attention_mask,
        )
        counterfactual_states = {"zero": zero_segment_x}
        counterfactual_mappings = {
            "permuted": (0, 2, 3, 1, 4),
            "zero_nx": (0, 0, 0, 3, 4),
            "zero_t": (0, 1, 2, 0, 4),
            "swap_n_x": (0, 2, 1, 3, 4),
        }
        for name, mapping in counterfactual_mappings.items():
            mapping_tensor = torch.tensor(
                mapping,
                dtype=torch.long,
                device=segment_ids.device,
            )
            remapped_state = base_token_state + self.segment_embedding(
                mapping_tensor[segment_ids]
            )
            counterfactual_x, _, _ = self._run_processor(
                remapped_state,
                position_signal,
                attention_mask,
            )
            counterfactual_states[name] = counterfactual_x

        return {
            "segment_token_counts": segment_token_counts,
            "segment_embedding": self._debug_segment_embedding_stats(),
            "parameter_stats": self._debug_parameter_stats(),
            "final_state_rms_by_segment": final_state_rms,
            "final_logit_entropy_by_segment": final_logit_entropy,
            "layers": layer_diagnostics,
            "stage_logits": stage_logits,
            "segment_counterfactual_logits": {
                name: self.head(self.final_norm(state)).detach()
                for name, state in counterfactual_states.items()
            },
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_signal = self.position_embedding(positions)
        segment_markers = (
            input_ids.eq(N_TOKEN_ID).long()
            + 2 * input_ids.eq(X_TOKEN_ID).long()
            + 3 * input_ids.eq(T_TOKEN_ID).long()
            + 4 * input_ids.eq(ANS_TOKEN_ID).long()
        )
        valid_tokens = input_ids.ne(PAD_TOKEN_ID)
        if attention_mask is not None and attention_mask.shape == input_ids.shape:
            valid_tokens = attention_mask.bool()
        segment_markers = segment_markers * valid_tokens.long()
        segment_ids = segment_markers.cummax(dim=1).values
        segment_ids = segment_ids.masked_fill(~valid_tokens, 0)
        segment_signal = self.segment_embedding(segment_ids)
        collect_training_diagnostics = (
            DBUG
            and self.training
            and torch.is_grad_enabled()
            and self.collect_training_diagnostics
        )
        if collect_training_diagnostics:
            self._debug_training_context = None
            segment_signal.retain_grad()
        token_state = self.token_embedding(input_ids) + segment_signal
        collect_act_diagnostics = (
            self.collect_act_diagnostics if DBUG else False
        )
        collect_model_diagnostics = (
            self.collect_model_diagnostics if DBUG else False
        )
        layer_diagnostics = [] if collect_model_diagnostics else None
        stage_states = [] if collect_model_diagnostics else None
        training_stage_states = [] if collect_training_diagnostics else None
        hypothesis_states = (
            []
            if USE_LATENT_HYPOTHESES and not self.use_act
            else None
        )
        x, ponder_cost, act_diagnostics = self._run_processor(
            token_state,
            position_signal,
            attention_mask,
            segment_ids=(segment_ids if collect_model_diagnostics else None),
            collect_act_diagnostics=collect_act_diagnostics,
            layer_diagnostics=layer_diagnostics,
            stage_states=stage_states,
            training_stage_states=training_stage_states,
            hypothesis_states=hypothesis_states,
        )
        if collect_training_diagnostics:
            assert training_stage_states is not None
            self._debug_training_context = {
                "segment_signal": segment_signal,
                "segment_ids": segment_ids.detach(),
                "valid_tokens": valid_tokens.detach(),
                "training_stage_states": training_stage_states,
                "synchronized": not self.use_act,
            }
        hypothesis_logits = None
        hypothesis_log_prior = None
        if hypothesis_states is not None:
            if len(hypothesis_states) != self.max_loops:
                raise RuntimeError("processor did not produce every hypothesis")
            stacked_hypothesis_states = torch.stack(
                hypothesis_states,
                dim=1,
            )
            hypothesis_logits = torch.stack(
                [
                    self.head(self.final_norm(state))
                    for state in hypothesis_states
                ],
                dim=1,
            )
            hypothesis_log_prior = F.log_softmax(
                self.hypothesis_prior,
                dim=0,
            )
            winning_hypothesis = self.hypothesis_prior.argmax(dim=0)
            x = stacked_hypothesis_states[:, winning_hypothesis]
            logits = hypothesis_logits[:, winning_hypothesis]
        else:
            logits = self.head(self.final_norm(x))

        training_auxiliary = {
            "ponder_cost": ponder_cost,
            "hypothesis_logits": hypothesis_logits,
            "hypothesis_log_prior": hypothesis_log_prior,
        }
        if not DBUG:
            if self.training:
                return logits, training_auxiliary
            return logits, ponder_cost

        if act_diagnostics is not None:
            act_diagnostics["ponder_weight"] = PONDER_WEIGHT
        model_diagnostics = None
        if collect_model_diagnostics:
            assert layer_diagnostics is not None
            assert stage_states is not None
            with torch.no_grad():
                model_diagnostics = self._build_model_diagnostics(
                    x=x,
                    logits=logits,
                    input_ids=input_ids,
                    position_signal=position_signal,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
                    valid_tokens=valid_tokens,
                    layer_diagnostics=layer_diagnostics,
                    stage_states=stage_states,
                )
        auxiliary = {
            "ponder_cost": ponder_cost,
            "act": act_diagnostics,
            "model_diagnostics": model_diagnostics,
            "hypothesis_logits": hypothesis_logits,
            "hypothesis_log_prior": hypothesis_log_prior,
        }
        return logits, auxiliary


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec, use_act=USE_ACT)
    assert_model_state(model, spec)
    if DBUG:
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        print(f"TOTAL_PARAMETERS: {total_parameters:,}")
        print(
            "Constants\n"
            f" D_MODEL: {D_MODEL}, NUM_HEADS: {NUM_HEADS}, D_FF: {D_FF}, "
            f"PONDER_WEIGHT: {PONDER_WEIGHT}, USE_ACT: {USE_ACT}, "
            f"FIXED_LOOPS: {FIXED_LOOPS}, ACT_MAX_LOOPS: {ACT_MAX_LOOPS}, "
            f"ACT_TAIL_HALT_FRACTION: {ACT_TAIL_HALT_FRACTION}, "
            f"USE_MUON: {USE_MUON}, MUON_LR: {MUON_LR}, "
            f"MUON_MOMENTUM: {MUON_MOMENTUM}, "
            f"MUON_WEIGHT_DECAY: {MUON_WEIGHT_DECAY}, "
            f"MUON_ADJUST_LR_FN: {MUON_ADJUST_LR_FN}, "
            f"LR_DECAY_START_STEP: {LR_DECAY_START_STEP}, "
            f"LR_DECAY_END_STEP: {LR_DECAY_END_STEP}, "
            f"LR_DECAY_MIN_FACTOR: {LR_DECAY_MIN_FACTOR}, "
            f"USE_LATENT_HYPOTHESES: {USE_LATENT_HYPOTHESES}, "
            f"HYPOTHESIS_TEMPERATURE: {HYPOTHESIS_TEMPERATURE}, "
            f"HYPOTHESIS_ALL_LOSS_WEIGHT: "
            f"{HYPOTHESIS_ALL_LOSS_WEIGHT}, "
            f"HYPOTHESIS_DEPTH_PENALTY: {HYPOTHESIS_DEPTH_PENALTY}, "
            f"NUM_SCRATCHPAD_TOKENS: {NUM_SCRATCHPAD_TOKENS}, "
            f"DBUG: {DBUG}"
        )
    return model


class CombinedOptimizer:
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [
            group
            for optimizer in self.optimizers
            for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        result = None
        for optimizer in self.optimizers:
            value = (
                optimizer.step(closure=closure)
                if closure is not None
                else optimizer.step()
            )
            if value is not None:
                result = value
        return result

    def state_dict(self) -> dict:
        return {
            "optimizers": [
                optimizer.state_dict() for optimizer in self.optimizers
            ]
        }


class CosineDecayScheduler:
    def __init__(
        self,
        optimizer,
        *,
        start_step: int,
        end_step: int,
        min_factor: float,
    ) -> None:
        if start_step < 0:
            raise ValueError("decay start step must be non-negative")
        if end_step <= start_step:
            raise ValueError("decay end step must be after the start step")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("minimum learning-rate factor must be in [0, 1]")

        self.optimizer = optimizer
        self.start_step = start_step
        self.end_step = end_step
        self.min_factor = min_factor
        self.completed_steps = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self) -> None:
        self.completed_steps += 1
        if self.completed_steps <= self.start_step:
            factor = 1.0
        elif self.completed_steps >= self.end_step:
            factor = self.min_factor
        else:
            progress = (
                (self.completed_steps - self.start_step)
                / (self.end_step - self.start_step)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_factor + (1.0 - self.min_factor) * cosine

        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * factor


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    if USE_MUON:
        muon_parameters = []
        adamw_parameters = []
        for name, parameter in model.named_parameters():
            use_muon = (
                parameter.ndim == 2
                and name.startswith("processor.block.")
                and name.endswith(".weight")
            )
            if use_muon:
                muon_parameters.append(parameter)
            else:
                adamw_parameters.append(parameter)

        muon = torch.optim.Muon(
            muon_parameters,
            lr=MUON_LR,
            momentum=MUON_MOMENTUM,
            weight_decay=MUON_WEIGHT_DECAY,
            adjust_lr_fn=MUON_ADJUST_LR_FN,
        )
        adamw = torch.optim.AdamW(
            adamw_parameters,
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
        optimizer = CombinedOptimizer([muon, adamw])
        scheduler = CosineDecayScheduler(
            optimizer,
            start_step=LR_DECAY_START_STEP,
            end_step=LR_DECAY_END_STEP,
            min_factor=LR_DECAY_MIN_FACTOR,
        )
        return OptimizerBundle(optimizer, scheduler=scheduler)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        capturable=spec.device_type == "cuda",
    )
    scheduler = CosineDecayScheduler(
        optimizer,
        start_step=LR_DECAY_START_STEP,
        end_step=LR_DECAY_END_STEP,
        min_factor=LR_DECAY_MIN_FACTOR,
    )
    return OptimizerBundle(optimizer, scheduler=scheduler)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
)
