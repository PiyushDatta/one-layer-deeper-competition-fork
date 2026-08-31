from __future__ import annotations

from collections.abc import Callable
import math
import time

import torch
import torch.nn.functional as F
from benchmark import (
    assert_model_state,
    BackwardPassContext,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
    TokenLossBatch,
)
from torch import nn, Tensor

# State is 12 * D_MODEL**2 + ~118 * D_MODEL: 12,703,873 of the 500M ceiling.
# Width is traded for loops on purpose, since cost per loop scales as D_MODEL**2
# while depth is what the T ladder rewards.
D_MODEL = 1024
NUM_HEADS = 8  # head_dim 128
D_FF = 4 * D_MODEL
PONDER_WEIGHT = 0.005
USE_ACT = False
# Training depth is bounded by the clock, evaluation depth is nearly free
# because it is forward-only over a few hundred rows.
TRAIN_LOOPS = 8
# The learned ladder settles on stride 2 (T=1->exit 2, T=2->exit 4,
# T=3->exit 6), so certifying T=64 needs 128 exits, not 64. Evaluation used
# 5.4s of its 30s budget at 64 loops, so this stays affordable.
EVAL_LOOPS = 128
ACT_MAX_LOOPS = 16
ACT_TAIL_HALT_FRACTION = None
USE_MUON = True
MUON_LR = 1e-3
MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.1
MUON_ADJUST_LR_FN = "match_rms_adamw"
# Fraction of the wall-clock budget, not a step count, so one value works at
# every tier. Constant then a ~15% decay is the warmup-stable-decay shape.
LR_DECAY_HOLD_FRACTION = 0.85
LR_DECAY_MIN_FACTOR = 0.001
USE_LATENT_HYPOTHESES = True
HYPOTHESIS_TEMPERATURE = 1.0
# The soft-min is satisfied when some exit is right, but evaluation reads the
# argmax-prior exit, so nothing penalises a wrong prior. This weight fits the
# prior to the detached posterior. Untuned. Set to 0.0 to disable.
SELECTOR_LOSS_WEIGHT = 1.0
# Measured: the exit prior collapses to exit 0 for every row at every T, and
# per-exit accuracy then decays monotonically with depth (0.26 -> 0.03 over
# eight exits) because only exit 0 ever receives gradient. The posterior is
# proportional to prior * exp(-loss), so a leading exit reinforces itself and
# the M step accelerates it. Two counter-measures:
#
#   * evidence-only posterior — drop the prior from the M step target so it
#     fits "which exit explains this row", not "which exit already won";
#   * mutual information between T and the exit choice — maximise the entropy
#     of the batch-marginal prior while minimising per-row entropy, so the
#     selector must be sharp per row yet spread across the batch. Since the
#     prior is a function of T only, spreading it across the batch is
#     spreading it across T, which is the exit-k-for-T-k alignment we want.
#
# Set EXIT_MI_WEIGHT to 0.0 to ablate.
EXIT_MI_WEIGHT = 1.0
POSTERIOR_USES_PRIOR = False
# Measured after the collapse was broken: the model picks an ANTI-monotone
# T -> exit map (T=1 -> exit 7, T=2 -> exit 4, T=3 -> exit 1) and returns to
# 100% train accuracy. A composable squaring needs depth to GROW with T, so an
# arbitrary permutation is only expressible if the loop index is being used as
# an index rather than as iterated computation -- B's residual state evolves
# distinguishably with loop count, so B^k can encode f(x, k), and the prior
# makes k a proxy for T. Change 7 hid T from attention; depth leaked it back.
# Forcing the exit centre to be stride * scalar(T) removes the permutation
# freedom and is also the only form that can extrapolate to T=64.
EXIT_MONOTONE_IN_T = True
EXIT_STRIDE_INIT = 1.0
EXIT_WIDTH_INIT = 1.0
EXIT_CENTRE_INIT = 2.0
# A free scalar head just learns to decrease in T. Order has to be built into
# the digit reader itself. Set False to fall back to the free head.
EXIT_ORDERED_DIGITS = True
DIGIT_OFFSET = 7
NUM_DIGITS = 10
MAX_T_DIGITS = 3
NUM_SCRATCHPAD_TOKENS = 4
# The tied head makes logit scale grow as sqrt(D_MODEL), which peaks the
# softmax and lets retokenization compound into a feedback loop across loops.
# Divide that back out. Only the D_MODEL=4096 value of 8.0 is measured, the
# scaling to other widths is inferred.
RETOKENIZE_TEMPERATURE = D_MODEL**0.5 / 8.0
# sigmoid(-2.0) is about 0.12, so the bottleneck starts nearly closed and the
# model opens it only if it pays for itself.
RETOKENIZE_GATE_INIT = -2.0
# MAIN's tape holds discrete tokens and its modules are not differentiable.
# Straight-through makes the forward pass commit to one token while the
# backward pass keeps the soft gradient.
RETOKENIZE_STRAIGHT_THROUGH = True
# Gates carry evidence about whether their mechanism pays for itself, but
# weight decay drifts a -2.0 logit toward 0, that is toward OPENING, whether or
# not it helps. Exempting them keeps the trajectory readable.
GATE_PARAMETER_NAMES = ("retokenize_gate", "history_gate")
GATE_WEIGHT_DECAY = 0.0
# Embeddings and the tied head live in AdamW, which is where memorising the
# answer distribution happens, so this needs to be tunable separately from
# the block's Muon decay.
ADAMW_WEIGHT_DECAY = 0.1
# 512 against E1's 600 training rows with drop_last is one batch per epoch, so
# full-batch descent. 64 gives 9 batches and real gradient noise. This is one
# import-time constant with no spec available, so it also applies to Hard,
# where 512 may well be right. Eval is pinned separately, otherwise the 14
# depth rungs would run at the training batch size and slow the ladder down.
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 512
# Sharpness-aware minimisation. Measure the gradient at w + rho * g/||g||, the
# locally worst nearby point, and apply it at w. Biases toward wide basins,
# which is where the compositional solution should live and where memorising
# 600 points individually does not. Costs a second evaluator-owned pass, so it
# halves the step count. 0.0 ablates back to one pass.
SAM_RHO = 0.05
# Action history. MAIN drops every task to 0/10 without it, so each loop is fed
# what changed last iteration and where. Low rank because a full D x D
# projection is 1.05M parameters at this width.
USE_ACTION_HISTORY = True
HISTORY_RANK = 64
HISTORY_GATE_INIT = -2.0
# The answer is read out at the last few valid prompt positions. Seeding those
# workspace slots from dedicated learned queries instead of prompt content took
# a probe on x*y mod 323 from 0.085 to 0.812 held-out exact.
USE_ANSWER_QUERIES = True
MAX_ANSWER_DIGITS = 3
# Control/data separation. The recurrent block may not read the T digits, so it
# cannot pick between x^2, x^4 and x^8 for one x and is pushed toward learning
# the single squaring step. Only the exit selector sees T.
HIDE_T_FROM_BLOCK = True
BUILD_ON_ACCELERATOR = True
PRINT_LOGS = True
DBUG = False

PAD_TOKEN_ID = 0
N_TOKEN_ID = 2
X_TOKEN_ID = 3
T_TOKEN_ID = 4
ANS_TOKEN_ID = 5
NUM_SEGMENTS = 5
T_SEGMENT = 3


def training_loss(
    logits: Tensor,
    labels: Tensor,
    auxiliary: object,
) -> Tensor:
    task_loss = F.cross_entropy(logits, labels)
    ponder_cost = auxiliary["ponder_cost"] if isinstance(auxiliary, dict) else auxiliary
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
    sequence_losses = (token_losses * candidate_valid_mask).sum(dim=-1) / target_counts
    rows_with_targets = batch.valid_mask.any(dim=-1)

    # Per-row soft-min over exits. A row's own label picks its depth: for a T=k
    # prompt only the exit whose readout equals x^(2^k) has low loss, so the
    # tied block is trained as one composable step rather than a T-specific
    # map. Averaging rows first would force a single exit on the whole batch,
    # and scoring every exit against the final label would demand B(B(x)) ==
    # B(x), which is the one operator shape that cannot extrapolate in T.
    temperature = HYPOTHESIS_TEMPERATURE
    row_losses = -temperature * torch.logsumexp(
        hypothesis_log_prior - sequence_losses / temperature,
        dim=1,
    )
    task_loss = row_losses[rows_with_targets].mean()

    if SELECTOR_LOSS_WEIGHT > 0.0:
        # Explicit M step. The posterior is detached, so this trains the exit
        # selector to predict which exit actually explains the row without
        # pulling the block toward any particular depth. Including the prior in
        # the target makes it self-reinforcing, which is how the collapse to
        # exit 0 happens, so the evidence-only form is the default.
        with torch.no_grad():
            posterior_logits = -sequence_losses / temperature
            if POSTERIOR_USES_PRIOR:
                posterior_logits = posterior_logits + hypothesis_log_prior
            posterior = F.softmax(posterior_logits, dim=1)
        selector_loss = -(posterior * hypothesis_log_prior).sum(dim=1)
        task_loss = task_loss + SELECTOR_LOSS_WEIGHT * (
            selector_loss[rows_with_targets].mean()
        )

    if EXIT_MI_WEIGHT > 0.0:
        # Maximise I(T; exit) = H(marginal exit) - E_row[H(exit | row)].
        # The first term stops every row routing to the same exit, the second
        # keeps each row's choice decisive. Standard load balancing, but the
        # information framing is the one that matters here: the prior is a
        # function of T alone, so a high-entropy marginal is only reachable by
        # sending different T to different exits.
        row_log_prior = hypothesis_log_prior
        if row_log_prior.dim() == 1:
            row_log_prior = row_log_prior.expand(batch_size, -1)
        scored_log_prior = row_log_prior[rows_with_targets]
        scored_prior = scored_log_prior.exp()
        marginal = scored_prior.mean(dim=0)
        marginal_entropy = -(marginal * marginal.clamp_min(1e-9).log()).sum()
        row_entropy = -(scored_prior * scored_log_prior).sum(dim=1).mean()
        task_loss = task_loss - EXIT_MI_WEIGHT * (marginal_entropy - row_entropy)

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
        mlp_update = self.down(F.gelu(self.up(self.mixer_norm(post_attention))))
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
            raise ValueError("diagnostic key_stream_ids must match the block sequence")

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
            attention_probabilities * attention_probabilities.clamp_min(1e-12).log()
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
                    mean_attention * selected_keys[:, None, :].to(mean_attention.dtype)
                ).sum(dim=-1)
                key_masses.append(cls._masked_mean(mass_per_query, selected_queries))
            attention_mass_rows.append(torch.stack(key_masses))
            stream_masses = []
            for key_stream in range(3):
                selected_keys = key_mask & key_stream_ids.eq(key_stream)
                mass_per_query = (
                    mean_attention * selected_keys[:, None, :].to(mean_attention.dtype)
                ).sum(dim=-1)
                stream_masses.append(cls._masked_mean(mass_per_query, selected_queries))
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
                "segment_query_counts": torch.stack(segment_query_counts).detach(),
                "attention_mass_by_segment": torch.stack(attention_mass_rows).detach(),
                "attention_mass_by_stream": torch.stack(attention_stream_rows).detach(),
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
        num_loops: int | None = None,
        retokenize: Callable[[Tensor], Tensor] | None = None,
        history: Callable[[Tensor], Tensor] | None = None,
        prompt_key_mask: Tensor | None = None,
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
        # The read-only prompt stream can be masked more tightly than the
        # workspace. The answer is read at target_positions, which sit on the
        # T segment, so the workspace half must stay fully readable even when
        # those same positions are withheld from the prompt half.
        prompt_key = prompt_mask if prompt_key_mask is None else prompt_key_mask.bool()
        work_mask = torch.cat(
            (control_mask, prompt_mask, scratchpad_mask),
            dim=1,
        )
        if attention_mask is not None or prompt_key_mask is not None:
            joint_mask = torch.cat((prompt_key, work_mask), dim=1)

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

        # Seeding this with the initial work state makes the first delta the
        # real first update rather than a step from nothing.
        previous_work_state = work_state
        loops = self.num_loops if num_loops is None else num_loops
        for _ in range(loops):
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
                    key_mask=torch.cat((prompt_key, work_mask), dim=1),
                    key_stream_ids=joint_stream_ids,
                )
            work_state = candidate_state[:, prompt_len:]
            if retokenize is not None:
                # Pull each loop's output back onto the token manifold before
                # the next application so long iteration cannot drift into
                # directions the 17-token vocabulary cannot express. Only the
                # prompt-aligned workspace is constrained, the control token
                # and scratchpad stay continuous working memory.
                work_state = torch.cat(
                    (
                        work_state[:, :1],
                        retokenize(work_state[:, 1:output_end]),
                        work_state[:, output_end:],
                    ),
                    dim=1,
                )
            if stage_states is not None:
                stage_states.append(work_state[:, 1:output_end].detach())
            if training_stage_states is not None:
                work_state.retain_grad()
                training_stage_states.append(work_state)
            if hypothesis_states is not None:
                hypothesis_states.append(work_state[:, 1:output_end])
            if history is not None:
                # Injected after the readout is recorded, so this is
                # scaffolding for the next loop rather than part of any exit's
                # answer.
                readout_state = work_state
                work_state = work_state + history(work_state - previous_work_state)
                previous_work_state = readout_state

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
                provisional_output = (
                    weighted_output
                    + (1.0 - curr_halt_prob) * valid_tokens.to(dtype=x.dtype) * x
                )
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
        # Sized for the deepest exit the selector must be able to address.
        self.max_loops = ACT_MAX_LOOPS if self.use_act else EVAL_LOOPS
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
                num_loops=TRAIN_LOOPS,
                num_scratchpad_tokens=NUM_SCRATCHPAD_TOKENS,
            )
            if USE_LATENT_HYPOTHESES:
                self.hypothesis_prior = nn.Parameter(torch.empty(self.max_loops))
                nn.init.normal_(self.hypothesis_prior, std=init_std)
            # Keep this after all existing fixed-path parameters so adding the
            # scratchpad does not shift their seeded initialization.
            self.scratchpad_embedding = nn.Embedding(
                NUM_SCRATCHPAD_TOKENS,
                D_MODEL,
            )
            nn.init.normal_(self.scratchpad_embedding.weight, std=init_std)
            # Same reason, these go last. With T hidden from the block the
            # selector is the only path from T to the output, so it starts
            # random rather than zeroed. Zero init would leave the model
            # T-blind for the first steps of a budget only ~200 steps long.
            if USE_LATENT_HYPOTHESES:
                self.exit_norm = RMSNorm(D_MODEL)
                self.exit_head = nn.Linear(D_MODEL, self.max_loops)
                nn.init.normal_(self.exit_head.weight, std=init_std)
                nn.init.zeros_(self.exit_head.bias)
                # Monotone exit parameterisation. A free per-exit head lets the
                # model pick any T -> exit permutation, and it picks an
                # anti-monotone one, which is only expressible if depth is an
                # index rather than iterated computation. Forcing the exit
                # centre to be stride * scalar(T) makes depth grow with T, so
                # the only way to serve every T is a genuine per-step operator.
                self.t_scalar_head = nn.Linear(D_MODEL, 1)
                nn.init.normal_(self.t_scalar_head.weight, std=init_std)
                # Bias the centre mid-ladder. At zero bias every row starts at
                # exit 0, which is where the collapse lives and where the
                # loop-0 readout predates any recurrence at all.
                nn.init.constant_(self.t_scalar_head.bias, EXIT_CENTRE_INIT)
                # A free scalar head is not enough: it simply learns to
                # DECREASE in T, reproducing the anti-monotone map. Order has
                # to come from the digit reader. Values are a cumulative sum of
                # positive increments over the digit tokens, so the map from
                # digit id to magnitude is increasing by construction, and
                # place weights are positive. Every value stays learned; only
                # the ordering is imposed, the way a positional encoding
                # imposes order without fixing the function.
                self.digit_increments = nn.Parameter(torch.zeros(NUM_DIGITS))
                self.place_increments = nn.Parameter(torch.zeros(MAX_T_DIGITS))
                self.exit_centre = nn.Parameter(torch.tensor(EXIT_CENTRE_INIT))
                self.exit_stride = nn.Parameter(torch.tensor(EXIT_STRIDE_INIT))
                self.exit_width = nn.Parameter(torch.tensor(EXIT_WIDTH_INIT))
                self.retokenize_gate = nn.Parameter(
                    torch.full((), RETOKENIZE_GATE_INIT)
                )
            # Seeds the workspace where the T digits are withheld.
            self.blank_token = nn.Parameter(torch.empty(D_MODEL))
            nn.init.normal_(self.blank_token, std=init_std)
            if USE_ANSWER_QUERIES:
                self.answer_query = nn.Embedding(MAX_ANSWER_DIGITS, D_MODEL)
                nn.init.normal_(self.answer_query.weight, std=init_std)
            if USE_ACTION_HISTORY:
                self.history_norm = RMSNorm(D_MODEL)
                self.history_down = nn.Linear(D_MODEL, HISTORY_RANK, bias=False)
                self.history_up = nn.Linear(HISTORY_RANK, D_MODEL, bias=False)
                self.history_where = nn.Parameter(torch.empty(D_MODEL))
                self.history_gate = nn.Parameter(
                    torch.full((), HISTORY_GATE_INIT)
                )
                nn.init.normal_(self.history_down.weight, std=init_std)
                # Zero so the signal starts silent and the gate opens it.
                nn.init.zeros_(self.history_up.weight)
                nn.init.normal_(self.history_where, std=init_std)

        if DBUG:
            self._debug_initial_parameter_buffers = {}
            with torch.no_grad():
                for index, (name, parameter) in enumerate(self.named_parameters()):
                    buffer_name = f"_debug_initial_parameter_{index}"
                    self.register_buffer(
                        buffer_name,
                        parameter.detach().clone(),
                        persistent=False,
                    )
                    self._debug_initial_parameter_buffers[name] = buffer_name

    def active_loops(self) -> int:
        if self.use_act:
            return self.max_loops
        return TRAIN_LOOPS if self.training else EVAL_LOOPS

    def _history_signal(self, delta: Tensor) -> Tensor:
        # What changed, low rank over the previous loop's update.
        what = self.history_up(self.history_down(self.history_norm(delta)))
        # Where it changed, MAIN's landmark channel for recently touched cells.
        change_rms = delta.pow(2).mean(-1, keepdim=True).sqrt()
        where = change_rms * self.history_where.view(1, 1, -1)
        return torch.sigmoid(self.history_gate) * (what + where)

    def _ordered_t_value(
        self,
        input_ids: Tensor,
        segment_ids: Tensor,
        valid_tokens: Tensor,
    ) -> Tensor:
        """A learned, order-respecting magnitude for the T digits.

        digit_value is increasing in the digit token id and place weights are
        positive, so the result is monotone in T without any value being
        hard-coded.
        """
        digit_values = torch.cumsum(F.softplus(self.digit_increments), dim=0)
        place_weights = torch.cumsum(F.softplus(self.place_increments), dim=0)

        is_digit = input_ids.ge(DIGIT_OFFSET) & valid_tokens
        t_mask = segment_ids.eq(T_SEGMENT) & is_digit
        digit_index = (input_ids - DIGIT_OFFSET).clamp(0, NUM_DIGITS - 1)
        value_at = digit_values[digit_index] * t_mask.to(digit_values.dtype)

        # Rank positions from the right within the T segment so the last digit
        # is the least significant, then weight by the positive place ladder.
        flipped = t_mask.flip(-1)
        rank_from_right = (flipped.cumsum(-1) - 1).clamp(0, MAX_T_DIGITS - 1).flip(-1)
        weights = place_weights[rank_from_right] * t_mask.to(place_weights.dtype)
        return (value_at * weights).sum(dim=-1)

    def _retokenize(self, workspace: Tensor) -> Tensor:
        normed = self.final_norm(workspace)
        probabilities = F.softmax(
            self.head(normed) / RETOKENIZE_TEMPERATURE,
            dim=-1,
        )
        if RETOKENIZE_STRAIGHT_THROUGH:
            hard = F.one_hot(
                probabilities.argmax(-1),
                probabilities.shape[-1],
            ).to(probabilities.dtype)
            probabilities = hard + probabilities - probabilities.detach()
        # head.weight is token_embedding.weight, so this decodes and re-encodes
        # through the same tied matrix.
        token_view = probabilities @ self.token_embedding.weight
        # Embeddings are std 0.02 and the residual stream is not, so match RMS
        # to make this a direction blend rather than an amplitude collapse.
        target_rms = workspace.pow(2).mean(-1, keepdim=True).sqrt()
        token_rms = token_view.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        token_view = token_view * (target_rms / token_rms)
        gate = torch.sigmoid(self.retokenize_gate)
        return workspace + gate * (token_view - workspace)

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
            if layer_diagnostics is not None or training_stage_states is not None:
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
        prompt_key_mask = None
        workspace_seed = prompt_memory
        if HIDE_T_FROM_BLOCK and segment_ids is not None:
            t_positions = segment_ids.eq(T_SEGMENT)
            if attention_mask is not None and attention_mask.shape == segment_ids.shape:
                valid = attention_mask.bool()
            else:
                valid = torch.ones_like(t_positions)
            prompt_key_mask = valid & ~t_positions
            # Blank only the seed. These slots stay readable and writable
            # because the answer is read out on top of them.
            workspace_seed = torch.where(
                t_positions[..., None],
                self.blank_token.view(1, 1, -1) + position_signal,
                prompt_memory,
            )
        if USE_ANSWER_QUERIES and attention_mask is not None:
            # The answer is read from the last few valid prompt positions. Seed
            # those workspace slots from dedicated learned queries rather than
            # from prompt content. Measured on x*y mod 323 with 74k pairs, a
            # probe reading the answer off input positions reached 0.085
            # held-out exact while the same model with dedicated answer slots
            # reached 0.812. The prompt stream stays readable, so nothing is
            # lost by not seeding these slots from it.
            lengths = attention_mask.bool().sum(dim=1)
            offsets = torch.arange(MAX_ANSWER_DIGITS, device=lengths.device)
            slots = (
                lengths[:, None] - MAX_ANSWER_DIGITS + offsets[None, :]
            ).clamp_min(0)
            queries = self.answer_query.weight.unsqueeze(0).expand(
                batch_size, -1, -1
            ) + position_signal[slots]
            workspace_seed = workspace_seed.scatter(
                1,
                slots[..., None].expand(-1, -1, workspace_seed.shape[-1]),
                queries.to(workspace_seed.dtype),
            )
        workspace_state = workspace_seed + self.workspace_token.view(1, 1, -1)
        scratchpad_state = self.scratchpad_embedding.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        work_state = torch.cat(
            (control_state, workspace_state, scratchpad_state),
            dim=1,
        )
        loop_kwargs = {
            "num_loops": self.active_loops(),
            "retokenize": (
                self._retokenize if USE_LATENT_HYPOTHESES else None
            ),
            "history": self._history_signal if USE_ACTION_HISTORY else None,
            "prompt_key_mask": prompt_key_mask,
        }
        if (
            layer_diagnostics is None
            and training_stage_states is None
            and hypothesis_states is None
        ):
            work_state = self.processor(
                prompt_memory,
                work_state,
                attention_mask,
                **loop_kwargs,
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
                **loop_kwargs,
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
                "relative_delta": (delta_norm / initial_norm.clamp_min(1e-12)).detach(),
            }
        return stats

    def _debug_segment_embedding_stats(self) -> dict[str, Tensor]:
        buffer_name = self._debug_initial_parameter_buffers["segment_embedding.weight"]
        initial = getattr(self, buffer_name).detach().float()
        current = self.segment_embedding.weight.detach().float()
        initial_norms = initial.norm(dim=-1)
        current_norms = current.norm(dim=-1)
        delta_norms = (current - initial).norm(dim=-1)
        cosine_denominator = current_norms * initial_norms
        initial_cosines = torch.where(
            cosine_denominator > 0,
            (current * initial).sum(dim=-1) / cosine_denominator.clamp_min(1e-12),
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
            "relative_deltas": (delta_norms / initial_norms.clamp_min(1e-12)).detach(),
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
                state_gradient = state_gradient[:, 1 : 1 + valid_tokens.shape[1]]
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
                stage["state_grad_rms"] / final_gradient_rms.clamp_min(1e-12)
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
        # Pass the real segment_ids so these probes vary only the segment
        # embedding. Dropping them would also lift the T mask and confound the
        # comparison against the normal forward.
        zero_segment_x, _, _ = self._run_processor(
            base_token_state,
            position_signal,
            attention_mask,
            segment_ids=segment_ids,
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
                segment_ids=segment_ids,
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
        collect_act_diagnostics = self.collect_act_diagnostics if DBUG else False
        collect_model_diagnostics = self.collect_model_diagnostics if DBUG else False
        layer_diagnostics = [] if collect_model_diagnostics else None
        stage_states = [] if collect_model_diagnostics else None
        training_stage_states = [] if collect_training_diagnostics else None
        hypothesis_states = [] if USE_LATENT_HYPOTHESES and not self.use_act else None
        x, ponder_cost, act_diagnostics = self._run_processor(
            token_state,
            position_signal,
            attention_mask,
            segment_ids=segment_ids,
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
            active_loops = self.active_loops()
            if len(hypothesis_states) != active_loops:
                raise RuntimeError("processor did not produce every hypothesis")
            hypothesis_logits = torch.stack(
                [self.head(self.final_norm(state)) for state in hypothesis_states],
                dim=1,
            )
            # Exit depth has to vary with the prompt's T. Pool the T segment
            # only, where position_signal keeps multi-digit T ordered.
            prompt_features = token_state + position_signal
            t_weights = (segment_ids.eq(T_SEGMENT) & valid_tokens).to(
                prompt_features.dtype
            )
            t_summary = (prompt_features * t_weights[..., None]).sum(dim=1) / (
                t_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            normed_t = self.exit_norm(t_summary)
            if EXIT_MONOTONE_IN_T:
                # Exit centre = stride * scalar(T), so depth is forced to grow
                # with T instead of being a free index the model can permute.
                # Everything here is learned: the scalar is read from the T
                # tokens, the stride and width are parameters. Only the
                # monotone shape is imposed, the way a positional encoding
                # imposes order without fixing the function.
                if EXIT_ORDERED_DIGITS:
                    t_scalar = self._ordered_t_value(input_ids, segment_ids, valid_tokens)
                else:
                    t_scalar = self.t_scalar_head(normed_t).squeeze(-1)
                centre = self.exit_stride.abs() * t_scalar + self.exit_centre
                positions = torch.arange(
                    active_loops,
                    device=t_scalar.device,
                    dtype=t_scalar.dtype,
                )
                width = self.exit_width.abs().clamp_min(1e-2)
                exit_logits = -(
                    (positions[None, :] - centre[:, None]) ** 2
                ) / (2.0 * width * width)
                exit_logits = exit_logits + self.hypothesis_prior[:active_loops]
            else:
                exit_logits = (
                    self.exit_head(normed_t) + self.hypothesis_prior
                )[:, :active_loops]
            hypothesis_log_prior = F.log_softmax(exit_logits, dim=-1)
            selected = hypothesis_log_prior.argmax(dim=-1)
            gather_index = selected[:, None, None, None]
            logits = hypothesis_logits.gather(
                1,
                gather_index.expand(-1, 1, *hypothesis_logits.shape[2:]),
            ).squeeze(1)
            if DBUG and collect_model_diagnostics:
                # x only feeds the diagnostic pass, so skip stacking every
                # retained state when it is not going to be read.
                x = torch.stack(hypothesis_states, dim=1).gather(
                    1,
                    selected[:, None, None, None].expand(
                        -1, 1, *hypothesis_states[0].shape[1:]
                    ),
                ).squeeze(1)
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


# build_model runs at the start of each seed, so this tracks the evaluator's
# own training deadline closely enough to anneal against.
_SEED_STARTED_AT: float | None = None


def _construction_device() -> torch.device:
    # Construction is charged to the training budget. CPU init plus the copy
    # costs ~4.8s of the 60s Easy budget at this width.
    if BUILD_ON_ACCELERATOR and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(spec: ModelSpec) -> Model:
    global _SEED_STARTED_AT
    _SEED_STARTED_AT = time.monotonic()
    with torch.device(_construction_device()):
        model = Model(spec, use_act=USE_ACT)
    state_elements = assert_model_state(model, spec)
    if not PRINT_LOGS:
        return model

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    ceiling = spec.maximum_model_state_elements
    print(
        f"MODEL | state {state_elements:,} of {ceiling:,} "
        f"({100.0 * state_elements / ceiling:.2f}%), "
        f"headroom {ceiling - state_elements:,} | trainable {trainable:,} | "
        f"vocab {spec.vocab_size} max_seq_len {spec.max_seq_len} | "
        f"built on {_construction_device()}",
        flush=True,
    )
    print(
        "CONSTANTS |"
        f" D_MODEL: {D_MODEL}, NUM_HEADS: {NUM_HEADS}, D_FF: {D_FF}, "
        f"PONDER_WEIGHT: {PONDER_WEIGHT}, USE_ACT: {USE_ACT}, "
        f"TRAIN_LOOPS: {TRAIN_LOOPS}, EVAL_LOOPS: {EVAL_LOOPS}, "
        f"ACT_MAX_LOOPS: {ACT_MAX_LOOPS}, "
        f"ACT_TAIL_HALT_FRACTION: {ACT_TAIL_HALT_FRACTION}, "
        f"USE_MUON: {USE_MUON}, MUON_LR: {MUON_LR}, "
        f"MUON_MOMENTUM: {MUON_MOMENTUM}, "
        f"MUON_WEIGHT_DECAY: {MUON_WEIGHT_DECAY}, "
        f"MUON_ADJUST_LR_FN: {MUON_ADJUST_LR_FN}, "
        f"LR_DECAY_HOLD_FRACTION: {LR_DECAY_HOLD_FRACTION}, "
        f"LR_DECAY_MIN_FACTOR: {LR_DECAY_MIN_FACTOR}, "
        f"USE_LATENT_HYPOTHESES: {USE_LATENT_HYPOTHESES}, "
        f"HYPOTHESIS_TEMPERATURE: {HYPOTHESIS_TEMPERATURE}, "
        f"SELECTOR_LOSS_WEIGHT: {SELECTOR_LOSS_WEIGHT}, "
            f"EXIT_MI_WEIGHT: {EXIT_MI_WEIGHT}, "
            f"EXIT_MONOTONE_IN_T: {EXIT_MONOTONE_IN_T}, "
            f"EXIT_ORDERED_DIGITS: {EXIT_ORDERED_DIGITS}, "
            f"POSTERIOR_USES_PRIOR: {POSTERIOR_USES_PRIOR}, "
        f"HIDE_T_FROM_BLOCK: {HIDE_T_FROM_BLOCK}, "
        f"USE_ANSWER_QUERIES: {USE_ANSWER_QUERIES}, "
        f"USE_ACTION_HISTORY: {USE_ACTION_HISTORY}, "
        f"HISTORY_RANK: {HISTORY_RANK}, "
        f"HISTORY_GATE_INIT: {HISTORY_GATE_INIT}, "
        f"NUM_SCRATCHPAD_TOKENS: {NUM_SCRATCHPAD_TOKENS}, "
        f"RETOKENIZE_TEMPERATURE: {RETOKENIZE_TEMPERATURE}, "
        f"RETOKENIZE_GATE_INIT: {RETOKENIZE_GATE_INIT}, "
        f"RETOKENIZE_STRAIGHT_THROUGH: {RETOKENIZE_STRAIGHT_THROUGH}, "
        f"GATE_WEIGHT_DECAY: {GATE_WEIGHT_DECAY}, "
        f"ADAMW_WEIGHT_DECAY: {ADAMW_WEIGHT_DECAY}, "
        f"TRAIN_BATCH_SIZE: {TRAIN_BATCH_SIZE}, "
        f"EVAL_BATCH_SIZE: {EVAL_BATCH_SIZE}, "
        f"SAM_RHO: {SAM_RHO}, "
        f"BUILD_ON_ACCELERATOR: {BUILD_ON_ACCELERATOR}, "
        f"PRINT_LOGS: {PRINT_LOGS}, DBUG: {DBUG}",
        flush=True,
    )
    return model


class CombinedOptimizer:
    def __init__(
        self,
        optimizers: list[torch.optim.Optimizer],
        restore: tuple[list[Tensor], list[Tensor | None]] | None = None,
    ) -> None:
        self.optimizers = optimizers
        self.restore = restore

    @property
    def param_groups(self) -> list[dict]:
        return [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        if self.restore is not None:
            # Undo the SAM perturbation first, so the gradient measured at the
            # worsened point is applied at the original one.
            parameters, perturbation = self.restore
            with torch.no_grad():
                for parameter, epsilon in zip(parameters, perturbation):
                    if epsilon is not None:
                        parameter.sub_(epsilon)
            perturbation[:] = [None] * len(perturbation)
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
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}


class CosineDecayScheduler:
    """Anneal every parameter group across the evaluator's wall-clock budget."""

    def __init__(
        self,
        optimizer,
        *,
        total_seconds: float,
        hold_fraction: float,
        min_factor: float,
        started_at: float | None = None,
        clock=time.monotonic,
    ) -> None:
        if total_seconds <= 0.0:
            raise ValueError("training budget must be positive")
        if not 0.0 <= hold_fraction < 1.0:
            raise ValueError("hold fraction must be in [0, 1)")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError("minimum learning-rate factor must be in [0, 1]")

        self.optimizer = optimizer
        self.total_seconds = float(total_seconds)
        self.hold_fraction = float(hold_fraction)
        self.min_factor = float(min_factor)
        self.clock = clock
        self.started_at = clock() if started_at is None else float(started_at)
        self.completed_steps = 0
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def factor(self) -> float:
        elapsed_fraction = (self.clock() - self.started_at) / self.total_seconds
        elapsed_fraction = min(max(elapsed_fraction, 0.0), 1.0)
        if elapsed_fraction <= self.hold_fraction:
            return 1.0
        decay_progress = (elapsed_fraction - self.hold_fraction) / (
            1.0 - self.hold_fraction
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine

    def step(self) -> None:
        self.completed_steps += 1
        factor = self.factor()
        for group, base_lr in zip(
            self.optimizer.param_groups,
            self.base_lrs,
            strict=True,
        ):
            group["lr"] = base_lr * factor


def _build_scheduler(optimizer, spec: OptimizerSpec) -> CosineDecayScheduler:
    return CosineDecayScheduler(
        optimizer,
        total_seconds=spec.training_time_seconds,
        hold_fraction=LR_DECAY_HOLD_FRACTION,
        min_factor=LR_DECAY_MIN_FACTOR,
        started_at=_SEED_STARTED_AT,
    )


def _adamw_groups(named_parameters) -> list[dict]:
    decayed = []
    gates = []
    for name, parameter in named_parameters:
        bucket = gates if name in GATE_PARAMETER_NAMES else decayed
        bucket.append(parameter)
    groups = [{"params": decayed, "weight_decay": ADAMW_WEIGHT_DECAY}]
    if gates:
        groups.append({"params": gates, "weight_decay": GATE_WEIGHT_DECAY})
    return groups


def _sam_bundle(
    optimizers: list[torch.optim.Optimizer],
    model: nn.Module,
    spec: OptimizerSpec,
) -> OptimizerBundle:
    """Wrap the optimizers, adding the SAM perturb/restore pair when enabled."""

    if SAM_RHO <= 0.0:
        optimizer = CombinedOptimizer(optimizers)
        return OptimizerBundle(optimizer, scheduler=_build_scheduler(optimizer, spec))

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    perturbation: list[Tensor | None] = [None] * len(parameters)

    def perturb(context: BackwardPassContext) -> None:
        # The evaluator only calls this before the final pass, and already
        # under no_grad. Gradients here are post-clip, so rho is measured
        # against a clipped norm.
        total_square = torch.zeros((), device=parameters[0].device)
        for parameter in parameters:
            if parameter.grad is not None:
                total_square = total_square + parameter.grad.pow(2).sum()
        scale = SAM_RHO / total_square.sqrt().clamp_min(1e-12)
        for index, parameter in enumerate(parameters):
            if parameter.grad is None:
                perturbation[index] = None
                continue
            step = parameter.grad * scale
            perturbation[index] = step
            parameter.add_(step)

    optimizer = CombinedOptimizer(optimizers, restore=(parameters, perturbation))
    return OptimizerBundle(
        optimizer,
        scheduler=_build_scheduler(optimizer, spec),
        backward_passes_per_step=2,
        between_backward_passes=perturb,
    )


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    if USE_MUON:
        muon_parameters = []
        adamw_named = []
        for name, parameter in model.named_parameters():
            use_muon = (
                parameter.ndim == 2
                and name.startswith("processor.block.")
                and name.endswith(".weight")
            )
            if use_muon:
                muon_parameters.append(parameter)
            else:
                adamw_named.append((name, parameter))

        muon = torch.optim.Muon(
            muon_parameters,
            lr=MUON_LR,
            momentum=MUON_MOMENTUM,
            weight_decay=MUON_WEIGHT_DECAY,
            adjust_lr_fn=MUON_ADJUST_LR_FN,
        )
        adamw = torch.optim.AdamW(
            _adamw_groups(adamw_named),
            lr=1e-3,
            betas=(0.9, 0.95),
            capturable=spec.device_type == "cuda",
        )
        return _sam_bundle([muon, adamw], model, spec)

    adamw = torch.optim.AdamW(
        _adamw_groups(model.named_parameters()),
        lr=1e-3,
        betas=(0.9, 0.95),
        capturable=spec.device_type == "cuda",
    )
    return _sam_bundle([adamw], model, spec)


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    token_training_loss=token_training_loss,
    batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
)
