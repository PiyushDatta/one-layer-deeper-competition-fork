from __future__ import annotations

import torch
import torch.nn.functional as F
from benchmark import (
    assert_model_state,
    ModelSpec,
    OptimizerBundle,
    OptimizerSpec,
    Submission,
)
from torch import nn, Tensor

D_MODEL = 128
NUM_HEADS = 4
PONDER_WEIGHT = 0.005
USE_ACT = True
FIXED_LOOPS = 16
ACT_MAX_LOOPS = 16
ACT_TAIL_HALT_FRACTION = None
DBUG = False

if DBUG:
    print(
        "Constants\n"
        f" D_MODEL: {D_MODEL}, NUM_HEADS: {NUM_HEADS}, "
        f"PONDER_WEIGHT: {PONDER_WEIGHT}, USE_ACT: {USE_ACT}, "
        f"FIXED_LOOPS: {FIXED_LOOPS}, ACT_MAX_LOOPS: {ACT_MAX_LOOPS}, "
        f"ACT_TAIL_HALT_FRACTION: {ACT_TAIL_HALT_FRACTION}, DBUG: {DBUG}"
    )


def training_loss(
    logits: Tensor,
    labels: Tensor,
    auxiliary: object,
) -> Tensor:
    task_loss = F.cross_entropy(logits, labels)
    ponder_cost = auxiliary["ponder_cost"] if DBUG else auxiliary
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
        self.up = nn.Linear(D_MODEL, 4 * D_MODEL)
        self.down = nn.Linear(4 * D_MODEL, D_MODEL)

    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
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
        x = residual + self.out(x)
        return x + self.down(F.gelu(self.up(self.mixer_norm(x))))


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
    ) -> tuple[Tensor, Tensor, dict[str, object] | None]:
        if self.use_act:
            return self.forward_act(
                x,
                position_signal,
                attention_mask,
                collect_act_diagnostics=collect_act_diagnostics,
            )
        return self.forward_fixed(x, position_signal, attention_mask)

    def forward_act(
        self,
        x: Tensor,
        position_signal: Tensor,
        attention_mask: Tensor | None = None,
        *,
        collect_act_diagnostics: bool = False,
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

        for step in range(self.max_loops):
            was_running = valid_tokens & (curr_halt_prob < threshold)
            step_signal = position_signal + self.time_embedding.weight[step]
            block_input = torch.where(was_running, x + step_signal, x)
            candidate_x = self.block(block_input, attention_mask)
            x = torch.where(was_running, candidate_x, x)
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
    ) -> tuple[Tensor, Tensor, None]:
        for step in range(self.max_loops):
            x = x + position_signal + self.time_embedding.weight[step]
            x = self.block(x, attention_mask)

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

        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        init_std = 0.02
        nn.init.normal_(self.token_embedding.weight, std=init_std)
        nn.init.normal_(self.position_embedding.weight, std=init_std)

        time_embedding = nn.Embedding(self.max_loops, D_MODEL)
        nn.init.normal_(time_embedding.weight, std=init_std)

        halting_unit = nn.Linear(D_MODEL, 1) if self.use_act else None
        self.processor = UniversalProcessor(
            block,
            time_embedding,
            use_act=self.use_act,
            max_loops=self.max_loops,
            halting_unit=halting_unit,
            tail_halt_fraction=(ACT_TAIL_HALT_FRACTION if self.use_act else None),
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, object]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids)
        position_signal = self.position_embedding(positions)
        collect_act_diagnostics = self.collect_act_diagnostics if DBUG else False
        x, ponder_cost, act_diagnostics = self.processor(
            x,
            position_signal,
            attention_mask,
            collect_act_diagnostics=collect_act_diagnostics,
        )
        logits = self.head(self.final_norm(x))
        if not DBUG:
            return logits, ponder_cost

        if act_diagnostics is not None:
            act_diagnostics["ponder_weight"] = PONDER_WEIGHT
        auxiliary = {
            "ponder_cost": ponder_cost,
            "act": act_diagnostics,
        }
        return logits, auxiliary


def build_model(spec: ModelSpec) -> Model:
    model = Model(spec, use_act=USE_ACT)
    assert_model_state(model, spec)
    return model


def build_optimizer(model: nn.Module, spec: OptimizerSpec) -> OptimizerBundle:
    return OptimizerBundle(
        torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            capturable=spec.device_type == "cuda",
        )
    )


SUBMISSION = Submission(
    build_model=build_model,
    build_optimizer=build_optimizer,
    training_loss=training_loss,
)
