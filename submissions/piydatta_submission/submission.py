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
PONDER_WEIGHT = 0.001
USE_ACT = True
FIXED_LOOPS = 16
ACT_MAX_LOOPS = 16
print(
    f"Constants\n D_MODEL: {D_MODEL}, NUM_HEADS: {NUM_HEADS}, PONDER_WEIGHT: {PONDER_WEIGHT}, USE_ACT: {USE_ACT}, FIXED_LOOPS: {FIXED_LOOPS}, ACT_MAX_LOOPS: {ACT_MAX_LOOPS}"
)


def training_loss(
    logits: Tensor,
    labels: Tensor,
    ponder_cost: Tensor,
) -> Tensor:
    task_loss = F.cross_entropy(logits, labels)
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


class Model(nn.Module):

    def __init__(self, spec: ModelSpec, use_act: bool = False) -> None:
        super().__init__()
        self.config = Config(spec.vocab_size, spec.max_seq_len)
        self.use_act = use_act
        self.max_loops = ACT_MAX_LOOPS if self.use_act else FIXED_LOOPS
        self.halting_prob_threshold = 0.01

        self.token_embedding = nn.Embedding(spec.vocab_size, D_MODEL)
        self.position_embedding = nn.Embedding(spec.max_seq_len, D_MODEL)
        self.block = Block()
        self.final_norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, spec.vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        init_std = 0.02
        nn.init.normal_(self.token_embedding.weight, std=init_std)
        nn.init.normal_(self.position_embedding.weight, std=init_std)

        self.time_embedding = nn.Embedding(self.max_loops, D_MODEL)
        nn.init.normal_(self.time_embedding.weight, std=init_std)

        if self.use_act:
            self.halting_unit = nn.Linear(D_MODEL, 1)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        for ACT:
            determine which tokens were running
            construct block input:
                active tokens receive position/time signal
                halted tokens preserve their state
            calculate candidate states
            accept candidates only for active tokens
            calculate h
            determine newly halted tokens
            update counts and remainders
            update weighted output
        """
        if self.use_act:
            return self.forward_act(input_ids, attention_mask)
        return self.forward_fixed(input_ids, attention_mask)

    def forward_act(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        for ACT:
            determine which tokens were running
            construct block input:
                active tokens receive position/time signal
                halted tokens preserve their state
            calculate candidate states
            accept candidates only for active tokens
            calculate h
            determine newly halted tokens
            update counts and remainders
            update weighted output
        """
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids)
        pos_signal = self.position_embedding(positions)
        batch_size, seq_len, _ = x.shape
        curr_halt_prob = x.new_zeros(batch_size, seq_len, 1)
        update_counts = torch.zeros_like(curr_halt_prob)
        remainders = torch.zeros_like(curr_halt_prob)
        weighted_output = torch.zeros_like(x)
        threshold = 1.0 - self.halting_prob_threshold
        if attention_mask is None:
            valid_tokens = torch.ones_like(curr_halt_prob, dtype=torch.bool)
        else:
            valid_tokens = attention_mask.unsqueeze(-1).bool()

        for step in range(self.max_loops):
            was_running = valid_tokens & (curr_halt_prob < threshold)
            step_signal = pos_signal + self.time_embedding.weight[step]
            block_input = torch.where(was_running, x + step_signal, x)
            candidate_x = self.block(block_input, attention_mask)
            x = torch.where(was_running, candidate_x, x)
            halting_logit = self.halting_unit(x)
            h = torch.sigmoid(halting_logit)
            if step == self.max_loops - 1:
                newly_halted = was_running
            else:
                newly_halted = was_running & (curr_halt_prob + h >= threshold)
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
        return self.head(self.final_norm(weighted_output)), ponder_cost

    def forward_fixed(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids)
        pos_signal = self.position_embedding(positions)

        for step in range(self.max_loops):
            x = x + pos_signal + self.time_embedding.weight[step]
            x = self.block(x, attention_mask)

        ponder_cost = x.new_zeros(())
        return self.head(self.final_norm(x)), ponder_cost


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
