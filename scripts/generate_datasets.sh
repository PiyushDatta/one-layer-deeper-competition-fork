#!/usr/bin/env bash
set -euo pipefail

# Squaring-mod difficulty ladder.
#
# Every tier uses separate prompt and output tensors, so models can attend
# bidirectionally over the complete prompt. Easy and medium use prompt-level IID
# splits. Every (N, x, T) prompt is still unique; these tiers intentionally
# measure interpolation over problem families seen in training.
# Runtime and score calibration must be measured against this separate-output
# representation; results from the former causal-LM datasets are not comparable.

# ---------------------------------------------------------------------------
# Easy: five datasets using the one-minute training budget.
# ---------------------------------------------------------------------------

# E1: tiny fixed N with three ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123 \
  --fixed_p 17 --fixed_q 19 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 250 --ood_examples_per_setting 100 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E2: larger fixed N and geometric ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_899_t124 \
  --fixed_p 29 --fixed_q 31 \
  --time_steps '[1,2,4]' --ood_time_steps '[7]' \
  --examples_per_setting 800 --ood_examples_per_setting 300 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E3: sampled N at fixed T over two small, exactly auditable bit cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1011_t2 \
  --modulus_bits '[10,11]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 2000 --ood_examples_per_setting 400 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E4: one bit harder than E3 with twice the per-cell row budget.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1112_t2 \
  --modulus_bits '[11,12]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 4000 --ood_examples_per_setting 600 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E5: joint N/T conditioning at small scale.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123 \
  --modulus_bits '[10,11]' \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 1000 --ood_examples_per_setting 300 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# ---------------------------------------------------------------------------
# Medium: five datasets using the ten-minute training budget.
# ---------------------------------------------------------------------------

# M1: 14-bit fixed N with a geometric T schedule.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_10403_t4816 \
  --fixed_p 101 --fixed_q 103 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 10000 --ood_examples_per_setting 3000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M2: 16-bit fixed N and a 95k-row complete dataset.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_38021_t4816 \
  --fixed_p 193 --fixed_q 197 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 30000 --ood_examples_per_setting 5000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M3: sampled N, fixed T, spanning 11-15 bits.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b111315_t2 \
  --modulus_bits '[11,13,15]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 8000 --ood_examples_per_setting 1000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M4: sampled N, fixed T, with larger 14-22 bit moduli.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b141822_t8 \
  --modulus_bits '[14,18,22]' --fixed_time_steps 8 \
  --ood_time_steps '[16]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M5: joint N/T conditioning across nine balanced ID cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248 \
  --modulus_bits '[12,14,16]' \
  --time_steps '[2,4,8]' --ood_time_steps '[16]' \
  --examples_per_setting 10000 --ood_examples_per_setting 1000 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true
