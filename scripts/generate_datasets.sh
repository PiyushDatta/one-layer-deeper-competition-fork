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
# Easy: ten datasets using the one-minute training budget.
# ---------------------------------------------------------------------------

# E1: tiny fixed N with three ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_323_t123 \
  --fixed_p 17 --fixed_q 19 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 250 --ood_examples_per_setting 100 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[10,11]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E2: larger fixed N and geometric ID depths.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_n_899_t124 \
  --fixed_p 29 --fixed_q 31 \
  --time_steps '[1,2,4]' --ood_time_steps '[7]' \
  --examples_per_setting 800 --ood_examples_per_setting 300 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[11,12]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E3: sampled N at fixed T over two small, exactly auditable bit cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1011_t2 \
  --modulus_bits '[10,11]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 2000 --ood_examples_per_setting 400 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E4: one bit harder than E3 with twice the per-cell row budget.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_fixed_t_b1112_t2 \
  --modulus_bits '[11,12]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 4000 --ood_examples_per_setting 600 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[13,14]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E5: joint N/T conditioning at small scale.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_easy_bidirectional_variable_b1011_t123 \
  --modulus_bits '[10,11]' \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 1000 --ood_examples_per_setting 300 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E6: top-signal rung with one additional supervised depth.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_easy_e6_bidirectional_fixed_n_247_t1234 \
  --fixed_p 13 --fixed_q 19 \
  --time_steps '[1,2,3,4]' --ood_time_steps '[5]' \
  --examples_per_setting 184 --ood_examples_per_setting 60 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 32 \
  --ood_n_depth_evaluation_modulus_bits '[9,10]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E7: same shallow training support with a nearby T=4 holdout.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_easy_e7_bidirectional_fixed_n_287_t123_ood4 \
  --fixed_p 7 --fixed_q 41 \
  --time_steps '[1,2,3]' --ood_time_steps '[4]' \
  --examples_per_setting 220 --ood_examples_per_setting 85 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[10,11]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E8: same N as E7, with the harder T=6 holdout.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_easy_e8_bidirectional_fixed_n_287_t123 \
  --fixed_p 7 --fixed_q 41 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 220 --ood_examples_per_setting 85 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[10,11]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E9: larger fixed modulus with the T=6 holdout.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_easy_e9_bidirectional_fixed_n_299_t123 \
  --fixed_p 13 --fixed_q 23 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 250 --ood_examples_per_setting 90 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[10,11]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# E10: final additive rung before the original Easy tail.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_easy_e10_bidirectional_fixed_n_403_t123 \
  --fixed_p 13 --fixed_q 31 \
  --time_steps '[1,2,3]' --ood_time_steps '[6]' \
  --examples_per_setting 330 --ood_examples_per_setting 125 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[10,11]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.8 --test_fraction 0.2 \
  --split_group prompt --seed 45 --separate_input_output true

# ---------------------------------------------------------------------------
# Medium: ten datasets using the ten-minute training budget.
# ---------------------------------------------------------------------------

# M1: 14-bit fixed N with a geometric T schedule.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_10403_t4816 \
  --fixed_p 101 --fixed_q 103 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 10000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 192 \
  --ood_n_depth_evaluation_modulus_bits '[15,16]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M2: 16-bit fixed N and a 95k-row complete dataset.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_n_38021_t4816 \
  --fixed_p 193 --fixed_q 197 \
  --time_steps '[4,8,16]' --ood_time_steps '[32]' \
  --examples_per_setting 30000 --ood_examples_per_setting 5000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 768 \
  --ood_n_depth_evaluation_modulus_bits '[17,18]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M3: sampled N, fixed T, spanning 11-15 bits.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b111315_t2 \
  --modulus_bits '[11,13,15]' --fixed_time_steps 2 \
  --ood_time_steps '[4]' \
  --examples_per_setting 8000 --ood_examples_per_setting 1000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[12,14,16]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M4: sampled N, fixed T, with larger 14-22 bit moduli.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_fixed_t_b141822_t8 \
  --modulus_bits '[14,18,22]' --fixed_time_steps 8 \
  --ood_time_steps '[16]' \
  --examples_per_setting 30000 --ood_examples_per_setting 3000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[16,20,24]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M5: joint N/T conditioning across nine balanced ID cells.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_new11_medium_bidirectional_variable_b121416_t248 \
  --modulus_bits '[12,14,16]' \
  --time_steps '[2,4,8]' --ood_time_steps '[16]' \
  --examples_per_setting 10000 --ood_examples_per_setting 1000 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_examples_per_setting 256 \
  --ood_n_depth_evaluation_modulus_bits '[13,15,18]' \
  --ood_n_depth_evaluation_examples_per_setting 256 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M6: easiest new Medium rung; both references remain well below Easy E1.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_medium_m6_bidirectional_fixed_n_1517_t124 \
  --fixed_p 37 --fixed_q 41 \
  --time_steps '[1,2,4]' --ood_time_steps '[8]' \
  --examples_per_setting 1300 --ood_examples_per_setting 500 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M7: second rung, lower on both reference submissions.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_medium_m7_bidirectional_fixed_n_1763_t124 \
  --fixed_p 41 --fixed_q 43 \
  --time_steps '[1,2,4]' --ood_time_steps '[8]' \
  --examples_per_setting 1500 --ood_examples_per_setting 600 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M8: third rung, selected from the broad semiprime probe grid.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_medium_m8_bidirectional_fixed_n_1333_t124 \
  --fixed_p 31 --fixed_q 43 \
  --time_steps '[1,2,4]' --ood_time_steps '[8]' \
  --examples_per_setting 1100 --ood_examples_per_setting 425 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M9: fourth rung, with lower scores from both reference submissions.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_medium_m9_bidirectional_fixed_n_1927_t124 \
  --fixed_p 41 --fixed_q 47 \
  --time_steps '[1,2,4]' --ood_time_steps '[8]' \
  --examples_per_setting 1550 --ood_examples_per_setting 600 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true

# M10: hardest new rung before the original near-zero Medium datasets.
python -m data.squaring_mod \
  --output_dir data/generated/squaring_mod_granular_medium_m10_bidirectional_fixed_n_1739_t124 \
  --fixed_p 37 --fixed_q 47 \
  --time_steps '[1,2,4]' --ood_time_steps '[8]' \
  --examples_per_setting 1400 --ood_examples_per_setting 550 \
  --depth_evaluation_time_steps '[1,2,4,8,16,32,64]' \
  --depth_evaluation_exhaustive_x true \
  --ood_n_depth_evaluation_modulus_bits '[12,13]' \
  --ood_n_depth_evaluation_examples_per_setting 128 \
  --train_fraction 0.9 --test_fraction 0.1 \
  --split_group prompt --seed 45 --separate_input_output true
