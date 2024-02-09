# This will resume training if data/training_checkpoint is not empty. If you want to start from scratch, delete the directory.

accelerate launch --num_processes 1 --mixed_precision bf16 steve1/train.py \
--in_model downloads/2x.model \
--in_weights downloads/rl-from-foundation-2x.weights \
--out_weights results/steve1/weights/trained_with_script.weights \
--trunc_t 64 \
--T 640 \
--batch_size 4 \
--gradient_accumulation_steps 4 \
--num_workers 4 \
--weight_decay 0.039428 \
--n_frames 100_000_000 \
--learning_rate 4e-5 \
--warmup_frames 10_000_000 \
--p_uncond 0.1 \
--min_btwn_goals 15 \
--max_btwn_goals 200 \
--checkpoint_dir results/steve1/training_checkpoint \
--val_freq 1000 \
--val_freq_begin 100 \
--val_freq_switch_steps 500 \
--val_every_nth 10 \
--save_each_val False \
--sampling seed1 \
--sampling_dir downloads/samplings/ \
--snapshot_every_n_frames 50_000_000 \
--val_every_nth 1
