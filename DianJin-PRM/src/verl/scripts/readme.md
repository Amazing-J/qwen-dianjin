# rule-based
python -m legacy_model_merger merge --backend fsdp --local_dir /data/yuanchen/fin/ReasonFlux_PRM/Application/examples/grpo_trainer/checkpoints/verl_grpo_dapo/qwen_grpo_rulebased_dapo/global_step_150/actor --target_dir /data/yuanchen/fin/ReasonFlux_PRM/Application/examples/grpo_trainer/cp_res/rule-based-156

# prm
python -m legacy_model_merger merge --backend fsdp --local_dir /data/yuanchen/fin/ReasonFlux_PRM/Application/examples/grpo_trainer/checkpoints/verl_grpo_dapo/qwen_grpo_fin_prm_dapo/global_step_156/actor  --target_dir /data/yuanchen/fin/ReasonFlux_PRM/Application/examples/grpo_trainer/cp_res/fin-prm-156