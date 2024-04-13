from llmtuner import run_exp


def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":

    main()


    # sys.argv = [
    #     '--stage sft',
    #     '--do_train yes',
    #     '--model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    #     '--dataset alpaca_gpt4_zh',
    #     '--template zephyr', # 和zephyr使用同样模板
    #     '--finetuning_type lora',
    #     '--lora_target q_proj,v_proj',
    #     '--output_dir sft_outputs',
    #     '--overwrite_cache',
    #     '--per_device_train_batch_size 1',
    #     '--gradient_accumulation_steps 4',
    #     '--lr_scheduler_type cosine',
    #     '--logging_steps 10',
    #     '--save_steps 100',
    #     '--learning_rate 5e-5',
    #     '--num_train_epochs 3.0',
    #     '--plot_loss',
    #     '--fp16'
    #     # '--max_length ',
    # ]

    run_exp()
