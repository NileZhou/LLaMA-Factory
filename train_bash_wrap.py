import subprocess
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


args = [
    'python', 'src/train_bash.py',
    '--stage', 'sft',
    '--do_train',
    '--model_name_or_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    '--dataset', 'alpaca_gpt4_zh',
    '--template', 'zephyr',
    '--finetuning_type', 'lora',
    '--lora_target', 'q_proj,v_proj',
    '--output_dir', 'sft_outputs',
    '--overwrite_cache',
    '--per_device_train_batch_size', '1',
    '--gradient_accumulation_steps', '4',
    '--lr_scheduler_type', 'cosine',
    '--logging_steps', '10',
    '--save_steps', '10',
    '--learning_rate', '5e-5',
    '--num_train_epochs', '3.0',
    '--plot_loss',
    '--fp16'
]

# Note: CUDA_VISIBLE_DEVICES is a special environment variable and might not be directly recognized in Windows the same way it is in Linux.
# You might need to set it differently if you're running this in a Windows environment.

result = subprocess.run(args, shell=True, capture_output=True, text=True)

print("stdout:", result.stdout)
print("stderr:", result.stderr)