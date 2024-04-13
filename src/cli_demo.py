from llmtuner import ChatModel
from llmtuner.extras.misc import torch_gc

import sys
import os

try:
    import platform
    if platform.system() != "Windows":
        import readline
except ImportError:
    print("Install `readline` for a better experience.")


def phi2_run():
    # 在这里模拟传递命令行参数
    sys.argv = [
        'cli_demo.py',  # 通常是脚本名称
        '--model_name_or_path', 'microsoft/phi-2',
        # # '--adapter_name_or_path', '',
        '--template', 'phi2_chat',
        # # '--finetuning_type', 'lora',
        # # '--per_device_eval_batch_size=1',
        # '--max_length ',
    ]

    chat_model = ChatModel()
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nAlice: ")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            torch_gc()
            print("History has been removed.")
            continue

        print("Bob: ", end="", flush=True)

        response = ""
        detect_end = False
        for new_text in chat_model.stream_chat(query, history):
            temp_str = response + new_text
            if "\nAlice" in temp_str:
                temp_str = temp_str[:temp_str.index("\nAlice")]
                new_text = temp_str[len(response):]
                detect_end = True

            print(new_text, end="", flush=True)
            response += new_text
            if detect_end:
                break

        print()

        history = history + [(query, response)]


def tinyllama_run():
    # 在这里模拟传递命令行参数
    sys.argv = [
        'cli_demo.py',  # 通常是脚本名称
        '--model_name_or_path', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        # # '--adapter_name_or_path', '',
        '--template', 'zephyr', # 和zephyr使用同样模板
        # # '--finetuning_type', 'lora',
        # # '--per_device_eval_batch_size=1',
        # '--max_length ',
    ]

    chat_model = ChatModel()
    history = []
    print("Welcome to the CLI application, use `clear` to remove the history, use `exit` to exit the application.")

    while True:
        try:
            query = input("\nuser:")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
            continue
        except Exception:
            raise

        if query.strip() == "exit":
            break

        if query.strip() == "clear":
            history = []
            torch_gc()
            print("History has been removed.")
            continue

        print("assistant:", end="</s>", flush=True)

        response = ""
        for new_text in chat_model.stream_chat(query, history):
            print(new_text, end="", flush=True)
            response += new_text

        print()

        history = history + [(query, response)]


if __name__ == "__main__":
    # os.environ['no_proxy'] = '*'  # 阻止requests库使用代理
    os.environ["all_proxy"] = 'socks5://127.0.0.1:10808'

    # 调用 main 函数
    # phi2_run()

    tinyllama_run()

