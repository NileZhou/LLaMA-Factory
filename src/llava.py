from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

import base64


base64_encode_str = ""
with open("/home/xiaoyi/Downloads/cnn.png", "rb") as f:
    image_data = f.read()
    base64_encode_data = base64.b64encode(image_data)
    base64_encode_str = base64_encode_data.decode("utf-8")

chat_handler = Llava15ChatHandler(clip_model_path="/home/xiaoyi/Downloads/mmproj-model-f16.gguf")
llm = Llama(
  model_path="/home/xiaoyi/Downloads/llava-v1.6-34b.Q4_K_M.gguf",
  chat_handler=chat_handler,
  n_ctx=2048,# n_ctx should be increased to accomodate the image embedding
  logits_all=True,# needed to make llava work
)



llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:,"+base64_encode_str}},
                # {"type": "image_url", "image_url": {"url": "https://.../image.png"}},
                {"type": "text", "text": "Describe this image in detail please."}
            ]
        }
    ]
)

