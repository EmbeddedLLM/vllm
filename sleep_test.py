# export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
# export VLLM_USE_V1=1
# export VLLM_USE_TRITON_FLASH_ATTN=1

import time
import torch
from vllm import LLM

if __name__ == "__main__":
    
    model_list = [
                  #("mistralai/Mixtral-8x22B-v0.1", 0.8, 4),
                  #('meta-llama/Llama-3.1-8B-Instruct', 0.6, 1),
                  ('meta-llama/Llama-3.1-8B-Instruct', 0.6, 4),
                  #('meta-llama/Llama-3.1-70B-Instruct', 0.8, 4),
                  # ("mistralai/Mixtral-8x22B-v0.1", 0.2, 4),
                  # ("meta-llama/Meta-Llama-3-405B-Instruct", 0.2, 8),
                  ]
    for model, gpu_memory_utilization, tensor_parallel_size in model_list:
        print(f"Testing {model} with {tensor_parallel_size} tensor parallel size and {gpu_memory_utilization} GPU memory utilization")
        llm = LLM(model=model, enable_sleep_mode=True, 
              tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization,
              #disable_custom_all_reduce=True,
              #disable_mm_preprocessor_cache=True,
              #skip_tokenizer_init=False,
              max_model_len=8192,
              max_num_batched_tokens=8192,
              #compilation_config={"level": 3, "cudagraph_capture_sizes": [1]}
              )

        def run_inference(prompt):
            outputs = llm.generate(prompt)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
        print("CUDA Memory Usage (after inference):")
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated()=}")

        run_inference("San Francisco is")
        llm.sleep()

        print("CUDA Memory Usage (after sleep):")
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated()=}")
        time.sleep(5)

        llm.wake_up()

        print("CUDA Memory Usage (after wakeup):")
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated()=}")

        run_inference("Paris is")
        llm.sleep()

        print("CUDA Memory Usage (after sleep):")
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated()=}")
        time.sleep(5)

        llm.wake_up()

        print("CUDA Memory Usage (after wakeup):")
        torch.cuda.empty_cache()
        print(f"{torch.cuda.memory_allocated()=}")

        run_inference("San Francisco is")
