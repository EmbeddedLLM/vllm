#example.py

from vllm import LLM, SamplingParams

def test():

    prompts = [
        "The color of the sky is blue but sometimes it can also be",
        "The capital of France is",
        "What is batch inference?",
        "What is batch inference?",
        "What is batch inference?"
    ]
    sampling_params = SamplingParams(temperature=0.6,
                                     top_p=0.1,
                                     max_tokens=256)
    # llm = LLM(
    #     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    #     tensor_parallel_size=8,
    #     max_model_len=430000,
    #     gpu_memory_utilization=0.7,
    #     enforce_eager=False, 
    # )


    # llm = LLM(
    #     model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    #     tensor_parallel_size=8,
    #     max_model_len=430000,
    #     gpu_memory_utilization=0.7,
    #     enforce_eager=False,
    # )

    
    llm = LLM(
        model="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
        tensor_parallel_size=8,
        max_model_len=430000,
        gpu_memory_utilization=0.7,
        enforce_eager=False,
    )

    # llm = LLM(
    #     model="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    #     tensor_parallel_size=8,
    #     max_model_len=430000,
    #     gpu_memory_utilization=0.9,
    #     enforce_eager=False, 
    # )

    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    test()