from vllm import LLM, SamplingParams

# small batch of prompt the model will respond

prompts = [
    "Hello, my name is Victor, what's your name?",
]

# parameters for configuration
sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.95,
    max_tokens=200,
    repetition_penalty=1.15
)

# create llm engine
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

# generate outputs for all prompts in a single batched call
outputs = llm.generate(prompts, sampling_params)

# print the results
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")