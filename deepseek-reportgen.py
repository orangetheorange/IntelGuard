import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def report(result):
    model_name = "deepseek-ai/deepseek-llm-7b-chat"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Construct chat messages
    question = (
        f"{result} \nThis is a result of a vulnerability scan for the target. "
        f"Generate a report containing a description and a solution. "
        f"Your response should only contain the report itself."
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant that generates reports for cybersecurity scan results."},
        {"role": "user", "content": question}
    ]

    # Apply chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    # Create attention mask
    attention_mask = torch.ones_like(inputs)

    # Generate response
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_new_tokens=400,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode response
    full_response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return full_response.strip()


# Example usage
print(report("Found vulnerability: EternalBlue"))
