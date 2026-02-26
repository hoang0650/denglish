def formatting_prompts_func(example):
    instruction = example['instruction']
    input_text = example['input']
    output = example['output']
    
    # Construct Llama 3 format
    user_content = instruction
    if input_text:
        user_content += f"\nInput:\n{input_text}"
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant for learning English and German grammar and pronunciation."},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output}
    ]
    
    # The SFTTrainer expects a field named "text"
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
