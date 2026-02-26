import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import yaml
import os
import wandb

def train():
    # Load configuration from config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_id = config["model"]["id"]
    dataset_path = config["model"]["dataset_path"]
    output_dir = config["model"]["output_dir"]

    # Initialize WandB
    wandb_project = config["wandb"]["project"]
    wandb_entity = config["wandb"]["entity"]
    
    # Ensure WANDB_API_KEY is set as an environment variable
    if "WANDB_API_KEY" not in os.environ:
        print("Warning: WANDB_API_KEY environment variable not set. WandB logging may fail.")
    
    wandb.init(project=wandb_project, entity=wandb_entity)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for fp16

    # 2. BitsAndBytes Config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA Config
    lora_config = LoraConfig(
        r=config["training"]["lora_r"],
        lora_alpha=config["training"]["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config["training"]["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # 5. Load Dataset
    dataset = load_from_disk(dataset_path)

    # Format dataset for Llama 3 Instruct
    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            instruction = example['instruction'][i]
            input_text = example['input'][i]
            output = example['output'][i]
            
            # Construct Llama 3 format
            # <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSystem Prompt<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nUser Prompt<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
            
            user_content = instruction
            if input_text:
                user_content += f"\nInput:\n{input_text}"
            
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant for learning English and German grammar and pronunciation."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output}
            ]
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            output_texts.append(text)
        return output_texts

    # 6. SFTConfig (replaces TrainingArguments)
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        logging_steps=config["training"]["logging_steps"],
        max_steps=config["training"]["max_steps"],
        save_strategy="steps",
        save_steps=config["training"]["save_steps"],
        optim="paged_adamw_32bit",
        fp16=config["training"]["fp16"],
        push_to_hub=False,
        report_to="wandb",
        run_name=wandb_project,
        gradient_checkpointing=True,
        max_seq_length=2048, # Moved from SFTTrainer to SFTConfig
        dataset_text_field=None, # Explicitly set to None as we use formatting_func
        packing=False,
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        args=training_args,
    )

    # 8. Start Training
    print("--- Starting Training with Vision-Enhanced Data ---")
    trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    train()
