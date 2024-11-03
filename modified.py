import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.cuda.amp import autocast, GradScaler
import subprocess
import threading
import os
import time
import numpy as np

# ============================
# GPU Monitoring Setup
# ============================

stop1_event = threading.Event()
memory_utilization_records = []

def nvidia_smi1(gpu_index):
    """
    Starts the GPU monitoring process using nvidia-smi.
    Monitors the specified GPU and writes the output to densenetsmi.txt.
    """
    monitor_command = f"nvidia-smi pmon -i {gpu_index} -f densenetsmi.txt"
    task = subprocess.Popen(monitor_command, shell=True)

    # Run until stop1_event is set
    while not stop1_event.is_set():
        time.sleep(1)

    # Terminate monitoring process
    task.terminate()
    time.sleep(1)  # Allow graceful termination
    if task.poll() is None:
        task.kill()
    print("Stopped GPU monitoring process.")

def read_memory_utilization():
    """
    Reads GPU memory utilization from densenetsmi.txt every 5 seconds.
    Appends the memory usage to memory_utilization_records.
    """
    while not stop1_event.is_set():
        time.sleep(5)
        try:
            with open("densenetsmi.txt", "r") as file:
                lines = file.readlines()
                if len(lines) > 1:  
                    last_line = lines[-1].strip()
                    fields = last_line.split()
                    if len(fields) > 4 and fields[4].isdigit():  # Check if mem utilization is present
                        mem_util = int(fields[4])  # Memory utilization is in the 5th column
                        memory_utilization_records.append(mem_util)
                        print(f"Memory Utilization of GPU {gpu_i}: {mem_util}%")
        except FileNotFoundError:
            print("Monitoring file not found. Waiting for data...")

# ============================
# Device and Model Setup
# ============================

# Set up device and scaler for mixed-precision training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = GradScaler()  # Use GradScaler with float16 precision

# Define model path and GPU index
base_model_path = "/home/pacs/Srini/hinadixit/Llama-3.2-1B"
trained_model_path = "./my_model"  # Directory where the trained model is saved
gpu_i = 2

# Function to load the model (trained or base)
def load_model(base_path, trained_path, device):
    """
    Loads the trained model if it exists; otherwise, loads the base model.
    """
    if os.path.exists(trained_path):
        print(f"Loading trained model from {trained_path}...")
        model = AutoModelForCausalLM.from_pretrained(trained_path, use_cache=False).to(device)
        print("Trained model loaded.")
    else:
        print(f"Trained model not found. Loading base model from {base_path}...")
        model = AutoModelForCausalLM.from_pretrained(base_path, use_cache=False).to(device)
        model.gradient_checkpointing_enable()
        print("Base model loaded and gradient checkpointing enabled.")
    return model

# Load the appropriate model
model = load_model(base_model_path, trained_model_path, device)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token

# ============================
# Dataset Preparation
# ============================

# Load and preprocess PIQA dataset
dataset = load_dataset("piqa")
small_train_dataset = dataset["train"]  # Use full train dataset
small_eval_dataset = dataset["validation"].select(range(200))  # Small evaluation subset for testing

def preprocess_function(examples):
    """
    Tokenizes the input and creates labels for causal language modeling.
    """
    inputs = [q + " " + s for q, s in zip(examples["goal"], examples["sol1"])]
    tokenized_inputs = tokenizer(
        inputs, padding="max_length", truncation=True, max_length=64
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

# Apply preprocessing
tokenized_train_dataset = small_train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = small_eval_dataset.map(preprocess_function, batched=True)

# Data collator with no masking (for causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# ============================
# Training Arguments
# ============================

args = TrainingArguments(
    output_dir="llama-piqa-finetune",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=1e-4,
    bf16=True,  # Use bfloat16 precision for GPUs that support it
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_total_limit=1,  # Only keep the latest checkpoint
    load_best_model_at_end=True,  # Load the best model found during training
    metric_for_best_model="accuracy",  # Specify the metric to determine the best model
)

# ============================
# Custom Data Loader
# ============================

def custom_data_loader(trainer, pin_memory=True):
    """
    Overrides the Trainer's get_train_dataloader method to enable pin_memory.
    """
    trainer.train_dataloader = lambda: torch.utils.data.DataLoader(
        trainer.train_dataset,
        batch_size=trainer.args.train_batch_size,
        shuffle=True,
        collate_fn=trainer.data_collator,
        pin_memory=pin_memory
    )

# ============================
# Initialize Trainer
# ============================

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,  # Provide eval_dataset here
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Apply custom data loader to enable pin_memory
custom_data_loader(trainer, pin_memory=True)

# ============================
# Start GPU Monitoring Threads
# ============================

monitoring_thread = threading.Thread(target=nvidia_smi1, args=(gpu_i,))
monitoring_thread.start()
read_thread = threading.Thread(target=read_memory_utilization)
read_thread.start()

# ============================
# Training and Evaluation
# ============================

try:
    if not os.path.exists(trained_model_path):
        # Ensure the directory for saving the model exists
        os.makedirs(trained_model_path, exist_ok=True)

        # Start training with mixed precision
        print("Starting training...")
        trainer.train()
        
        # Save the trained model directly
        print("Saving trained model...")
        model.save_pretrained(trained_model_path)
        tokenizer.save_pretrained(trained_model_path)  # Save tokenizer if needed
        print(f"Trained model saved to {trained_model_path}")
    else:
        print(f"Trained model already exists at {trained_model_path}. Skipping training.")
    
    # Perform evaluation after training (if training was done)
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

except Exception as e:
    print(f"An error occurred during training or evaluation: {e}")

finally:
    # Ensure GPU monitoring threads are stopped even if an error occurs
    stop1_event.set()
    monitoring_thread.join()
    read_thread.join()
    
    # Calculate and display average memory utilization
    if memory_utilization_records:
        average_memory_util = sum(memory_utilization_records) / len(memory_utilization_records)
    else:
        average_memory_util = 0
    print(f"Average Memory Utilization of GPU {gpu_i}: {average_memory_util:.2f}%")

# ============================
# Test Set Evaluation (Optional)
# ============================

# Load a separate test set for evaluation
test_dataset = dataset["validation"].select(range(200))  # Increase range for better results
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

def evaluate_test_set(trainer, tokenizer, test_dataset, device):
    """
    Evaluates the trained model on the test set by having it select between Solution 1 and Solution 2.
    Returns the accuracy of the model on the test set.
    """
    predictions, references = [], []

    for idx, sample in enumerate(test_dataset):
        #print(f"****************** Input {idx+1} ***************************")
        goal = sample["goal"]
        sol1 = sample["sol1"]
        sol2 = sample["sol2"]
        label = sample["label"]  # 0 for sol1, 1 for sol2

        prompt = (
            f"Goal: {goal}\n"
            f"Solution 1: {sol1}\n"
            f"Solution 2: {sol2}\n"
            f"Which solution is better? Solution"
        )

        # Tokenize the input prompt
        encoding = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # Adjust based on  data
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Generate the model's selection
        with autocast():  # Mixed precision during inference
            outputs = trainer.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,    # Pass attention_mask
                max_new_tokens=10,                # Limit to generating short answers
                num_return_sequences=1,           # Single output
                temperature=0.0,                   # Low temperature for deterministic output
                top_k=0,                           # Disable top_k
                top_p=1.0,                         # No nucleus sampling
                repetition_penalty=1.2,            # Mild repetition penalty
                do_sample=False,                   # Greedy decoding
                early_stopping=True                # Stop if EOS token is reached
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #print(f"****************** Output {idx+1} ***************************")
        #print(generated_text)
        #print("\n")

        # Extract the solution number from the generated text
        if "solution 1" in generated_text.lower():
            pred_label = 0
        elif "solution 2" in generated_text.lower():
            pred_label = 1
        elif generated_text.strip().endswith("1"):
            pred_label = 0
        elif generated_text.strip().endswith("2"):
            pred_label = 1
        else:
            # Handle unexpected outputs by assigning a default or skipping
            print(f"Unexpected output format for sample {idx+1}: {generated_text}")
            continue  # Skip this sample

        predictions.append(pred_label)
        references.append(label)

    # Calculate accuracy
    accuracy = accuracy_score(references, predictions)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    return accuracy

# Run test set evaluation
test_accuracy = evaluate_test_set(trainer, tokenizer, tokenized_test_dataset, device)
