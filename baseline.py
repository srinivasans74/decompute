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
import subprocess
import threading
import os
import time

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model path and GPU index
model_path = "/home/pacs/Srini/hinadixit/Llama-3.2-1B"
gpu_i = 2  # Replace with the GPU index you want to monitor

# Load model with gradient checkpointing to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    use_cache=False  # Avoids conflicts with gradient checkpointing
).to(device)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to the end-of-sequence token

# Load and preprocess PIQA dataset
dataset = load_dataset("piqa")
small_train_dataset = dataset["train"].select(range(1000))  # Use a subset for quicker training

# Preprocess function to tokenize the data
def preprocess_function(examples):
    inputs = [q + " " + s for q, s in zip(examples["goal"], examples["sol1"])]
    tokenized_inputs = tokenizer(
        inputs, padding="max_length", truncation=True, max_length=64  # Adjust max_length as needed
    )
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Use input_ids as labels
    return tokenized_inputs

# Apply preprocessing
tokenized_dataset = small_train_dataset.map(preprocess_function, batched=True)

# Use DataCollatorForLanguageModeling to handle padding and labels alignment
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # For causal language modeling, set mlm=False
)

# Training Arguments
args = TrainingArguments(
    output_dir="llama-piqa-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,  # Use bfloat16 precision for GPUs that support it
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
)

# Initialize Trainer with the data collator
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator  # Add the data collator here
)

# Global variable for monitoring
stop1_event = threading.Event()
memory_utilization_records = []

# Function to start GPU monitoring with nvidia-smi pmon and save output to file
def nvidia_smi1(gpu_index):
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

# Function to read GPU memory utilization from densenetsmi.txt every 5 seconds
def read_memory_utilization():
    while not stop1_event.is_set():
        time.sleep(5)
        try:
            with open("densenetsmi.txt", "r") as file:
                lines = file.readlines()
                if len(lines) > 1:  # Skip the header
                    last_line = lines[-1].strip()
                    fields = last_line.split()
                    if len(fields) > 4 and fields[4].isdigit():  # Check if mem utilization is present
                        mem_util = int(fields[4])  # Memory utilization is in the 5th column
                        memory_utilization_records.append(mem_util)
                        print(f"Memory Utilization of GPU {gpu_i}: {mem_util}%")
        except FileNotFoundError:
            print("Monitoring file not found. Waiting for data...")

# Start GPU monitoring thread
monitoring_thread = threading.Thread(target=nvidia_smi1, args=(gpu_i,))
monitoring_thread.start()

# Start thread to read memory utilization
read_thread = threading.Thread(target=read_memory_utilization)
read_thread.start()

# Start training
trainer.train()
trainer.save_model()

# Stop GPU monitoring after training
stop1_event.set()  # Signal both threads to stop
monitoring_thread.join()
read_thread.join()

# Calculate and display average memory utilization
average_memory_util = sum(memory_utilization_records) / len(memory_utilization_records) if memory_utilization_records else 0
print(f"Average Memory Utilization of GPU {gpu_i}: {average_memory_util:.2f}%")

# Evaluation on the test set
test_dataset = dataset["validation"].select(range(100))  # Use a subset of the validation set for quick evaluation
tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

# Generate predictions
predictions, references = [], []
for sample in tokenized_test_dataset:
    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    outputs = model.generate(input_ids, max_new_tokens=50)
    predicted_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Append the predicted and actual output for accuracy calculation
    predictions.append(predicted_output.strip())
    references.append(sample["labels"])

# Calculate accuracy (if applicable)
accuracy = accuracy_score(references, predictions)
print(f"Model Accuracy on Validation Set: {accuracy * 100:.2f}%")