**README**


Overview

This project includes:

	•	Two Python Scripts:
	•	baseline.py: A straightforward, standard training setup.
	•	modified.py: An optimized version with improvements in training efficiency, memory management, and model performance.
	•	Memory Monitoring: Real-time monitoring of GPU memory utilization using NVIDIA’s nvidia-smi.
	•	Model: Fine-tuning is conducted on the LLAMA-3.1B model from Hugging Face’s Model Hub.

Requirements

Ensure you have the following installed:

	•	Python 3.8 or higher
	•	NVIDIA GPU with CUDA support
	•	Python Packages:
	•	torch with CUDA support
	•	transformers
	•	datasets
	•	sklearn

Install the necessary packages:

pip install -r requirements.txt

Usage

Baseline Version

The Baseline version provides a standard approach to model training.

	1.	Run baseline.py: python3 baseline.py

This script includes:
	•	Standard training configurations for the LLAMA-3.1B model.
	•	Basic GPU memory monitoring using nvidia-smi.
	•	Evaluation of model performance on a subset of the validation set.

Modified Version

The Modified version includes several enhancements to optimize training and memory efficiency.

	1.	Run modified.py: python3 modified.py

Modifications in modified.py:
	•	Mixed-Precision Training: Uses autocast and GradScaler from torch.cuda.amp for mixed-precision, reducing memory usage and improving speed.
	•	Gradient Accumulation and Checkpointing:
	•	Gradient Accumulation: Maintains gradient_accumulation_steps=4 to simulate larger batch sizes.
	•	Gradient Checkpointing: Continues to use gradient checkpointing for memory efficiency.
	•	Checkpoint Management:
	•	save_total_limit=1: Saves only the latest checkpoint to conserve disk space.
	•	load_best_model_at_end=True: Automatically loads the best-performing model based on the specified metric.
	•	metric_for_best_model="accuracy": Specifies accuracy as the key metric for selecting the best model.
	•	Pin Memory: Enables pin_memory=True in data loaders to accelerate data transfer to GPUs.
	•	Enhanced Evaluation: Includes an improved evaluation function that structures prompts for better model guidance and parses outputs for accurate performance measurement.

Monitoring GPU Memory Utilization

Both versions utilize NVIDIA’s nvidia-smi to log memory usage in real-time. Memory utilization is recorded every 5 seconds and summarized at the end of the training.

	1.	Ensure nvidia-smi is installed (typically included with NVIDIA drivers).
	2.	Real-Time Monitoring: During training, memory utilization is logged in densenetsmi.txt and printed periodically.

Model Loading

The model is sourced from Hugging Face. Update the model_path in both scripts if necessary:

model_path = "path/to/llama-3.1b"  # Update to your path if needed


## Key Differences Between Baseline and Modified Versions

| Feature                   | `baseline.py`                              | `modified.py`                                                            |
|---------------------------|--------------------------------------------|---------------------------------------------------------------------------|
| **Precision**             | Standard (full precision)                  | Mixed-precision with `autocast` and `GradScaler`                         |
| **Checkpoint Management** | Saves all checkpoints                      | Saves only the latest checkpoint (`save_total_limit=1`)                   |
| **Model Selection**       | Final model after training                 | Loads best-performing model based on accuracy (`load_best_model_at_end=True`) |
| **Gradient Accumulation** | Yes (steps=4)                              | Yes (steps=4)                                                             |
| **Gradient Checkpointing**| Enabled                                    | Enabled                                                                  |
| **Pin Memory**            | No                                         | Yes (`pin_memory=True`)                                                  |
| **GPU Memory Monitoring** | Basic monitoring with `nvidia-smi`         | Enhanced monitoring and logging                                          |
| **Evaluation**            | Standard accuracy calculation              | Structured prompting and parsing for improved evaluation                 |

Additional Information

	•	Training Output: Both scripts save the model and report training metrics after completion.
	•	Evaluation Results: Both versions provide accuracy metrics on a subset of the validation set, with the Modified version offering enhanced evaluation capabilities.

