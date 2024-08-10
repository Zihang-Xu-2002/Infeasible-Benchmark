from datasets import load_dataset

# Load all sections
dataset = load_dataset("TrustLLM/TrustLLM-dataset")

# Load one of the sections
dataset = load_dataset("TrustLLM/TrustLLM-dataset", data_dir="robustness/ood_generalization.json")

print(dataset)