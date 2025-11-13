"""
Data loading and preprocessing for GPT model training.
Includes dataset classes and dataloader creation utilities.
"""
import os
import requests
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


def load_text_data(file_path: str = "the-verdict.txt",
                   url: str = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt") -> str:
    """
    Load text data from file or download from URL if file doesn't exist.
    
    Args:
        file_path: Local file path to store/load text.
        url: URL to download text from if file doesn't exist.
    
    Returns:
        Text data as a string.
    """
    if not os.path.exists(file_path):
        print(f"Downloading text from {url}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_data = response.text
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
        print(f"✓ Text saved to {file_path}")
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        print(f"✓ Text loaded from {file_path}")
    
    return text_data


def print_text_stats(text_data: str):
    """Print character and token statistics for text data."""
    tokenizer = tiktoken.get_encoding("gpt2")
    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print(f"Characters: {total_characters}")
    print(f"Tokens: {total_tokens}")


# Load text data on module import
text_data = load_text_data()
