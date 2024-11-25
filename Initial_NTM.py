import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MEMORY_SIZE = 128  # Memory slots
MEMORY_DIM = 20    # Dimensionality of each memory slot
INPUT_DIM = 10     # Input dimension (e.g., embedding size for text)
HIDDEN_DIM = 50    # Controller's hidden dimension
BETA = 2           # Attention strength
SEQ_LENGTH = 5     # Sequence length for input text

# Simple Embedding for Text Input
class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        return self.embedding(x)

# Neural Turing Machine
class NTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, memory_size, memory_dim):
        super(NTM, self).__init__()
        self.controller = nn.Linear(input_dim, hidden_dim)  # Simple controller
        self.memory_init = nn.Parameter(torch.randn(memory_size, memory_dim))  # External Memory
        self.read_head = ReadHead(hidden_dim, memory_dim)
        self.write_head = WriteHead(hidden_dim, memory_dim)
        
    def forward(self, x):
        # Assuming x is already embedded
        out = torch.zeros(MEMORY_DIM)  
        memory = self.memory_init.clone()  # Initialize memory for this forward pass
        for seq in x:
            hidden = F.relu(self.controller(seq))  # Controller output
            read_vector = self.read_head(hidden, memory)
            out += read_vector  
            memory = self.write_head(hidden, memory)  # Update memory for this time step
        return out.unsqueeze(0)  

class ReadHead(nn.Module):
    def __init__(self, hidden_dim, memory_dim):
        super(ReadHead, self).__init__()
        self.key_generator = nn.Linear(hidden_dim, memory_dim)
        
    def forward(self, hidden, memory):
        key = torch.tanh(self.key_generator(hidden))  # Search key
        weights = self.content_based_addressing(key, memory)
        read_vector = weights.unsqueeze(1) * memory
        return read_vector.sum(dim=0)
    
    def content_based_addressing(self, key, memory):
        sim = F.cosine_similarity(key.unsqueeze(0), memory, dim=1, eps=1e-6)
        weights = F.softmax(BETA * sim, dim=0)
        return weights

class WriteHead(nn.Module):
    def __init__(self, hidden_dim, memory_dim):
        super(WriteHead, self).__init__()
        self.erase_generator = nn.Linear(hidden_dim, memory_dim)
        self.add_generator = nn.Linear(hidden_dim, memory_dim)
        self.key_generator = nn.Linear(hidden_dim, memory_dim)
        
    def forward(self, hidden, memory):
        erase_vector = torch.sigmoid(self.erase_generator(hidden))
        add_vector = torch.tanh(self.add_generator(hidden))
        key = torch.tanh(self.key_generator(hidden))
        weights = self.content_based_addressing(key, memory)
        weighted_erase = weights.unsqueeze(1) * erase_vector
        weighted_add = weights.unsqueeze(1) * add_vector
        memory_update = memory * (1 - weighted_erase) + weighted_add
        return memory_update  # Return the updated memory
    
    def content_based_addressing(self, key, memory):
        sim = F.cosine_similarity(key.unsqueeze(0), memory, dim=1, eps=1e-6)
        weights = F.softmax(BETA * sim, dim=0)
        return weights

# Example Usage with Custom Text Input
if __name__ == "__main__":
    # Dummy text input (e.g., from a vocabulary of size 100)
    vocab_size = 100
    text_input = np.random.randint(0, vocab_size, size=SEQ_LENGTH) # Assume the text input was converted into the integer value
    text_input = torch.from_numpy(text_input).long()
    
    # Embedding Layer
    embedding = TextEmbedding(vocab_size, INPUT_DIM)
    embedded_input = embedding(text_input)
    
    # NTM
    ntm = NTM(INPUT_DIM, HIDDEN_DIM, MEMORY_SIZE, MEMORY_DIM)
    output = ntm(embedded_input)  # <--- Removed unsqueeze(1) as it's not needed here
    print("Output Shape:", output.shape)
    print("Output:", output)
