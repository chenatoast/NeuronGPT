import os
import modal

# Create the Modal app (formerly known as Stub)
app = modal.App("gpt-language-model")

# Create a container image with PyTorch
image = modal.Image.debian_slim().pip_install(
    "torch==2.0.1",
    "numpy",
)

# Add input.txt to the image
image = image.add_local_file("NeuronGPT/input.txt", "/root/input.txt")

# Create a volume for model checkpoints
volume = modal.Volume.from_name("gpt-training-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/root/model": volume}
)
def train_gpt_model():
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import os
    
    # Create model directory
    os.makedirs("/root/model", exist_ok=True)
    
    # hyperparameters
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    
    print(f"Using device: {device}")
    torch.manual_seed(1337)
    
    # Read the input file
    with open('/root/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Input text length: {len(text)} characters")
    
    # Create character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Create mappings
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # data loading
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    class Head(nn.Module):
        def __init__(self, head_size):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            B,T,C = x.shape
            k = self.key(x)
            q = self.query(x)
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
            wei = F.softmax(wei, dim=-1)
            wei = self.dropout(wei)
            v = self.value(x)
            out = wei @ v
            return out
    
    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
            self.proj = nn.Linear(head_size * num_heads, n_embd)
            self.dropout = nn.Dropout(dropout)
    
        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            out = self.dropout(self.proj(out))
            return out
    
    class FeedFoward(nn.Module):
        def __init__(self, n_embd):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.ReLU(),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(dropout),
            )
    
        def forward(self, x):
            return self.net(x)
    
    class Block(nn.Module):
        def __init__(self, n_embd, n_head):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = MultiHeadAttention(n_head, head_size)
            self.ffwd = FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
    
        def forward(self, x):
            x = x + self.sa(self.ln1(x))
            x = x + self.ffwd(self.ln2(x))
            return x
    
    class GPTLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            self.ln_f = nn.LayerNorm(n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.apply(self._init_weights)
    
        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
        def forward(self, idx, targets=None):
            B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device))
            x = tok_emb + pos_emb
            x = self.blocks(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
    
            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
    
            return logits, loss
    
        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    
    # Create model
    model = GPTLanguageModel()
    model = model.to(device)
    
    # Print model size
    param_count = sum(p.numel() for p in model.parameters())/1e6
    print(f"{param_count:.2f} M parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # Save checkpoint
            checkpoint_path = "/root/model/gpt_model_checkpoint.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': {'stoi': stoi, 'itos': itos}
            }, checkpoint_path)
            
        # Sample batch and train
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # Generate sample text
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
    print("\nSample generated text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    # Save final model
    final_path = "/root/model/gpt_model_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': {'stoi': stoi, 'itos': itos}
    }, final_path)
    
    return "Training complete!"

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={"/root/model": volume}
)
def generate_text(max_tokens=500):
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    
    # Load the model
    model_path = "/root/model/gpt_model_final.pt"
    if not os.path.exists(model_path):
        return "No model found. Please train the model first."
        
    checkpoint = torch.load(model_path)
    stoi = checkpoint['vocab']['stoi']
    itos = checkpoint['vocab']['itos']
    
    # Define parameters
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_size = len(stoi)
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    
    # Define model classes (same as in training)
    # [Include the model classes: Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel]
    # This is the same code from the training function
    
    # Generate text code
    # [Include code to generate text]
    
    # Return the generated text
    return "Generated text would go here"

@app.local_entrypoint()
def main():
    print("Starting model training...")
    result = train_gpt_model.remote()
    print(result)