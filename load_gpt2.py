# %% [markdown]
# # GPT-2 Model Loading and Exploration
# 
# This notebook loads the GPT-2 model using HuggingFace Transformers and demonstrates basic usage.

# %%
# Import necessary libraries
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

#%%
device = "cuda" if torch.cuda.is_available() else "cpu"
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model.config.output_attentions = True

# %%
# Display model information
print(f"Number of parameters: {model.num_parameters():,}")
print(f"Number of layers: {model.config.n_layer}")
print(f"Hidden size: {model.config.n_embd}")
print(f"Number of attention heads: {model.config.n_head}")
print(f"Vocabulary size: {model.config.vocab_size}")
print(f"\nModel config output_attentions: {model.config.output_attentions}")
print(f"Model config output_hidden_states: {model.config.output_hidden_states}")

# %%
# Test that attention outputs work
test_input = tokenizer("Hello world", return_tensors="pt").to(device)
model.eval()
model.config.output_attentions = True
with torch.no_grad():
    test_output = model(**test_input, output_attentions=True)

print("Layer 0 attention shape:", test_output.attentions[0].shape)
# %%
# Test text generation
prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize input
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
model.eval()
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{generated_text}")

# %% [markdown]
# ## Attention Pattern Visualization
# 
# Visualize attention patterns for specific layers and heads

# %%
# Import visualization libraries
import matplotlib.pyplot as plt
import numpy as np
import circuitsvis as cv
from IPython.display import display

# %%
def visualize_attention(model, tokenizer, text, layer_idx=0, head_idx=0, use_manual=False):
    """
    Visualize attention patterns for a specific layer and head.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Input text to analyze
        layer_idx: Layer index (0 to n_layer-1)
        head_idx: Attention head index (0 to n_head-1)
        use_manual: If True, use manual extraction method
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Use standard output_attentions method
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    print(f"Number of layers with attention: {len(outputs.attentions)}")
    print(f"Attention shape for layer {layer_idx}: {outputs.attentions[layer_idx].shape}")
    
    # Extract attention for the specified layer and head
    # attentions shape: (batch, num_heads, seq_len, seq_len)
    attention = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attention, cmap='viridis', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, ha='center')
    ax.set_yticklabels(tokens)
    
    # Labels and title
    ax.set_xlabel("Key (attending to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"Attention Pattern - Layer {layer_idx}, Head {head_idx}")
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label="Attention Weight")
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, attention, tokens

# %%
# Visualize attention patterns for a sample prompt
sample_text = "The cat sat on the mat and looked around"

# Visualize layer 0, head 0
fig, attention_weights, tokens = visualize_attention(
    model, tokenizer, sample_text, 
    layer_idx=0, 
    head_idx=0,
)
plt.show()

print(f"\nTokens: {tokens}")
print(f"Attention matrix shape: {attention_weights.shape}")

# %%
# Visualize multiple heads from the same layer
def visualize_multiple_heads(model, tokenizer, text, layer_idx=0, num_heads=4, use_manual=False):
    """
    Visualize attention patterns for multiple heads in a layer.
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Check if attentions are returned
    if outputs.attentions is None or outputs.attentions[layer_idx] is None:
        print("Warning: output_attentions returned None. Falling back to manual extraction.")
        return visualize_multiple_heads(model, tokenizer, text, layer_idx, num_heads, use_manual=True)
    
    # Extract attention for the specified layer
    attention = outputs.attentions[layer_idx][0].cpu().numpy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for head_idx in range(min(num_heads, len(axes))):
        ax = axes[head_idx]
        
        # Plot heatmap
        im = ax.imshow(attention[head_idx], cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, ha='center', fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        
        # Labels and title
        ax.set_xlabel("Key")
        ax.set_ylabel("Query")
        ax.set_title(f"Layer {layer_idx}, Head {head_idx}")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label="Attention")
    
    plt.tight_layout()
    return fig

# %%
# Compare attention patterns across multiple heads
fig = visualize_multiple_heads(
    model, tokenizer, sample_text,
    layer_idx=0,
    num_heads=4,
)
plt.show()

# %%
# Analyze attention statistics
def analyze_attention_stats(model, tokenizer, text, layer_idx=0, use_manual=False):
    """
    Compute statistics about attention patterns in a layer.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Check if attentions are returned
    if outputs.attentions is None or outputs.attentions[layer_idx] is None:
        print("Warning: output_attentions returned None. Falling back to manual extraction.")
        return analyze_attention_stats(model, tokenizer, text, layer_idx, use_manual=True)
    
    attention = outputs.attentions[layer_idx][0].cpu().numpy()
    
    print(f"Layer {layer_idx} Attention Statistics:")
    print(f"  Number of heads: {attention.shape[0]}")
    print(f"  Sequence length: {attention.shape[1]}")
    print(f"  Attention mean: {attention.mean():.4f}")
    print(f"  Attention std: {attention.std():.4f}")
    print(f"  Attention min: {attention.min():.4f}")
    print(f"  Attention max: {attention.max():.4f}")
    
    # Analyze which positions attend to each other most
    avg_attention = attention.mean(axis=0)  # Average across heads
    
    print("\nAverage attention to each token:")
    for i, token in enumerate(tokens):
        print(f"  {token}: {avg_attention[:, i].mean():.4f}")
    
    return attention

# %%
# Run attention statistics
attention_stats = analyze_attention_stats(
    model, tokenizer, sample_text, 
    layer_idx=0,
)

# %%
import circuitsvis as cv
# Visualize layer 1 attention patterns using circuitsvis
def plot_attention_circuitsvis(model, tokenizer, text, layer_idx=1):
    """
    Plot attention patterns for a specific layer using circuitsvis.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        text: Input text to analyze
        layer_idx: Layer index (0 to n_layer-1)
    """
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Get attention patterns
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Extract attention for the specified layer
    # attentions shape: (batch, num_heads, seq_len, seq_len)
    attention_pattern = outputs.attentions[layer_idx][0].cpu()  # Remove batch dim
    
    # Create head names
    attention_head_names = [f"L{layer_idx}H{i}" for i in range(model.config.n_head)]
    
    # Display using circuitsvis
    print(f"Layer {layer_idx} Attention Patterns:")
    display(
        cv.attention.attention_heads(
            tokens=tokens,
            attention=attention_pattern,
            attention_head_names=attention_head_names,
        )
    )
    
    return attention_pattern, tokens

# %%
# Plot layer 1 attention heads using circuitsvis
layer1_attention, layer1_tokens = plot_attention_circuitsvis(
    model, tokenizer, sample_text, layer_idx=1
)

# %%
