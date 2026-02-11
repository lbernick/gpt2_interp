# %% [markdown]
# # GPT-2 Model Loading and Exploration
# 
# This notebook loads the GPT-2 model using HuggingFace Transformers and demonstrates basic usage.

# %%
# Import necessary libraries
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# %%
# Load GPT-2 model and tokenizer
print("Loading GPT-2 model...")
model_name = "gpt2"  # Options: gpt2, gpt2-medium, gpt2-large, gpt2-xl
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# %%
# Display model information
print(f"Model name: {model_name}")
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
with torch.no_grad():
    test_output = model(**test_input, output_attentions=True)

print(f"Attentions returned: {test_output.attentions is not None}")
if test_output.attentions is not None:
    print(f"Number of layers: {len(test_output.attentions)}")
    print(f"Type of attentions: {type(test_output.attentions)}")
    print(f"First element: {test_output.attentions[0]}")
    print(f"First element type: {type(test_output.attentions[0])}")
    
    # Check each layer
    for i, attn in enumerate(test_output.attentions):
        if attn is not None:
            print(f"Layer {i} attention shape: {attn.shape}")
        else:
            print(f"Layer {i} attention is None")
else:
    print("WARNING: Attentions are None! This shouldn't happen.")

# %%
# Test text generation
prompt = "The future of artificial intelligence is"
print(f"Prompt: {prompt}")

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

# %%
# Examine model structure
print("Model architecture:")
print(model)

# %% [markdown]
# ## Attention Pattern Visualization
# 
# Visualize attention patterns for specific layers and heads

# %%
# Import visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# %%
# Alternative method: Extract attention weights manually
def get_attention_weights_manual(model, tokenizer, text):
    """
    Manually extract attention weights by hooking into the model.
    This is an alternative if output_attentions doesn't work properly.
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Store attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        # output is a tuple: (hidden_states, (present, attn_weights))
        # or just (hidden_states,) depending on config
        if len(output) > 1 and output[1] is not None:
            if isinstance(output[1], tuple) and len(output[1]) > 1:
                attention_weights.append(output[1][1])  # attn_weights
    
    # Register hooks on each transformer block
    hooks = []
    for block in model.transformer.h:
        hook = block.attn.register_forward_hook(hook_fn)
        hooks.append(hook)
    
    # Run forward pass
    model.eval()
    with torch.no_grad():
        _ = model(**inputs)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_weights, inputs["input_ids"]

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
    
    if use_manual:
        # Use manual extraction method
        all_attentions, _ = get_attention_weights_manual(model, tokenizer, text)
        if layer_idx >= len(all_attentions):
            raise ValueError(f"Layer {layer_idx} not found. Model has {len(all_attentions)} layers.")
        attention = all_attentions[layer_idx][0, head_idx].cpu().numpy()
    else:
        # Use standard output_attentions method
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Check if attentions are returned properly
        if outputs.attentions is None or outputs.attentions[layer_idx] is None:
            print("Warning: output_attentions returned None. Falling back to manual extraction.")
            return visualize_attention(model, tokenizer, text, layer_idx, head_idx, use_manual=True)
        
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
# If you get errors with attentions being None, add use_manual=True parameter
fig, attention_weights, tokens = visualize_attention(
    model, tokenizer, sample_text, 
    layer_idx=0, 
    head_idx=0,
    use_manual=True  # Use this if output_attentions returns None
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
    
    if use_manual:
        # Use manual extraction
        all_attentions, _ = get_attention_weights_manual(model, tokenizer, text)
        attention = all_attentions[layer_idx][0].cpu().numpy()
    else:
        # Get attention weights
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
    use_manual=True  # Use this if output_attentions returns None
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
    
    if use_manual:
        # Use manual extraction
        all_attentions, _ = get_attention_weights_manual(model, tokenizer, text)
        attention = all_attentions[layer_idx][0].cpu().numpy()
    else:
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
    use_manual=True  # Use this if output_attentions returns None
)

# %%
