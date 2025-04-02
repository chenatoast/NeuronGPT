# NeuronGPT: Transformer Implementation from Scratch

## Introduction
Hello. NeuronGPT is a minimal yet effective implementation of a Transformer-based language model using **PyTorch**. It is built using the very basics of the transformer model. Inspired by the famous paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), this project was built to explore the inner workings of modern NLP models. YouTube tutorials by 
Andrej Karpathy and research papers helped get this implementation.

This project generates text character-by-character using Shakespeare's works as training data in input.txt. It includes two files:
1. **Bigram Model** - A simple character-level model using embeddings.
2. **Transformer Model Blocks and Functionalities** - A scaled-down version of a Transformer with multi-head self-attention.

---

## Model Architecture

Below is a simplified Transformer diagram (borrowed from "Attention Is All You Need") to visualize its working:


![image](https://github.com/user-attachments/assets/74b916c2-ec75-4034-ac73-82248f9b6a1f)

- **Embedding Size:** 256
- **Number of Attention Heads:** 8
- **Number of Layers:** 6
- **Hidden Dimension in Feedforward Layer:** 1024
- **Dropout Rate:** 0.1
- **Sequence Length:** 128
- **Vocabulary Size:** 65 (for character-level modeling)



- The **Bigram Model** only considers two-character dependencies.
- The **GPT-like model** implements **self-attention** with multiple layers to build deeper representations of text.

---

## Concepts Used
Here’s a breakdown of key components used in this project:

### 🔹 Attention Mechanism
At the heart of the Transformer model is **self-attention**, which allows the model to weigh the importance of different parts of the input while making predictions. Specifically, we implemented **multi-head self-attention**, calculated as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

where:
- \( Q, K, V \) are query, key, and value matrices,
- \( d_k \) is the dimension of key vectors,
- Softmax ensures attention weights sum to 1.

### 🔹 Add & Norm (Residual Connections + Layer Normalization)
Transformers use **residual connections** to allow gradients to flow better during training and **layer normalization** to stabilize the learning process. This helps the model converge faster and perform better.

### 🔹 Token Embeddings & Positional Encoding
Since Transformers don’t inherently understand order, we added **positional encodings** so that the model knows where each token appears in a sequence:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
$$
$$
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
$$

where \( pos \) is the position and \( d_{model} \) is the embedding size.

### 🔹 Feedforward Networks
Each Transformer block includes a **fully connected feedforward network** that helps refine the representations learned by the attention mechanism:

$$
FFN(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

where:
- \( W_1, W_2 \) are learned weight matrices,
- \( b_1, b_2 \) are biases,
- ReLU activation is used for non-linearity.

### 🔹 Dropout
To prevent overfitting, **dropout layers** with a rate of **0.1** are used throughout the model, which randomly deactivates some neurons during training.

---

## Training & Text Generation
- **Dataset:** Shakespeare's text.
- **Batch Size:** 64
- **Learning Rate:** 3e-4
- **Optimizer:** AdamW
- **Loss Function:** Cross-Entropy Loss
- **Number of Training Iterations:** 50,000

The model generates text **character by character** based on learned patterns.

### Sample Generated Text:
```
Thou art to me the fairest in my sight,
And in thy presence, all the world is bright.
```
*Note: The output gets better with more training!*

---
