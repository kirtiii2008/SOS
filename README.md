# SOS

The Real_sentiment.py file has the sentiment model which analyse the movie reviews and label them as postive/negative (1/0).
In mid term submission i built a **toy_sentiment** with my own vocab of 7-8 words and own training data.
After that i took some time to scale it over **imdb** datset and then moved to **Transformers**.

I have tried many things to optimise it not to overfit but it still does after 8-9 epochs. I ran it on goggle collab using t4-gpu because it was taking a veryy long time on vs code. Some of the optimisation done was intially I was using each and every word in the review as a candidate for vocab but their might be some words which are typos or some might be moive specific like some character so only words which appear atleast 10 times are used in vocab which incresed the accuracy.

Did regularisation by adding a dropout layer before the final linear layer and after lstm output, using packed_pad_sequence instead of pad_sequence.

The graph of test,train accuracy vs epochs is shown in the graph.png also running this code will anyways generate it.

# Tokenization

Tokenization is an essential NLP technique that involves breaking down a larger stream of text into smaller textual units called **tokens**. These tokens can represent words, subwords, characters, or even phrases, depending on the level of decomposition required.

Tokenization is necessary because machine learning models cannot understand raw text directly. The text must first be converted (mapped) into numerical representations before being fed into the network.

In the sentiment analysis model I built in Week 3, I used **Whitespace Tokenization**, which simply splits the text using spaces as delimiters and then assigns every word a numerical ID before processing it through the model.

---

# Embeddings

An embedding is a method for representing words as dense vectors. A word embedding projects textual data into a lower-dimensional continuous vector space, where words with similar semantic meanings are mapped to mathematically close coordinates.

After training, each embedding vector stores semantic and syntactic information about its corresponding word.

## Measuring Semantic Similarity

The semantic similarity of two words can be estimated by taking the **dot product** of their corresponding embedding vectors.

- **High positive dot product:** The vectors point in similar directions, meaning the words are closely related (e.g., *boy* and *man*).
- **Near-zero dot product:** The vectors are nearly orthogonal, meaning the words are largely unrelated (e.g., *apple* and *train*).
- **Negative dot product:** The vectors point in opposite directions, indicating contrasting structural contexts.

This geometric relationship forms the basis of the Transformer's **Self-Attention** mechanism, where dot products are used to determine how strongly one word relates to every other word in a sequence.

---

# Transformers

Transformers are significantly faster and more efficient than LSTMs/RNNs because they process the entire sequence **in parallel** rather than sequentially.

In an LSTM, if the input sentence contains 100 words, the network processes them one at a time. It computes the hidden state and cell state for the first word, then moves to the second word, and so on until the end of the sentence.

This sequential computation is slow and also introduces another limitation: as the sequence becomes longer, the model gradually forgets information from the beginning of the sentence, reducing the quality of its predictions.

Transformers solve this problem by using an **Encoder-Decoder architecture**, allowing every token to interact with every other token simultaneously.

---

## Structure of an Encoder

The encoder consists of a stack of identical encoder blocks.

Each encoder receives the output of the encoder below it (except for the first encoder, which receives the input embeddings directly).

Every encoder contains two main components:

- **Multi-Head Self-Attention**
- **Feed Forward Neural Network (FFN)**

---

## Data Processing inside an Encoder

The input sentence is first tokenized and then converted into embedding vectors.

These embeddings are passed into the encoder, where they first go through the **Self-Attention layer**.

Unlike LSTMs, which maintain a hidden state while processing tokens sequentially, self-attention allows every word to directly look at every other word in the sentence to gather contextual information.

Each attention head contains three learnable matrices:

- **Query Matrix (Wq)**
- **Key Matrix (Wk)**
- **Value Matrix (Wv)**

The input embedding matrix is multiplied by these matrices to produce the Query, Key, and Value vectors for every token.

Intuitively:

- **Value vectors** store the actual information about a word.
- **Query vectors** ask what information the current word needs.
- **Key vectors** describe what information every word contains.

---

### Calculating Attention Scores

The next step is computing attention scores.

This is done by multiplying the Query matrix with the transpose of the Key matrix.

For example, for Word 1, we compute:

```
q₁·k₁
q₁·k₂
q₁·k₃
...
q₁·kₙ
```

where **·** represents the dot product.

These scores indicate how strongly Word 1 is related to every other word in the sentence.

- Larger values indicate stronger relationships.
- Smaller values indicate weaker relationships.

---

### Scaling

The attention scores are divided by

\[
\sqrt{d_k}
\]

where \(d_k\) is the dimension of the Key vectors.

This prevents extremely large dot products, leading to more stable gradients during training.

---

### Softmax

A **Softmax** function is applied over the attention scores.

The resulting values

```
p₁, p₂, ..., pₙ
```

represent probabilities that sum to 1.

Intuitively, after training, the largest attention score for a word is often its own score (qᵢ·kᵢ), although this is **not guaranteed**. The model is free to attend more strongly to another word if it helps solve the task.

---

### Final Attention Output

The output vector is computed as

```
p₁v₁ + p₂v₂ + ... + pₙvₙ
```

where each probability weights its corresponding Value vector.

Intuitively, this combines information from all words into a new representation that captures contextual information for the current token.

This computation is performed for every token **in parallel** using matrix multiplication.

The final output of the attention layer is called the **Z matrix**, whose shape for a single sequence is

```
(sequence_length, value_dimension)
```

<img width="1406" height="684" alt="image" src="https://github.com/user-attachments/assets/417b8698-72d1-49e1-a4ac-1fe7c78b56e7" />

---

## Multi-Head Attention

Instead of using only one set of Query, Key, and Value matrices, Transformers use multiple independent sets.

Each set computes its own attention output, producing matrices

```
Z₁, Z₂, ..., Zₘ
```

where **m** is the number of attention heads.

Typically, Transformers use **6–12 attention heads** (larger models often use many more).

These outputs are concatenated to form a matrix of shape

```
(sequence_length, number_of_heads × value_dimension)
```

Finally, this concatenated matrix is multiplied by another learnable projection matrix **Wₒ** to project it back to the original hidden dimension.

This allows the model to combine information learned by all attention heads into a single representation.

<img width="1922" height="1060" alt="image" src="https://github.com/user-attachments/assets/82ae408d-185d-4df5-8b87-6984ef2cb03f" />

---

## Positional Encoding

Since Transformers process all tokens simultaneously, they have no inherent understanding of word order.

Positional Encoding solves this by adding a position vector to every embedding.

```
x₁ → x₁ + p₁
x₂ → x₂ + p₂
...
xₙ → xₙ + pₙ
```

These position vectors follow a carefully designed mathematical pattern that enables the model to learn relative positions and distances between words.

The intuition is that adding positional information modifies the embeddings before they are projected into Query, Key, and Value vectors, allowing attention to consider both semantic meaning and word order.

---

## Layer Normalization (LayerNorm)

LayerNorm normalizes the hidden features of every token independently so that they have approximately zero mean and unit variance.

This stabilizes training and helps prevent exploding or vanishing gradients as information flows through many encoder layers.

---

## Feed Forward Network (FFN)

After the **Multi-Head Self-Attention** layer, every token has gathered contextual information from the rest of the sequence.

However, attention mainly performs **information exchange** between tokens. It does not perform complex nonlinear computation on each token individually.

The Feed Forward Network is simply a two-layer fully connected neural network applied **independently** to every token.

First, the token representation is projected into a higher-dimensional space.

Then it is projected back to the original hidden dimension.

The expansion gives the network greater expressive power and allows it to learn more complex feature transformations.

Without this expansion, the model would have significantly less capacity.

<img width="1482" height="1186" alt="image" src="https://github.com/user-attachments/assets/44e38e63-f9a8-4d80-bec6-026874756411" />

---

# Final Intuition of an Encoder

- **Self-Attention:** Every token communicates with every other token.
- **Feed Forward Network:** Every token independently processes the information it has just gathered.

## Strcuture of a decoder

The following steps repeat the process until a special symbol is reached indicating the transformer decoder has completed its output. The output of each step is fed to the bottom decoder in the next time step, and the decoders passes their decoding results just like the encoders did. and just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs to indicate the position of each word.

The self attention layers in the decoder operate in a slightly different way than the one in the encoder

In the decoder, the self-attention layer is only allowed to attend to earlier positions in the output sequence. This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation.

The “Encoder-Decoder Attention” layer works just like multiheaded self-attention, except it creates its Queries matrix from the layer below it, and takes the Keys and Values matrix from the output of the encoder stack.

