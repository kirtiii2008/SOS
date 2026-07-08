# SOS

## Tokenization

Tokenization is an essential NLP that involves breaking down a larger stream of text into smaller textual units, called tokens which can be in various forms. These tokens can range words or phrases, depending on the level of decomposition required. Tokenization is neccessary beacuse ML models don't understand direct text. It must be convert/mapped to numbers before feeding it into the network.
In the sentiment analysis model I made in Week 4 I used **Whitespace Tokenisation** which is essentially splitting the text using space as delimiter and then assigning each word a numerical vector to process it into the model.

## Embeddings

An embedding is a method for representing words as vectors. A word vector projects textual data into a lower-dimensional continuous space, where words with similar semantic meanings are mapped to mathematically close coordinates.
After a model is trained, each vector of real numbers inherently stores semantic and syntactic information about its corresponding word.

### Measuring Semantic Similarity
The semantic **similarity** of two words can be mathematically verified by taking the **dot product** of their corresponding embedding vectors:

* **High positive dot product:** Indicates that the vectors point in a similar direction, meaning the words are highly related or share a context eg. ("boy" , "man").
* **Near-zero dot product:** Indicates orthogonality, meaning the words are entirely unrelated ("apple" and "train").
* **Negative dot product:** Indicates the words hold contrasting or opposite structural context in the vector space.
  
This geometric relationship is the foundation of the Transformer's **Self-Attention layer**, where dot product attention is calculating exactly how much a specific word relates to every other word within a given sequence.

##Transformers

The transfomrers are fatser and efficient than LSTMs/RNNs beacuse in LSTMs the text/data is processes step by step or they have sequential processing, like in LSTM if we input a sentence of length 100 the LSTM would satrt from word 1 compute the hidden state, cell state then go to word 2 compute its cell & hidden state and so on. This is a very time consuming process and one more drawback is if the data is too long the LSTMs "foregts" the context as it travels along the end of data and therefore decreasing the quality of outputs of LSTMs.

Transformers uses concept of encoding component(stack of encoders) and decoding component(statck decoder layers). 
###Strcuture of Encoders
Each encoder is stacked over each other the input from one encoder is the output of the one below it except for the lowest encoder which recevies input from our side. The encoder has two parts within it a **Self Attetion Layer** and a **Feed Forward Neural Network**.

###Data processing in Encoders

Firstly the input is broken into tokens and then embedded ie converted to tensors(array of real number) and feed into the encoder where it goes through the self-attention layer. As the model processes each word self attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word. In LSTMs/RNNs it was done by maintaining a hidden state while processing words.
Each encoder in the encoding layer has a set of three matrices (W_v,W_q,W_k) that are (value matrix,query matrix,key matrix). Now in the first step input matrix where each row represnts the tensor corresponding to each word is multipled with these three matrices to get value,key,query vectors for each token/word in the input.

The intutuion is that value vector of a word stores the actual inforamtion about that word while the key and query vector are used to calculate the relationship of this word with rest of the sentence.The second step in calculating self-attention is to calculate a score. To do this we multiply the matrix if query vectors with the transpose of key vectors. what we are essentially doing here is that suppose for word 1 we have (k1,q1) the query and key vector, we calculate k1.q1,k2.q1,k3.q3.....kn.q1 where **.** represent the **dot product**. These values tells us that how related is the word1 to all the other words in the sentence. A higher dot means they resembles closely, similarly for zero and negative dots mentioned in the embedding section.

Nextly we divide the scores by the square root of the dimension of the key vectors. This leads to having more stable gradients. After this we apply a **softmax** function over all the n scores we get where n is the number of words in the sentence. This converts all these scores into probablities which sum to 1 (p1,p2,...pn) . Intutively we can say that for ith word the maximum score will be ki.qi after the model is trained because the word will resemble to itself the most in that sentence.

Note that the key,query and value vector all have same dimension otherwise we can't use dot product. 

Now the final output for the input vector will be calculated by multiplying these probablities with the corresponding value vectors ie p1(v1)+p2(v2)....pn(vn) which is "intutively" realting all the words in the sentence to store context. It's just intutution no one actually knows what is the real reason for doing this.

This whole process is carried out for all the input tensors in parallel by matrix mutiplication. Below image shows what actually happens in the attention layer within an encoder in a transformer.

<img width="1406" height="684" alt="image" src="https://github.com/user-attachments/assets/417b8698-72d1-49e1-a4ac-1fe7c78b56e7" />




