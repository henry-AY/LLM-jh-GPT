# LLM-jh-GPT
A <ins>Generative Pre-trained Transformer</ins> (GPT) is a type of artificial intelligence that understands and generates human-like text. We will be using the <a href="https://pytorch.org/docs/stable/nn.html"><ins>PyTorch.nn</a> (Neural network) library</ins> which houses transformer architecture. The goal of jhGPT is to output linguistic text similar to humans' capabilities. Ultimately, we want the model to produce undifferentiable text (compared to a human). The model will have a range of languages, initially starting with English, and then moving forward to other languages. The majority and basis of the architecture come from Andrej Karpathy's <a href="https://github.com/karpathy/nanoGPT">nanoGPT</a> GitHub repo, however, all analyses, and text files are independent and licensed uniquely.

## Transformer Architecture used in jhGPT
<p align="center">
  <img src="https://github.com/Hy8012/LLM-jh-GPT/blob/main/md_files/Transformer.png?raw=true" width="400" height="675"/>
</p>

The picture above is the transformer architecture as described and depicted in <a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"><i>Attention Is All You Need</i></a>. In essence, a transformer is a type of artificial intelligence model that learns and analyzes patterns in heaps of data to generate new output. Transformers are a current cutting-edge natural language processing (NLP) model relying on a different type of encoder-decoder architecture. Previous encoder-decoder architectures relied mainly on Recurrent Neural Networks (RNNs), however, Transformers can entirely remove said recurrence.

### <ins>Encoder Workflow</ins>

The figure below (the left half of the transformer) is the Encoder.
<p align="center">
  <img src="https://github.com/Hy8012/LLM-jh-GPT/blob/main/md_files/Input_Transformer.png?raw=true" width="200" height="475"/>
</p>

#### Step 1 - Input Embeddings

It is important to note that the embedding process only happens in the bottom-most encoder, not each encoder. The encoder begins by converting the input into tokens--words, subwords, or characters--into vectors using embedding layers. The embeddings > capture the semantic meaning of the tokens and convert them into numerical vectors.

#### Step 2 - Positional Encoding

Because Transformers lack a recurrence mechanism such as Recurrent Neural Networks (RNNs), a mathematical approach must be applied to introduce position-specific patterns to each token in a sequence. This process is called 'Positional Encoding', where a combination of sine and cosine functions are used to create a positional vector.

##### <ins>Positional Encoding using Sine</ins>
$\ PE(\text{pos}, 2i) = \sin \left( \frac{\text{pos}}{10000 \cdot \left( \frac{2i}{d_{\text{model}}} \right)} \right) \$

```
PE\left(pos,\ 2i\right)\ =\sin\left(\frac{pos}{10000\left(\frac{2i}{d_{model}}\right)}\right)
```

##### <ins>Positional Encoding using Cosine</ins>
$\ PE(\text{pos}, 2i + 1) = \cos \left( \frac{\text{pos}}{10000 \cdot \left( \frac{2i}{d_{\text{model}}} \right)} \right) \$

```
PE\left(pos,\ 2i\ +\ 1\right)\ =\cos\left(\frac{pos}{10000\left(\frac{2i}{d_{model}}\right)}\right)
```

The equations and process of positional encoding will be further detailed and explored in <i>Fundamentals of jh-GPT - A Deep-Dive into a Transformer-Based Language Model</i>

#### Step 3 - Multi-Headed Attention



### <ins>Decoder Workflow</ins>

The figure below (the right half of the transformer) is the Decoder
<p align="center">
  <img src="https://github.com/Hy8012/LLM-jh-GPT/blob/main/md_files/Decoder_Transformer.png?raw=true" width ="200" height="675"/>
</p>
