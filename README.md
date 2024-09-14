# LLM-jh-GPT
A <ins>Generative Pre-trained Transformer</ins> (GPT) is a type of artificial intelligence that understands and generates human-like text. We will be using the <a href="https://pytorch.org/docs/stable/nn.html"><ins>PyTorch.nn</a> (Neural network) library</ins> which houses transformer architecture. The goal of jhGPT is to output linguistic text similar to the capabilities of humans. Ultimately, we want the model to produce undifferentiable text (compared to a human). The model will have a range of languages, initially starting with English, and then moving forward to other languages.


## Transformer Architecture used in jhGPT
<p align="center">
  <img src="https://github.com/Hy8012/LLM-jh-GPT/blob/main/md_files/Transformer.png?raw=true" width="400" height="675"/>
</p>

The picture above is the transformer architecture as described and depicted in <a href="https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"><i>Attention Is All You Need</i></a>. In essence, a transformer is a type of artificial intelligence model that learns and analyzes patterns in heaps of data to generate new output. Transformers are a current cutting-edge natural language processing (NLP) model relying on a different type of encoder-decoder architecture. Previous encoder-decoder architectures relied mainly on Recurrent Neural Networks (RNNs), however, Transformers can entirely remove said recurrency.
