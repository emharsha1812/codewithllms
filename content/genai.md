---
title: Generative AI From theory to product launch
---

#### Encoder Decoder Architectures

These encoder-decoder architectures, such as **variational autoencoders** **(VAEs)**, became a popular choice in applying probabilistic inference to encode the data into a representational latent space while imposing reasonable constraints for smooth interpolation and manipulations

![[images/genai/Pasted image 20240525215935.png]]
Following is a brief description of each model and its use cases:

- **(A) Convolutional Neural Network:** A trained CNN on a large-scale image dataset with object categories takes an image as input and runs through a series of convolutional stages to predict the object categories as output. The feature processing in a convolutional layer is performed by a large bank of tuned filters/kernels that generate a set of feature maps as a forward pass for the next layers in the network.
- **(B) Recurrent Neural Network:** A trained CNN-RNN architecture takes a series of input frames as a visual input, and runs them through a series of non-linear convolutional and recurrent stages, potentially predicting the next frame as well as the stimulus category. This type of network is robust in learning the spatiotemporal structures in the input.
- **(C) Variational Autoencoder:** A variational autoencoder (VAE), with its encoding-decoding architecture, is a powerful framework for learning latent structures in a given dataset. Once trained, the encoder of this model can generate smart embeddings through its dimensionality reduction feature. VAEs and their variants can also facilitate applications for image reconstruction or data augmentation.
- **(D) Generative Adversarial Network:** A generative adversarial network (GAN) consists of two modules: 1) generator and 2) discriminator. The generator is mostly an encoder-decoder-based architecture that learns the features in a given dataset during training and generates similar images that are detected by the discriminator module. The two modules compete with each other, and finally, a trained GAN is able to generate fake samples of a given image category that a discriminator is unable to distinguish from the real samples. GANs are widely used in applications such as image generation, augmentation, and segmentation.
- **(E) Transformer:** A **Vision Transformer (ViT),** inspired by the powerful Transformer in NLP, is a self-attention-based architecture. The input image is distributed into nine (flattened 2D) patches, and linear embeddings of these patches are fed as an input to the encoder of the trained ViT. The image patches are embedded as tokens. The encoder block consists of several layers. Each layer begins with a normalization step, followed by a multiheaded layer with self-attention. Furthermore, a **multilayer perceptron (MLP)** with a single hidden layer is used as a classification head that predicts the object categories present in the input image. ViT and similar architectures are widely applicable for image and video classification tasks. When paired with VAEs, these models are also used to learn multisensory data. For example, text-image pairs can be used to generate novel images from text descriptions as well as apply transformations to the existing images.

Beyond transformers, this revolution in generative models resulted in the development of AI systems with real-world efficacy that could create new texts, images, videos, and other multimodal content. It also captured the widespread interest of both machine learning researchers and engineers, to push the boundaries of this field and create the world of Generative AI—not just predict it. A language model (ChatGPT) by OpenAI, an image generation model (Stable Diffusion) by Stability AI, and a video generation model (Gen-2) by Runway Research are a few examples of content creation-based products in the field of GenAI.

## Overview of generative AI systems

The User Interface module captures the input command as text prompts through a web/mobile running application. The GenAI module processes the commands through a natural language processor and generates the response command as desired by the user in the form of text, image, or video content.

![[images/genai/Pasted image 20240525220241.png]]

## Transformer Networks

### Sequence-to-sequence (seq2seq) modelling

Recurrent architectures such as RNNs have long dominated **seq2seq modeling**. These model architectures process the sequences, such as text, iteratively (i.e., one element at a time and in order). This sequential handling imposes a challenge when the model needs to learn long-range dependencies due to rising issues such as vanishing gradients. As the gap between relevant token elements increases, these models tend to lose track of learned sequences from early time steps, resulting in incomplete context understanding, which is highly necessary for language learning.

Let’s take a look at an example: “The cat that the dog chased ran up a tree.” This sentence contains long-range dependencies between the earlier (e.g., cat) and later (e.g., ran) words. The RNN will process this sentence iteratively (i.e., token-by-token) and needs to learn the long-range dependencies. In this case, the RNN may not be able to connect the relationship between “cat” and “ran” together since several words are present in between.

To solve this problem, how about we design a model that can process the entire sequence “The cat that the dog chased ran up a tree” in parallel and capture the relationship between all pairs of tokens in the given sequence—simultaneously. This is precisely what the **Transformer model** does. It models long-range dependencies across the entire sequence using the self-attention mechanism and computes the relationship between all pairs of tokens via dot-product attention.

## Transformers

The transformer architecture was introduced in the paper “Attention is All You Need.” This directly represents a departure from the sequential processing paradigm of previous models like RNNs and CNNs. The transformer relies on the concept of **self-attention**, a mechanism that allows the model to weigh the importance of different parts of the input data when making predictions. Self-attention mechanism computes attention scores between all elements in an input sequence (𝑋). Consider an input sequence represented as a set of vectors:

$X=x_1,x_2,x_3...x_n$

The self-attention operation computes the attention scores, also referred to as attention weights, for each element (i.e., token) in the sequence with respect to all other elements. These attention scores are computed using the dot product of the query (𝑄) and key (𝐾) vectors, followed by a softmax operation to normalize the scores:

$Attention(Q,K)=softmax(QK^T/\sqrt{d_k})V$

Here, 𝑄, 𝐾, and 𝑉 are linear projections of the input vectors learned during the training of the model and 𝑑𝑘dk​ is the dimension of the key vectors. The softmax operation is applied to normalize the attention scores in order to create a probability distribution ranging from 0 to 1. Transformers apply self-attention through powerful **multi-head attention** mechanisms. Instead of learning a single set of attention weights, the model learns “heads”—each capturing different aspects of the input data, whether a word token or an image patch. These heads operate in parallel; afterward, the prediction outputs are concatenated and transformed in a linear fashion to produce the final output. This allows the model to attend to different parts of the input sequence/patches simultaneously, enhancing its ability to capture complex patterns and relationships in the presented data

### Vision transformers

While transformers were initially designed for sequential data like text, their success in NLP led to their adaptation for computer vision tasks like image classification. Vision transformers (ViTs) apply the transformer architecture to the image data by splitting an image into smaller non-overlapping patches, which are treated as the input sequence for the transformer model. The transformation of images into sequences of patch embeddings allows transformers to leverage their self-attention mechanism to capture long-range dependencies in images.

ViTs have shown remarkable performance in tasks like image classification and object detection, outperforming convolutional neural networks (CNNs). The figure below shows the explained mechanism in the ViT. The input image of a man in a black suit is distributed into nine (flattened 2D) patches, and linear embeddings of these patches are fed as input to the encoder of the trained ViT. The image patches are embedded as tokens. The encoder block has several multi-headed layers with self-attention, along with a normalization layer, at the start of each layer. Furthermore, a multilayer perceptron (MLP) with a single hidden layer is used as a classification head that predicts the object categories present in the input image. ViT and its variant architectures are widely applicable for image and video classification tasks. When paired with VAEs, these models are also designed to learn multisensory data. One example is text-image pairs, which are used to generate novel images from text descriptions as well as apply transformations to the existing images.

![[images/genai/Pasted image 20240525221801.png]]

### Multimodal transformers

Similarly, the same architecture is generalized for a range of data modalities, such as a sequence of input audio, video, etc. The figure below shows the extension of the transformer from single to multimodal data processing, showing the efficacy of transformers as building blocks of almost every generative AI product today.

![[images/genai/Pasted image 20240525221902.png]]

A range of multimodal data is shown as inputs to the transformer architecture. This includes video, audio, and text modalities. The model architecture is generalized to generate positional embeddings of the respective modality it is selected to get trained on for a variety of downstream tasks as output.

## Text-to-Text Generation Products

Now that we have gone through the basics of the Transformer model, let’s dive deep into some amazing real-world applications of generative AI empowered by the transformers. There has been a tremendous increase in interest in building text-to-text-driven products. Imagine a language model that can be fine-tuned in a customized context and be able to work for questions and response-based tasks. Similarly, we can take a powerful language model with a given input sentence and generate a paragraph or even write an entire book. Such fine-tuned models have applications in writing product descriptions, code generation, sentiment analysis, personalized recommendations, and text summarization, to name just a few.

In this section, we will take a look at two commonly used model architectures in academia and the industry: DistilBERT for query-response tasks and GPT-2 for text generation-based tasks. As addressed before, the same models can also be fine-tuned for several other applications. The latest LLM products, such as ChatGPT by OpenAI, Claude by Anthropic, and Bard by Google, are mostly extensions of the same GPT-based model architecture with bigger parameters and huge data exposure.

## DistilBERT

**Bidirectional Encoder Representations from Transformers (BERT)** is a commonly used model for natural language processing with more than 300 million parameters. Launched in 2018 by Google, it serves applications for several of the most common language tasks, such as sentiment analysis and named entity recognition. DistilBERT is a compressed version of BERT. It also uses the transformer architecture for efficient language understanding but only has around 60 million parameters. The BERT model relies on self-attention mechanisms to capture contextual word relationships that can mathematically be expressed through attention scores computed using softmax normalization:

$A=softmax(Q.K^T)$

Here, 𝑄 is the query matrix obtained from input text embeddings and represents the values that we want to use for making the prediction. Each row of the 𝑄 matrix is a word or element in the given text sequence. Similarly, 𝐾 is the key matrix that is used to define the relationship between the word tokens in the input sequence. 𝑇 represents the transpose operation applied to 𝐾.

After training and optimizing the model, DistilBERT’s efficiency is achieved through **model distillation**, where a larger pre-trained model’s knowledge is transferred to a smaller one, allowing for faster inference while maintaining accuracy in real-time. Below is an example of fine-tuning DistilBERT on “Beginner’s Pancake Cooking Guide” and “Technical Software Engineering Interview Preparation” query-response tasks:

- Insert code here

In the examples above, we first provide the model with the context and then assign a few questions related to the given context. The model predicts the responses with a certain likelihood. Feel free to feed in additional context and frame questions related to other applications in the coding session to observe the performance of this fine-tuned model for GenAI product prototypes.

## GPT-2

GPT-1 was a pioneer in the GPT series, employing a stack of transformer decoder layers. It also had self-attention for contextual word dependencies, mathematically captured through attention scores, as we observed in the DistilBERT earlier. The softmax layers normalized these scores to create a probability distribution. GPT-1 was trained on a large corpus of publicly available text from the internet; it had several applications in text generation, completion, and summarization applications. This paved the way for many advanced models to come across various commercial applications.

GPT-2 has significantly more parameters than GPT-1, resulting in more complex embeddings and transformations, but the architecture is more or less the same. The major differences include handling features through multiple layers of multi-head, self-attention, and feedforward networks. GPT-2 was trained on a larger dataset than GPT-1 that included internet text, books, blogs, etc., leading to improved text generation capabilities. The model is commercially used for several applications, such as content generation, chatbots, and text-based games. Let’s take a look at an example of text generation using GPT-2:

```
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# Load pre-trained model tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load pre-trained model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set model to eval mode
model.eval()

# Encode context to input ids
context = "I absolutely Love my girlfriend, I think she is the most beautiful girl in the whole wide world. I think I am in"
input_ids = tokenizer.encode(context, return_tensors='pt')

# Generate text
generated = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id,max_length=200, do_sample=True)

# Decode generated text
text = tokenizer.decode(generated[0], skip_special_tokens=True)

print(text)
```

To explain how the above examples work, after installing the required libraries, we import the necessary classes—the `tokenizer` to process the text and the `model` itself, which has the pre-trained weights. Then, we create instances of the tokenizer and the small GPT-2 model. This model was trained on tons of text data to learn how to generate realistic sentences and paragraphs. Now, it’s crucial we set the model to evaluation mode with `model.eval()`. To feed text into the model, we use the tokenizer’s `encode()` method to convert the string into numeric IDs that the model understands. We specify `return_tensors = 'pt'` to give us a PyTorch tensor ready for the model input. The `generate()` method on the model takes the `input_ids` and returns the generated text continuation. We can set parameters to control how it generates text—the length, creativity, etc. Finally, we decode the numeric output back into actual text using the tokenizer’s `decode()` method. We skip the special tokens, so we just get the real generated text.

In just a few lines of code, we can load a powerful text generation model (GPT-2) and use it to continue an input sentence or paragraph. In the coding session, feel free to tweak the input, output length, and sampling to observe the change in the generated text.

After GPT-2, GPT-3 and GPT-4 were also launched with billions and trillions of parameters in their model architectures. GPT-4’s recent applications include virtual assistants, advanced content generation, and domain-specific natural language understanding, which was highly focused on personalized recommendations and several similarly commercial applications. ChatGPT by OpenAI has also recently deployed GPT-4 after training the model on more than 300 billion word tokens.

## Image-to-text generation

Transformer models have become the dominant approach for image captioning, partly due to their enhanced feature to model global context and long-range dependencies while generating coherent text. Recent state-of-the-art models leverage encoder-decoder transformers. The encoder processes the features of the image into an intermediate representation. The decoder auto-regressively generates the caption text, token-by-token, attending to relevant parts of the encoded image context at each generation (or step) through cross-attention layers.

Decoders such as GPT-2 and GPT-3 have been adapted for caption generation by modifying their self-attention masking to prevent the model from attending to future token positions during training. Vision transformers, as seen previously, are also commonly used as the encoder architectures in this schema.

The following figure shows a high-level block diagram of commonly used image-to-text model architectures:

![[images/genai/Pasted image 20240525231135.png]]

A sample image from the movie, *The Matrix,* is passed as input to a trained image-to-text model, and text tokens (i.e., captions) are generated as output.

**GIT** (with CLIP-GPT) is a high-performing model for generating descriptive tags and captions for images, a fusion of computer vision and natural language processing. The architecture consists of an image encoder, followed by the transformer encoder block. For decoding, it contains GPT-3 decoded for image-to-text tasks. It is conditioned on two types of tokens: image tokens derived from CLIP (representing visual features), and text tokens (crafting the generated caption).

The model is trained on millions of image-caption pairs that serve as the instructional backdrop, allowing the model to translate images into human-readable descriptions. The model is well suited for image and video captioning, visual question answering on images and videos as well as image classification tasks. The following figure shows the input image fed to a trained model, followed by the word-by-word image captioning:

![[images/genai/Pasted image 20240525231616.png]]
The input image of a kitten playing with a toy is shown to the image-to-text model with a word-by-word generation of captions captured through attention layers by Grad-CAM.

The following code example shows how to load the trained GIT model, feed a random image, and generate “a kitten playing with a toy” as a caption. The figure above also demonstrates the processing of model attention mechanisms on word-by-word generation of captions mapped to their image features. For example, notice the heat maps of attention layers generated by the Grad-CAM technique for each visual concept and how they are related to visual reasoning

```
from transformers import pipeline
import matplotlib.pyplot as plt
import urllib
import numpy as np
from PIL import Image
import torch

model = "microsoft/git-base" #@param ["Salesforce/blip2-opt-2.7b", "microsoft/git-base"]
model_pipe = pipeline("image-to-text", model=model)
image_path = 'cat.jpg' #@param {type:"string"}

if image_path.startswith('http'):
  img = np.array(Image.open(urllib.request.urlopen(image_path)))
else:
  img = plt.imread(image_path)
 
caption = model_pipe(image_path)[0]['generated_text']
print('Caption:', caption)
plt.axis('off')
plt.imshow(img)

pip install salesforce-lavis
import lavis
from lavis.models import load_model_and_preprocess
from lavis.processors import load_processor
from lavis.common.gradcam import getAttMap
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam



device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model, vis_processors, text_processors = load_model_and_preprocess("blip_image_text_matching", "base", device=device, is_eval=True)




def visualize_attention(img, full_caption):
    raw_image = Image.fromarray(img).convert('RGB')
    dst_w = 720
    w, h = raw_image.size
    scaling_factor = dst_w / w
    resized_img = raw_image.resize((int(w * scaling_factor), int(h * scaling_factor)))
    norm_img = np.float32(resized_img) / 255
    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    txt = text_processors["eval"](full_caption)
    txt_tokens = model.tokenizer(txt, return_tensors="pt").to(device)
    gradcam, _ = compute_gradcam(model, img, txt, txt_tokens, block_num=7)
    gradcam[0] = gradcam[0].numpy().astype(np.float32)
    num_image = len(txt_tokens.input_ids[0]) - 2
    fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))
    gradcam_iter = iter(gradcam[0][2:-1])
    token_id_iter = iter(txt_tokens.input_ids[0][1:-1])
    for i, (gradcam, token_id) in enumerate(zip(gradcam_iter, token_id_iter)):
        word = model.tokenizer.decode([token_id])
        gradcam_image = getAttMap(norm_img, gradcam, blur=True)
        gradcam_image = (gradcam_image * 255).astype(np.uint8)
        ax[i].imshow(gradcam_image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_xlabel(word)
visualize_attention(img, caption)
```

## Text-to-image generation[](https://www.educative.io/courses/generative-ai-from-theory-to-product-launch/text-guided-image-generation-products#Text-to-image-generation)

Recently, companies like Midjourney have revolutionized the image content generation space through prompt engineering. With a revenue of more than $250 million, their text-to-image product is empowered by the stable diffusion model. The following figure shows the model architecture functionality of stable diffusion. Here, the input text prompt “a man flying in the air with cars” is passed to the text encoder that maps the textual descriptions or instructions into a numerical representation, used as a conditioning input for the model.

In stable diffusion, this can potentially mean encoding text-based descriptions of desired image attributes, styles, or content. Next, the text representation is passed to the diffusion image decoder, consisting of U-Net (semantic mapper) architecture, adapted and integrated into the larger model to generate images that align with textual descriptions. The decoder is a crucial component in the architecture of diffusion models, as it directly contributes to the generation of high-quality and realistic images by iteratively denoising the noisy inputs. The resulting representation is passed to the variational autoencoder (VAE)-based decoder that generates the output image. The following figure illustrates the entire process:

![[Pasted image 20240526000624.png]]\*\*\*\*

A sample caption (text tokens), “a man flying in the air with cars” is passed as input to a trained text-to-image model and the resulting output image is generated.

There are several text-to-image model architectures in production these days. For the coding session, let’s take a look at two commonly used models: DALL·E by OpenAI and Stable Diffusion by Stability AI.
