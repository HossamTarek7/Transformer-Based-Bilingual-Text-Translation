# Transformer-Based Bilingual Text Translation

## Overview

This repository provides a comprehensive implementation of a transformer-based model tailored for bilingual text translation, built on the foundational principles established in the "Attention is All You Need" paper. The model is designed to effectively handle translation tasks between source and target languages by leveraging a robust encoder-decoder architecture with advanced attention mechanisms.


### Key Design Features

- **Modular Architecture**: The model is constructed with a modular design, ensuring that components such as the encoder, decoder, and attention mechanisms can be easily modified or extended. This design choice promotes flexibility and makes the model adaptable to various translation needs and improvements.

- **Attention Mechanisms**: Inspired by the "Attention is All You Need" paper, the model employs self-attention and cross-attention mechanisms. These mechanisms allow the model to focus on different parts of the input sequence and align it effectively with the target sequence, improving translation accuracy and contextual understanding.

- **Custom `BilingualDataset` Class**: The repository includes a custom PyTorch `BilingualDataset` class designed to manage bilingual text data efficiently. This class supports both single-source text processing and bilingual text pairs, facilitating diverse use cases such as training and evaluation of translation models.

- **Preprocessing Pipelines**: The project incorporates robust preprocessing pipelines that handle tokenization, padding, and special tokens ([SOS], [EOS], [PAD]). These pipelines ensure that text data is correctly prepared for model input, maintaining consistency and compatibility throughout the training and evaluation phases.

- **Causal Mask Function**: The implementation features a causal mask function that ensures proper sequence generation during training. This function prevents the model from attending to future tokens, adhering to the autoregressive nature of sequence-to-sequence tasks.

- **Scalability and Performance**: Designed with scalability in mind, the model can be trained on large datasets.


- **Flexible Configuration**: The model configuration is designed to be flexible and easily adjustable through a configuration file. This allows users to experiment with different hyperparameters, model architectures, and training settings without modifying the core code.

Overall, this project aims to provide a flexible and high-performance solution for bilingual text translation, leveraging state-of-the-art techniques in the field of natural language processing (NLP) to deliver accurate and meaningful translations. The modular design, comprehensive preprocessing, and robust evaluation tools make it a valuable resource for both research and practical applications in machine translation.
## Performance

- **Loss**: The model achieved a loss of 1.6.
- **Performance on Long Sentences**: The model performs exceptionally well on long sentences, averaging between 30-40 words, demonstrating its ability to handle complex and lengthy text effectively.

## Dataset

The dataset used for training and evaluation can be accessed at [Dataset Link](https://huggingface.co/datasets/stas/wmt14-en-de-pre-processed). Please ensure to download and preprocess the data as described in the repository.

## Examples

Below are examples of German sentences and their translations generated by the model:

### Example 1
**Source:** Die Politiker diskutieren über neue Gesetze.  

**Predicted:** The politicians are discussing new laws.

### Example 2
**Source:** Die Wahlen sind nächstes Jahr  

**Predicted:** The next year ' s elections are next year.

### Example 3
**Source:** Es ist eine Schande, daß es uns nicht gelingt, den Rassismus und die Fremdenfeindlichkeit in der Europäischen Union auszumerzen, die in meinem eigenen Land ebenso weit verbreitet sind wie in der gesamten EU.  

**Predicted:** It is a disgrace that we have not managed to eradicate racism and xenophobia in the European Union, a policy that is as widely as many as possible in my own country, as throughout the EU.

### Example 4
**Source:** Die Umweltpolitik spielt eine immer wichtigere Rolle, da der Klimawandel eine der größten Herausforderungen unserer Zeit darstellt und nur durch internationale 
Zusammenarbeit gelöst werden kann  

**Predicted:** The environmental policy is becoming more important as climate change is one of the main challenges in our time and can only be solved by international cooperation.

### Example 5
**Source:** Die Regierung plant, in den nächsten Jahren mehr in Bildung und Gesundheit zu investieren, um sicherzustellen, dass alle Bürger Zugang zu den besten Möglichkeiten haben und gleichzeitig die wirtschaftliche Stabilität des Landes zu gewährleisten.  

**Predicted:** The government intends to invest more in education and health over the next few years in order to ensure that everyone can access the best opportunities that are best placed and guarantee the economic stability of the country.

### Example 6
**Source:** Angesichts der fortschreitenden technologischen Entwicklungen und der dringenden Notwendigkeit, den Klimawandel zu bekämpfen, steht Deutschland vor der komplexen Aufgabe, seine industrielle Produktion zu modernisieren, den Übergang zu erneuerbaren Energien zu beschleunigen und gleichzeitig sicherzustellen, dass die soziale Gerechtigkeit und der Wohlstand der Bevölkerung nicht gefährdet werden, während es sich in einem zunehmend globalisierten Wettbewerb behauptet.  

**Predicted:** Given the constant technological developments and the urgent need to combat natural changes in the complex world, Germany's task of avoiding the establishment of its industrial production, to speed up the transition to renewable energies, and, at the same time, to ensure that social justice and prosperity are not put at risk, as a result of increasingly widespread competition in a globalized country.

## Requirements

- Python
- PyTorch
- Tokenizers
- tqdm
- numpy
- spacy
- datasets
