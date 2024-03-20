# LLM - Large Language Models

## Introduction

They are a subset of Deep Learning models. They also intersects with Generative AI, both are a subset of deep learning, but they are not the same.

## What are Large Language Models?

They are pre-trained models that can generate human-like text. They are trained on a large corpus of text data and can learn and predict words and sentences. 

### What pre-trained and fine-tuned means?

A simple model is trained for a simple and specific task, like teaching a dog to sit. A pre-trained model and a fine-tuned model are trained for a more complex and general task, like teaching a dog to do tricks.

A pre-trained model is like having a dog that has already learned a variety of tricks and basic commands. It has undergone initial training in a broader and more complex task, such as learning to respond to various commands and perform different tricks.

A fine-tuned model is like taking that dog that already knows many tricks and further training it for a specific task or refining it to perform better in a specific set of commands or contexts. For example, it could involve teaching the dog a specific trick it didn't know before or improving its ability to perform a particular trick in a specific way.

## Porpuses

They can be used for a variety of purposes, such as text classification, question answering, document summarization, and more.

They are also used to solve specific problems in different fields, like retail, finance, entertainment, and more.

## Acronyms explained 

### "L - Large"

It refers two things:

- The size of the model, which is usually very large, like petabytes of data.

- Parameters count, which is usually very high, like billions of parameters.

### "L - Language" 

Refers to the universality of human language and resource constraints, indicating that these models are designed to handle the complexity and diversity of natural language while also considering computational resource limitations.

### "M - Models" 

Refers to pre-trained models capable of generating text similar to human text, i.e., the large language models discussed earlier.

## Benefits

- A single model can be use for a variety of tasks.

- Obtain a decent performance with a small amount of data. They can be use for a few-shot learning (training a model with minimum data) or zero-shot learning (implies that a model can recognize things that have not explicity been taught).

- The performance is improving with more data and more computational resources.

## Transformer Models

![Transformer model](/1.png)

They work as the following:

1. They receive an input sequence of tokens. Those tokens are the words of a sentence.

2. They encode the input sequence into a sequence of hidden states.

3. They decode the sequence of hidden states into an output sequence of tokens.

4. The output sequence of tokens is the predicted sentence.

Following the image we can se a example of a transformer model. The input sequence is "Optimus Prime is a cool robot." and the output sequence is some japanese text. The model is translating the input sequence into the output sequence.

--- 

In LLM all that is need is think about prompt design. Using pre-trained API can is useful because there is no need to train a model from scratch.

## What are Prompt and Prompt Engineering?

Prompts design involve instructions and examples that are used to guide the model to generate the desired output. They are used to guide the model to generate the desired output.

Prompt engineering is the process of designing and refining prompts to achieve the desired output. It involves experimenting with different prompts and examples to find the best combination that produces the desired output. This practice is based of developing and optimizing prompts to achieve the desired output.

## Three types of LLM

### Generics Language Models

- They job is to predict the next word in a sentence based on the previous words. The next word is a token based on the training data. Example: Copilot

### Instruction Tuned 

- Training to predict a response to a specific instruction. Example: ChatGPT

### Dialog Tuned

- Trained to have a dialog by predicting the next response. Example: Chatbot

## Tuning and Fine-Tuning

### Tuning

- It is the process of adjusting the parameters of a model to improve its performance on a specific task. It involves training the model on a specific dataset and adjusting the parameters to achieve the desired performance.

### Fine-Tuning

- It is bringing your own dataset and retrain the model on it. This requires a lot of computational resources and time.