# SC4001 Project - Sentiment Analysis 

Welcome to our Sentiment Analysis Project! In this project, we explore and implement three different approaches to perform domain adaptation for sentiment analysis.

## Approaches
### 0: Embeddings Visualization
To visualize the word representations from BERT, we randomly selected a set of common finance-related words, tokenized them, and individually passed them through the model’s embedding layer without any context to extract their embeddings. Please refer to the `Embeddings Visualization` notebook for more details.

### 1: Zero-Shot Learning via Prompt Engineering
In this approach, we leverage the OpenAI API's conversational model to perform sentiment classification. Detailed instructions are provided in the `Prompt Engineering` notebook. To execute the code within the notebook, you'll need an OpenAI account with a valid API key. Once you have your API key, insert it into the code as shown below:
```python
# Set up auth key and initialize the API
os.environ['OPENAI_API_KEY'] = ""
```

### 2: Domain-Adversarial Neural Network (DANN) with BERT
In this approach, we deploy a supervised DANN with BERT for domain adaptation. There are two related notebooks: `DANN Tuning`, which contains the code to optimize hyperparameters using Optuna with cross-validation, and `DANN`, which is used to train the model based on the best hyperparameters.

### 3: Sequential Fine-Tuning with BERT
In this approach, we focus on performing sequential fine-tuning with BERT for domain adaptation. We demonstrate how to adapt a model to a new domain by fine-tuning it successively with datasets from different domains. To run the script, follow the instructions in the notebook: `Sequential Fine-Tuning`.

## Side Note
You may need to modify the file paths in the notebooks accordingly to ensure that the code runs properly.