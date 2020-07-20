Sentiment classification using BERT pre-trained model

Bidirectional Encoder Representations from Transformers (BERT) are used to to perform one of the downstream NLP tasks, sentiment classification. The objective is to classify tweets into positive, negative, and neutral based on the semantics in the tweets.

The reasons for using BERT pre-trained representations and fine-tuning it for the given task are as follows: 

1. Pre-trained BERT model is one of the state-of-the-art models, and expected to give the best performance by fine-tuning.
2. Faster training, training ~15K samples took around 6 minutes of Tesla K80 GPU time. The available pre-trained BERT model weights encode a lot of information and only 2-4 epochs of fine-tuning can yield better performance, and saves a lot of time. On the contrary, training LSTM model/BERT from scratch requires more computation resources and time.
3. Bidirectional, parallel functioning in Transformers, Self-attention mechanism for better context capturing.


There are multiple interfaces present for various purposes viz. BertForNextSentencePrediction, BertForQuestionAnswering etc. For this task, I would be using BertForSequenceClassification from the ‘transformers’ library  from “Huggingface”. Basically, BertForSequenceClassification model is built on top of a trained BERT model. All that is required to be done is it’s fine-tuning for our task, i.e. fine-tuning for sentiment classification problem. Using Bert base Uncased model. For architectural details and working of the BERT model, please refer to the research paper (https://arxiv.org/abs/1810.04805).


# Methodology:

# 1. Preprocessing the text data:

Explaining the details on the preprocessing steps are beyond the scope of this work. However, presenting the major steps in order to clean and understand the data.
# Data cleaning:
Includes special characters, numbers, punctuation, short words, tweeter handle symbol @; Sanity checks for tweets size,Tokenization and Stemming.
# Exploratory data analysis (EDA): 
Understanding common words in the dataset for each of the sentiment present. Word cloud gives an understanding of the frequent words for a sentiment. Checking # examples or in each class.

# 2. BERT modeling

The “Transformers” library from “Huggingface” is used to perform the downstream sentiment classification task. Following are the steps taken in the modeling process:

i. BERT input formatting: Various steps like Tokenization, converting tokens to ids, padding, special character insertion, attention masks, etc. are done using BERT packages. These packages ease out the efforts of formatting the input as required by the BERT fine-tuning process.

	# Following code snippet shows working of a BERT tokenizer 
	from transformers import BertTokenizer
	# Load the BERT tokenizer.
	print('Loading BERT tokenizer...')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)      
	# Having a look at the original sentencel, its tokenized version done by BertTokenizer and the 	#Token ids for all the tokens.
	print('Original: ', sentences[0])
	print('Tokenized: ', tokenizer.tokenize(sentences[0]))
	print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))

	Output:
	Original: What said
	Tokenized: ['what', 'said'] 
	Token IDs: [2054, 2056]

ii. Finding the maximum length of the sentence in training dataset: BERT  tokenizer function encode_plus requires the max_length argument to be provided. Based on which it pads or truncates the sentence. The maximum length of a sentence in training data is 32. So, setting the max_length parameter to 32.

    # Following code snipped demonstrates finding the maximum length of sentence
	max_len = 0
	# For every sentence...
	for sent in sentences:
		# Tokenize the text and add `[CLS]` and `[SEP]` tokens.
		input_ids = tokenizer.encode(sent, add_special_tokens=True)
		# Update the maximum sentence length.
		max_len = max(max_len, len(input_ids))
	print('Max sentence length: ', max_len)

   Output:
   Max sentence length: 32

iii. Encoding the sentences using encode_plus:`encode_plus` will:

	(1) Tokenize the sentence.
	(2) Prepend the `[CLS]` token to the start.
	(3) Append the `[SEP]` token to the end.
	(4) Map tokens to their IDs.
	(5) Pad or truncate the sentence to `max_length`	
       (6) Create attention masks for [PAD] tokens.

   Output:
   Original: What said 
   Token IDs: tensor([ 101, 2054, 2056, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
   Attention mask tensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

iv. Training:
Using BertForSequenceClassification from Transformers.

# 3. Inference code
Once we get a descent accuracy on the validation set, we want to make use of the trained model for predicting the sentiment of a given text. For this, an inference code for classifying real-world data is provided. In other words, using the trained model to classify a sentence(for which no label is present) to a positive, negative or neutral sentiment.
