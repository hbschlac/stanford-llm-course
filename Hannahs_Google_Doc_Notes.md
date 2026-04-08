# Hannah's Personal Notes — Stanford LLM Course

**Source:** [Google Doc](https://docs.google.com/document/d/1_wnQy_Kh8W_JZ8Jg53TxzzTgQK8tpGefnj2um95ut3c/edit)
**Last Modified:** 2026-04-08

---

# Class 1

STANFORD LLM COURSE - [https://cme295.stanford.edu/](https://cme295.stanford.edu/)

LECTURE 1 - LLM & GENAI COURSE
[fall25-cme295-lecture1.pdf](https://drive.google.com/file/d/16kHuTL6VKHX2OgaqF5F6kfZPkYA8CT95/view?usp=drive_link)
Stanford

Part 1: Overview NLP
NLP = natural language processing

* Input text
* Model (Predict)

Use case
Classification

* Sentiment extraction
* Intent detection
* Language detection
* Topic modeling

Example

* Find data sets (eg. Amazon reviews, IMBD movie reviews)
* Eval metrics
  * Accuracy: % observations correctly predicted
  * Precision: % predicted positive that were correct
    * Eg. 10 predicted positive; 6 were correctly positive (4 really negative)
  * Recall: % actually correct were correct
    * Eg. 12 are correct; only 10 were identified as positive
  * F1 score: mean of precision and recall

Multi-clasification

* Predict more than 1 thing (number of tasks)
* Name entity recognition (identify category of words)
* Part of speech tagging
* Dependency parsing
* Constituency parsing

You predict multiple things based on the input text
At a token-level, per entity type

Generation (popular these days)
Text as inputs and outputs; variable length (output text)

* Machine translation
* Question answering
* Summarization
* Text generation

Evals >> Problem - always need reference text; need to have labels (expensive! time!)

* BLEU - quality of text translated; similar to precision
  * Rule based metric
  * The higher the better
* ROUGE - quality of text generated; similar to recall
  * Suite of metrics
  * The higher the better
* Perplexity - quantifies how 'surprised' the model is to see some words together
  * Looks at probabilities output of the model
  * The lower the better

Part 2: Timeline

* Blew up in 2022; but began in 1980s
* Word2vec - compute meaningfully embeddings (2013)
* Transformers (2017) - foundation of models we see today
* LLMs (2020s) - scaled up; computing, data used for training

Part 3: Tokenization

* We want a model that understands text; however, model only understands numbers
* Eg "cute teddy bear reading" -> how do we make this sentence be able to be passed to model?
  * Arbitrary - 1 unit of text for each word
    * Each unit is called a token
    * OOV - out of vocabulary (risk)
  * Word - "bear" vs. "bears" are 2 different words but similar; BUT if we do word-level tokenization, then each word would be a different token
    * Sub-word (tokenizers) - leverage roots of words to find common roots in each word (eg. BEAR, BEARs)
      * Sequence is longer for this method
  * Character
    * With sub-word, a mis-spelled word might not be recognized; so, characters can
    * Sequence is longer, but it takes more time for model to process sequence
    * Hard to know what representation of each letter means

Part 3: Word Representation
So, we took text, we made it into tokens (units) - now we need to take representation for each unit
OOV (out of vocabulary) - you can't just use a dictionary; like new word that's used but not in dictionary; because LLM is so big the typo can still be mapped to the actual word

Vector
"One soft teddy bear" >> how do we take all these words and make them into numbers
One hot encoding (OHE)

* Every word in vocabulary is another axis
* We put an index or combination of numbers for each word
* So, 'soft' = 1 0 0 (always in that spot on 'book shelf')
  * Think of the matrix - series of equations
* Limitation is no intent or context beyond the location -> COSINE SIMILARITY
* All the vectors are same distance from one another; so we don't know the relationship between all the tokens because all same equally far away

Learned embedding

* Tokens similar (high similarity) should be together vs. non-similar tokens be orthogonal
* The 'index' location of each token (representation) is closer so that additional math on top ends up in same space
* So soft = 0.95, 0.25, 0.04
* Normalization would mean we make all vectors same length; but that doesn't really matter b/c we don't care about the length of the vector

Word2vec (word to vector)

* Neural network with proxy task over billions of words worth of text; learns an embedding layer
* Proxy task >> leverage text we have, try to predict something that's part of text based on context
  * CBOW = continuous bag of words
  * Skip-gram
    * Start with target word, predict the words around it
  * Goal = to learn representation of the word that's meaningful (not to predict); if a model knows how to predict the next word, then the model has some understanding of how language works
  * If 'cat' and 'dog' show up next to the same similar words
  * (go to the word 2 vec website) - find word relative to other words
  * probability/statistics to guess what the next word is; you have a matrix, you condense it into 2, the weight is what you change for each index matrix); final vector is probability it's any of the particular addresses - so eg. you put in 'a', you want to get 'cute' on other side; when you put in 'cute' want to get 'teddy'
    * Back propagation: take input vector, basic math; hidden states; you have weights to multiply and expand it back out; back-prop; i put in 'a' but wanted 'cute' - how far off am I? Then you push it back through to make the weights correct
    * ML - take input vector, put in some simple math layers / matrix, then expand out to make prediction; how far off is my prediction? Subtract difference, push back through model so that next time it's correct
    * I NEED TO RE-FAMILIAR MYSELF
    * Above, is the training - modifying the weights to get answers we want
    * When talking to chatgpt, it outputs based on what you think you want; but nothing is fed back into the model

Part IV: RNNs
Recurrent neural networks: instead of processing words 1 at a time; they keep hidden representation only so far, and consider tokens one at a time (context vector, hidden state)

* Takes into consideration the meaning of sentence so far; produces output vector to predict the next word
* RNN means word order matters; sentence can be encoded in a more natural way
* Hidden state = matrix box of numbers
  * So hidden state is representation the model maintains as it processes sequences of tokens, one at a time. The hidden state is updated based on the context so far in sentence. The previous token output is used as input for next token

LSTM Long Short term Memory

* Keep track of important things to remember on top of the hidden state (activation)

History of attention: want to remember something from the past to try to predict something in future (hidden state will de-code)

Part V: Self-attention Mechanism
Transformer (2017 - Attention is All You Need)

* Relies on self-attention mechanism
* There were start of art results on machine translation tasks
* So you have a query; want to see what other tokens are most similar?
  * Look at all other tokens composed of keys and values; compare query to key - quantify how similar the query is to a given key. Then, take corresponding value
  * Eg. query 'teddy bear' - compare it to all the other keys, then see which element is most similar? Then weight the most similar ones and take the associated value
  * Softmax formula can be used to know the values for weights ..
* GPUs love matrices - efficient computations -> softmax of query entity
* Value and key
  * Key - you figure what which is most similar to query
  * Value - associated with that element
* Projections are learned by mode; can obtain the quantities

Architecture

* Architecture = main coder, de-coder
  * Attention layer - input text (main coder)
  * De coder - target language (that you predict)
* You compute meaningful endings to input by passing through the encoder
  * All tokens in input text attend to one another - so all the tokens in text are a function of one another
* Feed forward layer - let model learn
* Positional encoding
* Overview
  * Decoder-decoder attention / self attention
  * Encoder decoder attention
* Encoder
  * Need representation from inputs
  * Parameters v and d_model

* Label smoothing
  * Overconfident is bad; introduce noise in true labels
  * This makes your model more 'unsure' about prediction

End to end example

* D

---

# Claude Study Guide (Explain Like You're 12)

Okay, I've read through all your notes from Lecture 1 of the Stanford CME295 LLM course. You've made it through the whole lecture and are sitting right at the "End to end example" section at the very bottom. Here's everything explained like you're 12:

---

## Part 1: What is NLP?

**NLP = Natural Language Processing.** It's just a fancy way of saying "teaching computers to understand human language." You give it text, and the computer does something smart with it.

There are a few different things you can do with this:

**Classification** is where you give the computer some text, and it puts it into a category. Like if you have movie reviews, it can figure out if someone is happy or angry (sentiment), what language the text is in, or what topic it's about.

**Multi-classification** is the same thing but now you're predicting many things at once. For example, in a sentence like "Apple is looking at buying U.K. startup," the model can label "Apple" as a company and "U.K." as a country -- all at the same time.

**Generation** is where the model actually *writes* new text. This is the cool stuff you know from ChatGPT -- translation, summarization, answering questions, writing text.

**How do we know if the model is doing a good job?** Your notes cover three measurement tools:
- **Accuracy** -- "How often did it get it right overall?"
- **Precision** -- "When it said something was positive, was it actually positive?" (Out of 10 it said were positive, 6 really were -> 60% precision)
- **Recall** -- "Out of all the actual positives, how many did it catch?" (There were 12 correct ones, but it only found 10 -> 83% recall)
- **F1 Score** -- a single number that blends precision and recall together

For text generation specifically, they use different measuring tools: **BLEU** (checks if translated words match closely), **ROUGE** (checks if the key content is there), and **Perplexity** (measures how "surprised" the model is by certain text -- lower surprise = better model).

---

## Part 2: Timeline -- How Did We Get Here?

Think of it like this:

- **1980s**: People started trying to teach computers about language
- **2013 (Word2Vec)**: A big breakthrough -- computers learned to give every word a meaningful number-location in space (more on this in Part 3!)
- **2017 (Transformers)**: The invention that changed EVERYTHING -- this is the foundation of all modern AI like ChatGPT
- **2020s (LLMs)**: People built MASSIVE versions of these models using enormous amounts of data -- this is what you use today

---

## Part 3: Tokenization -- Chopping Up Text

Computers don't understand words -- they only understand numbers. So the first step is to convert your text into little chunks called **tokens**, and then turn those tokens into numbers.

Imagine you have the sentence: *"cute teddy bear reading."* You need to break it apart so the model can process it.

There are three ways to do this:

**Word-level**: Each word is one token. Simple! But the problem is if a word appears that the model has never seen before (called OOV -- "Out of Vocabulary"), it gets confused. Also, "bear" and "bears" would be treated as totally different words.

**Sub-word level** (like WordPiece or BPE): Split words into smaller known pieces -- like "BEAR" and "s". So even if the model hasn't seen "BEARs," it can handle "BEAR" + "s." This is what most modern models use.

**Character-level**: Every single letter is its own token. It almost never runs into unknown words, but it makes sentences super long and hard to process, and you lose meaning (what does "b" mean by itself?).

---

## Part 3 (continued): Word Representation -- Giving Words Numbers

Okay, now you have your tokens. But how do you give each one a number? This is where it gets fun.

**One Hot Encoding (OHE)**: Imagine a giant grid where every word in the dictionary gets its own column. For the word "soft," you put a "1" in the "soft" column and "0" everywhere else. Like: soft = [1, 0, 0, 0, 0...]. It works, but the problem is every word is the same distance from every other word -- the model has no idea that "dog" and "puppy" are related, and "dog" and "volcano" are not.

**Learned Embeddings**: Instead of putting a word in one spot in a giant grid, you give it a *location* in a smaller space of numbers (like coordinates on a map). Words that are similar get placed nearby each other. So "soft" might be at coordinates [0.95, 0.25, 0.04]. Now "dog" and "puppy" would be close together on this map, and "volcano" would be far away. This is way more useful!

**Word2Vec (Word to Vector)**: This is a neural network (a type of computer brain) that *learns* these coordinate-locations automatically by looking at billions of words of text. It uses a trick called a "proxy task" -- instead of teaching it anything directly, you just ask it to *predict* what words tend to appear next to other words. In doing this, it figures out the meaning of words on its own!

The two main techniques are:
- **CBOW** (Continuous Bag of Words): given the words around it, predict the missing middle word
- **Skip-gram**: given one word, predict the words around it

The goal isn't actually to be good at predicting -- the goal is that in *learning* to predict, the model figures out deep relationships between words. If a model knows that you say "cute" near "teddy" and "soft" near "fluffy," it learns something about what those words mean.

---

## Part 4: RNNs -- Reading Text in Order

Now we have words as numbers. How do we actually process a whole sentence?

**Recurrent Neural Networks (RNNs)** are like a reader that goes word by word through a sentence, keeping notes as they go. As each new word comes in, they update a little memory box called the **hidden state**.

Think of it like reading a mystery novel. As you read each sentence, you keep a mental note of what's happened so far -- the hidden state is that mental note. It gets updated with each new sentence.

The important thing is that **word order matters** here. "The dog bit the man" and "The man bit the dog" are very different, and RNNs understand that.

**LSTM (Long Short-Term Memory)**: A fancier version of RNNs that's better at remembering things from way earlier in the sentence. Normal RNNs have a problem where they "forget" things from the beginning of a long text (called the vanishing gradient problem). LSTMs have a special extra memory slot that helps them hold onto important information for longer.

But RNNs still have problems: they're slow, and they don't always capture meaning well.

---

## Part 5: Self-Attention -- The Transformer Revolution

This is the BIG one. In 2017, a famous paper called **"Attention is All You Need"** introduced something totally new.

Imagine you're reading: *"The teddy bear sat on the soft couch because it was comfortable."* What does "it" refer to -- the bear or the couch? You have to look back at the earlier words and figure out which one "it" connects to. That's **attention** -- looking at all the other words in the sentence to understand the current word better.

**Self-attention** is how Transformers do this. For every word, the model looks at *all* the other words and figures out which ones are most relevant to understanding the current word. It asks:

- **Query**: "What am I looking for?" (the current word asking "who's relevant to me?")
- **Keys**: "What do I offer?" (each other word saying "here's what I'm about")
- **Values**: "What's my actual content?" (the information that gets passed along if you're relevant)

The model compares the Query to all the Keys using math (softmax formula) to figure out which words to pay attention to. Then it collects the Values from the most relevant words. These are all *learned* -- the model figures out what's a good query/key/value on its own.

This is much better than RNNs because it can look at ALL words at once (instead of one by one), and it can connect words that are far apart in a sentence directly.

---

## Architecture: Putting It All Together (The Encoder-Decoder)

The full Transformer model has two main parts:

**Encoder**: This is the "understanding" part. It takes your input text (like a sentence in English) and reads ALL the tokens, letting them all pay attention to each other. The result is a rich numerical representation of the meaning of the input. It uses two key parameters: `v` (values) and `d_model` (the size/depth of the representations).

**Decoder**: This is the "generating" part. It takes that rich understanding from the encoder and, one word at a time, generates the output (like the French translation). It can look at what it's already generated AND at the encoder's understanding of the input.

There are actually two types of attention happening:
- **Self-attention** in the decoder: the output so far looks at itself (decoder-decoder attention)
- **Encoder-decoder attention**: the decoder looks at the encoder's understanding of the input

**Feed Forward Layer**: After attention, each token goes through a simple math layer to help the model learn more complex patterns.

**Positional Encoding**: Here's a problem -- unlike RNNs, Transformers look at all words at once, so they don't naturally know the *order* of words. Positional encoding adds a little number-fingerprint to each token to tell the model "you're word #1," "you're word #2," etc.

---

## Label Smoothing

When training, if your model is too confident -- always saying "I'm 100% sure it's this word!" -- it tends to make overconfident mistakes. **Label smoothing** is a trick where you deliberately add a tiny bit of uncertainty to the training ("make the model slightly less sure"), which actually makes it *better* overall. It's like telling a student "never be overconfident, there's always a small chance you're wrong."

---

## End to End Example (Where You Are Now)

Your notes stop here with an empty bullet -- it looks like this is where the lecture was heading next, basically showing a complete example of a transformer doing a translation from start to finish. This would tie all the pieces together: tokenize the input -> embed the tokens -> pass through the encoder -> pass through the decoder -> generate output text.

---

**The big summary**: Text goes in -> it gets chopped into tokens -> tokens become numbers on a map -> a Transformer reads all those numbers at once (using attention) -> it outputs new text. That's fundamentally what ChatGPT and every other LLM is doing under the hood!
