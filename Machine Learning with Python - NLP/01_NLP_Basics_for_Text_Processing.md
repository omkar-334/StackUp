**Step 1: What is NLP?**

Welcome to the first quest in the introduction to natural language processing (NLP) campaign! In this step, let's run through a brief overview of what NLP is and how it can be utilised.

NLP involves using algorithms and computational models to understand, analyse and generate human language data. The goal of NLP is to enable computers to understand and process human language as it is spoken and written, and to generate language that is natural and meaningful to humans. In order to achieve this, NLP draws on many different disciplines such as linguistics, computer science, statistics and mathematics.

**Step 2: Applications of NLP**

NLP covers a wide range of tasks, and can be more prominent in our world than you may realise. Everyday things like the predictive texts that pop up while texting and detecting spam emails in your inbox all make use of NLP. Here are some other examples of NLP applications that you might have come across:

1. **Voice assistants** : Voice assistants such as Amazon Alexa and Apple Siri make use of NLP to understand and process the commands and queries from users.
2. **Chatbots** : Chatbots used for customer service and support on websites and social media platforms also use NLP to interact with users in natural language.
3. **Social media** : NLP can be used to analyse social media posts for sentiment analysis, topic identification and language translation
4. **Search engines** : NLP is used to improve a user’s search results by understanding the meaning and context of search queries and matching them with relevant documents
5. **News and media** : NLP can also be used here to analyse and classify news articles into categories, identify fake news, and summarise news stories for readers

NLP has evidently become an integral part of many technologies that we use in our daily lives. NLP is a rapidly evolving field, and as technology continues to advance, we can expect to see more innovative and useful applications of NLP in the future.

**Step 3: Overview of a Typical NLP Project**

When used in data science, a typical NLP project can be split into the following stages:

1. Text preprocessing
2. Feature engineering and Vectorization
3. Model building
4. Model evaluation
5. Deployment

In this quest, we will be learning about the first stage, which is the text preprocessing and preparation of your data before it can be analysed. This stage is important as it ensures that the raw text data is cleaned, normalised and transformed into a format that can be easily understood and processed by machine learning algorithms. The basic text preprocessing we will cover in this quest are tokenization, removing stop-words, stemming and lemmatization.

**Step 4: Tokenization**

Tokenization is the process of breaking down a piece of text into smaller individual units, called tokens. These tokens could be words, phrases or even individual characters such as punctuation marks, depending on the specific task and requirements.

Tokenization is essential in NLP as most natural language processing tasks require text data to be represented in a numerical format that can be processed by algorithms. Tokenization helps to convert raw text data into a sequence of tokens, which can then be used for further analysis and processing.

After tokenization, the resulting tokens can be further processed using techniques such as stemming, lemmatization, or part-of-speech tagging. This makes tokenization a fundamental technique in NLP, where the quality and accuracy of the tokens creates a significant impact on the performance of how the systems operate.

In this step, we will make use of the natural language toolkit (NLTK) Python library for its built-in methods. NLTK is commonly used to work with human language data. More information can be found [here](https://www.nltk.org/) and you can install it by referring [here](https://www.nltk.org/install.html) before we begin to perform simple tokenization using Jupyter notebook.

**Step 5: Removing Stop Words**

In NLP, stop words are commonly occurring words that are generally considered to have little or no meaning. These words are often removed from text data during preprocessing. Some examples of stop words in English include “the”, “a”, “an”, “in”, “of” and “and”.

Stop words are often removed during the preprocessing stage to reduce the size of vocabulary and data that is being worked with. It also improves the accuracy of NLP models by removing noise from the data. With fewer words to analyse, this speeds up the processing of text data.

There are various ways to remove stop words in NLP, such as using a pre-built list of stop words, or manually creating a custom list of stop words to suit your project’s needs. Machine learning techniques can also be used to identify and remove stop words based on their frequency and other features.

Now let’s see it in action! **Open up the Jupyter notebook** and refer to the section on Removing Stop words, where we will make use of NLTK’s pre-defined English stop words and remove them from the tokens created in the previous step.

**Step 6: Stemming and Lemmatization**

Moving on to stemming and lemmatization, these are two common techniques used in NLP to shorten words in text data. Some text may contain grammar that does not influence the meaning of the word in text processing. In order to remove this, we use techniques such as stemming and lemmatization.

Stemming is the process of removing the ends of a word, to get the stem. This is done by removing the [suffixes](https://dictionary.cambridge.org/grammar/british-grammar/suffixes) from the words. Stemming is a relatively simple and fast technique, but it can sometimes lead to incorrect results or stems that are not actual words in the dictionary as it does not take the context of the word.

On the other hand, lemmatization is the process of reducing words to their base form, known as the lemma. Unlike stemming, lemmatization takes into account the context of the words and their part of speech. For example, stemming the word ‘caring’ would return ‘car’ as the suffix ‘-ing’ is removed, while lemmatizing the word ‘caring’ would return ‘care’. Lemmatization is a more complex and accurate technique, but it can be slower and require more computational resources than stemming. As a result, the choice between the two techniques depends on the specific needs of the NLP application being developed.

**Step 7: Wrap-Up!**

In this quest, we covered three types of text preprocessing that are normally used in NLP. There are also other possible types of preprocessing that one can use to prepare text for an NLP project. Such includes removing text that contains digits, converting all text to lowercase, and even expanding contractions such as ‘don't’, ‘isn't’ and ‘aren't’.

While not covered in this campaign, another important process is Part-of-Speech (POS) tagging, which categorises words in a text with its particular part of speech, such as verb, noun, adjective and adverb. This allows users to get a better sense of how a word is used in a sentence. More on POS tagging can be found [here](https://towardsdatascience.com/part-of-speech-tagging-for-beginners-3a0754b2ebba).
