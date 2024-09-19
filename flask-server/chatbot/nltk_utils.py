import re
import numpy as np
import nltk
#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def vietnamese_tokenizer(text):
  
  tokens = re.split(r"[ \t\n\r!\"#$%&()*+,./:;<=>?@\[\]^_`{|}~\\-]", text)
  tokens = [token for token in tokens if token.strip() != ""]
  return tokens

def stem(word):
  return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
  tokenized_sentence = [stem(w) for w in tokenized_sentence]
  bag = np.zeros(len(all_words), dtype = np.float32)
  for idx, w, in enumerate(all_words):
    if w in tokenized_sentence:
      bag[idx] = 1.0
  
  return bag

# Example usage
a = "Cho mình hỏi cái này được không?"
print(a)
a = vietnamese_tokenizer(a)
print(a)
print(stem(a[0])) 