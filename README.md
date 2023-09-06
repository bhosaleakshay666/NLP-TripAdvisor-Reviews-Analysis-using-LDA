## LDA on TripAdvisor Review Data

### Importing Libraries and Dataset


```python
import re
import numpy as np
import pandas as  pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
from pyLDAvis import gensim_models
import matplotlib.pyplot as plt
%matplotlib inline
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\aksha\AppData\Roaming\nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    




    True




```python
df=pd.read_excel(r'C:\Users\aksha\Downloads\TripAdvisor Reviews.xlsx')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Topic</th>
      <th>Body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Review 2194: 229036_1543288</td>
      <td>5</td>
      <td>Beautiful hotel wonderful service My husband ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Review 9352: 239263_13650915</td>
      <td>5</td>
      <td>Excellent I picked the hotel out of a hotel g...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Review 7063: 529623_4027170</td>
      <td>3</td>
      <td>Okay for budget travel Hostel pension for bud...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Review 3131: 298628_4742688</td>
      <td>4</td>
      <td>Really loved this fab Hotel Myself my mother ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Review 519: 232867_2265435</td>
      <td>5</td>
      <td>Best You will Get In Rome The last few review...</td>
    </tr>
  </tbody>
</table>
</div>



### Defining Stopwords


```python
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
stop_words = set(stopwords.words("english"))
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words = stop_words.union(new_words)
#stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
```

### Converting Reviews to list and Removing Unnecessary Characters


```python
# Convert to list 
data = df.Body.values.tolist()  

data = [re.sub('\s+', ' ', sent) for sent in data]  

data = [re.sub("\'", "", sent) for sent in data]  
pprint(data[:1])
```

    <>:4: DeprecationWarning: invalid escape sequence \s
    <>:4: DeprecationWarning: invalid escape sequence \s
    C:\Users\aksha\AppData\Local\Temp/ipykernel_8856/602162079.py:4: DeprecationWarning: invalid escape sequence \s
      data = [re.sub('\s+', ' ', sent) for sent in data]
    

    [' Beautiful hotel wonderful service My husband and I spent Christmas and New '
     'Years in Rome and the splendide royal was our address while there It was '
     'wonderful-the room was beautifully decorated the service was impeccable and '
     'the location was perfect We were able to enjoy Romes wonderful fireworks '
     'display from the elegant terrace of the hotels Mirabelle restaurant I highly '
     'recommend this hotel-it made for one of our most memorable vacations '
     'together ']
    

### Tokenize and Cleaning Reviews


```python
def sent_to_words(sentences):
  for sentence in sentences:
    yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))            
data_words = list(sent_to_words(data))
print(data_words[:1])
```

    [['beautiful', 'hotel', 'wonderful', 'service', 'my', 'husband', 'and', 'spent', 'christmas', 'and', 'new', 'years', 'in', 'rome', 'and', 'the', 'splendide', 'royal', 'was', 'our', 'address', 'while', 'there', 'it', 'was', 'wonderful', 'the', 'room', 'was', 'beautifully', 'decorated', 'the', 'service', 'was', 'impeccable', 'and', 'the', 'location', 'was', 'perfect', 'we', 'were', 'able', 'to', 'enjoy', 'romes', 'wonderful', 'fireworks', 'display', 'from', 'the', 'elegant', 'terrace', 'of', 'the', 'hotels', 'mirabelle', 'restaurant', 'highly', 'recommend', 'this', 'hotel', 'it', 'made', 'for', 'one', 'of', 'our', 'most', 'memorable', 'vacations', 'together']]
    

### Build Bigrams & Trigrams Models


```python
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)
print(trigram_mod[bigram_mod[data_words[0]]])
```

    ['beautiful', 'hotel', 'wonderful', 'service', 'my', 'husband', 'and', 'spent', 'christmas', 'and', 'new', 'years', 'in', 'rome', 'and', 'the', 'splendide_royal', 'was', 'our', 'address', 'while', 'there', 'it', 'was', 'wonderful', 'the', 'room', 'was', 'beautifully_decorated', 'the', 'service', 'was', 'impeccable', 'and', 'the', 'location', 'was', 'perfect', 'we', 'were', 'able', 'to', 'enjoy', 'romes', 'wonderful', 'fireworks', 'display', 'from', 'the', 'elegant', 'terrace', 'of', 'the', 'hotels', 'mirabelle', 'restaurant', 'highly', 'recommend', 'this', 'hotel', 'it', 'made', 'for', 'one', 'of', 'our', 'most', 'memorable', 'vacations', 'together']
    

### Define & Call Functions for removing Stopwords, creating Bigrams & Trigrams and Lemmatizing


```python
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
```


```python
data_words_nostops = remove_stopwords(data_words)

#Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])
```

    [['beautiful', 'hotel', 'wonderful', 'service', 'husband', 'spend', 'year', 'rome', 'splendide_royal', 'address', 'wonderful', 'room', 'beautifully_decorate', 'service', 'impeccable', 'location', 'perfect', 'able', 'enjoy', 'rome', 'wonderful', 'firework', 'display', 'elegant', 'terrace', 'hotel', 'restaurant', 'highly', 'recommend', 'hotel', 'make', 'memorable', 'vacation', 'together']]
    

### Create Corpus Dictionary


```python
# Create Dictionary 
id2word = corpora.Dictionary(data_lemmatized)  
# Create Corpus 
texts = data_lemmatized  
# Term Document Frequency 
corpus = [id2word.doc2bow(text) for text in texts]
print(corpus[:1])
```

    [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1), (9, 3), (10, 1), (11, 1), (12, 1), (13, 1), (14, 1), (15, 1), (16, 1), (17, 1), (18, 2), (19, 1), (20, 2), (21, 1), (22, 1), (23, 1), (24, 1), (25, 1), (26, 3), (27, 1)]]
    


```python
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
```




    [[('able', 1),
      ('address', 1),
      ('beautiful', 1),
      ('beautifully_decorate', 1),
      ('display', 1),
      ('elegant', 1),
      ('enjoy', 1),
      ('firework', 1),
      ('highly', 1),
      ('hotel', 3),
      ('husband', 1),
      ('impeccable', 1),
      ('location', 1),
      ('make', 1),
      ('memorable', 1),
      ('perfect', 1),
      ('recommend', 1),
      ('restaurant', 1),
      ('rome', 2),
      ('room', 1),
      ('service', 2),
      ('spend', 1),
      ('splendide_royal', 1),
      ('terrace', 1),
      ('together', 1),
      ('vacation', 1),
      ('wonderful', 3),
      ('year', 1)]]



### LDA Modeling for Six Topics


```python
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
```

### Print keywords of Topics


```python
# Print the keyword of topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

    [(0,
      '0.050*"hotel" + 0.044*"room" + 0.017*"night" + 0.014*"nice" + 0.014*"get" + '
      '0.013*"good" + 0.012*"go" + 0.011*"small" + 0.011*"day" + 0.010*"stay"'),
     (1,
      '0.029*"service" + 0.020*"wonderful" + 0.018*"view" + 0.017*"make" + '
      '0.017*"restaurant" + 0.016*"beautiful" + 0.013*"love" + 0.012*"stay" + '
      '0.011*"enjoy" + 0.010*"dinner"'),
     (2,
      '0.021*"say" + 0.019*"ask" + 0.017*"bad" + 0.015*"check" + 0.014*"book" + '
      '0.014*"tell" + 0.012*"give" + 0.012*"call" + 0.010*"room" + 0.009*"never"'),
     (3,
      '0.077*"hotel" + 0.054*"great" + 0.052*"stay" + 0.052*"staff" + '
      '0.049*"location" + 0.040*"clean" + 0.035*"breakfast" + 0.030*"good" + '
      '0.026*"helpful" + 0.024*"recommend"'),
     (4,
      '0.049*"floor" + 0.035*"door" + 0.026*"open" + 0.026*"noise" + '
      '0.024*"window" + 0.021*"hear" + 0.019*"noisy" + 0.017*"building" + '
      '0.016*"wall" + 0.016*"shower"'),
     (5,
      '0.040*"access" + 0.021*"pillow" + 0.021*"accurate" + 0.017*"pavement" + '
      '0.016*"report" + 0.016*"position" + 0.015*"recognize" + '
      '0.015*"disappointment" + 0.014*"gelato" + 0.013*"dropping"')]
    



