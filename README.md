# Biterm Topic Model

This is a simple Python implementation of the awesome
[Biterm Topic Model](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.402.4032&rep=rep1&type=pdf).
This model is accurate in short text classification.
It explicitly models the word co-occurrence patterns in the whole corpus to solve the problem of sparse word co-occurrence at document-level.

Simply install by:
```
pip install biterm
```

Load some short texts and vectorize them via sklearn.

```python
    from sklearn.feature_extraction.text import CountVectorizer

    texts = open('./data/reuters.titles').read().splitlines()[:50]
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()
```
Get the vocabulary and the biterms from the texts.
```python
    from biterm.utility import vec_to_biterms

    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(X)
```
Create a BTM and pass the biterms to train it.
```python
    from biterm.btm import oBTM

    btm = oBTM(num_topics=20, V=vocab)
    topics = btm.fit_transform(biterms, iterations=100)
```
Save a topic plot using pyLDAvis and explore the results! (also see *simple_btml.py*)
```python
    from biterm.btm import oBTM

    btm = oBTM(num_topics=20, V=vocab)
    topics = btm.fit_transform(biterms, iterations=100)
```
![pyLDAvis Visualization](https://github.com/markoarnauto/biterm/blob/master/vis/simple_btm.png)

Inference is done with Gibbs Sampling and it's not really fast. The implementation is not meant for production.
But if you have to classify a lot of texts you can try using online learning. 
```python
import numpy as np
import pyLDAvis
from biterm.btm import oBTM 
from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary # helper functions

if __name__ == "__main__":

    texts = open('./data/reuters.titles').read().splitlines() # path of data file

    # vectorize texts
    vec = CountVectorizer(stop_words='english')
    X = vec.fit_transform(texts).toarray()

    # get vocabulary
    vocab = np.array(vec.get_feature_names())

    # get biterms
    biterms = vec_to_biterms(X)

    # create btm
    btm = oBTM(num_topics=20, V=vocab)

    print("\n\n Train Online BTM ..")
    for i in range(0, len(biterms), 100): # prozess chunk of 200 texts
        biterms_chunk = biterms[i:i + 100]
        btm.fit(biterms_chunk, iterations=50)
    topics = btm.transform(biterms)

    print("\n\n Visualize Topics ..")
    vis = pyLDAvis.prepare(btm.phi_wz.T, topics, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
    pyLDAvis.save_html(vis, './vis/online_btm.html')  # path to output

    print("\n\n Topic coherence ..")
    topic_summuary(btm.phi_wz.T, X, vocab, 10)

    print("\n\n Texts & Topics ..")
    for i in range(len(texts)):
        print("{} (topic: {})".format(texts[i], topics[i].argmax()))
```
Use the Cython version to speed up performance. Therefore, you can download the repo and build the cbtm.pyx for the operating system of your choice.
Afterwards use `from biterm.cbtm import oBTM` to use the cythonic version.
