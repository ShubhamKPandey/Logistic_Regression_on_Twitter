from io import IncrementalNewlineDecoder
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt 

word_embeddings = pickle.load(open("word_embeddings_subset.p", "rb"))
len(word_embeddings)
print(len(word_embeddings))

countryVector = word_embeddings['country']
print(type(countryVector))
print(countryVector)

#Get the vector for a given word:
def vec(w):
    return word_embeddings[w]

words = ['oil', 'gas', 'happy', 'sad', 'city', 'town', 'village', 'country', 'continent', 'petroleum', 'joyful']

bag2d = np.array([vec(word) for word in words])

fig, ax = plt.subplots(figsize = (10, 10))

col1 = 3
col2 = 2

for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width = 0.005, head_length = 0.005, fc = 'r', ec= 'r', width = 1e-5)

ax.scatter(bag2d[:,col1], bag2d[:, col2])

for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]))

plt.savefig('..\po.png')

words = ['sad', 'happy', 'town', 'village']

bag2d = np.array([vec(word) for word in words])

fig, ax = plt.subplots(figsize = (10, 10))

col1 = 3
col2 = 2

for word in bag2d:
    ax.arrow(0, 0, word[col1], word[col2], head_width = 0.0005, head_length = 0.0005, fc = 'r', ec = 'r', width = 1e-5)

village = vec('village')
town = vec('town')
diff = town - village
ax.arrow(village[col1], village[col2], diff[col1], diff[col2], fc = 'b', ec = 'b', width = 1e-5)

sad = vec('sad')
happy = vec('happy')
diff = happy - sad
ax.arrow(sad[col1], sad[col2], diff[col1], diff[col2], fc = 'b', ec = 'b', width = 1e-5)

for i in range(0, len(words)):
    ax.annotate(words[i], (bag2d[i, col1], bag2d[i, col2]) )

plt.savefig('./to.png')

print(np.linalg.norm(vec('town')))
print(np.linalg.norm(vec('sad')))

capital = vec('France') - vec('Paris')
country = vec('Madrid') + capital
print(country[0:5])

diff = country - vec('Spain')
print(diff[0:10])

keys = word_embeddings.keys()
data = []
for key in keys:
    data.append(word_embeddings[key])

embedding = pd.DataFrame(data  = data,  index = keys)

def find_closest_word(v, k = 1):
    # We are trying to calculate the vector difference of the input word with each word in the dataframe

    diff = embedding.values - v
    delta = np.sum(diff * diff, axis = 1)

    i = np.argmin(delta)
    #Return the row name for this item
    return embedding.iloc[i].name

embedding.head(10)
find_closest_word(country)

find_closest_word(vec('Italy') - vec('Rome') + vec('Madrid'))

print(find_closest_word(vec('Berlin') + capital))
print(find_closest_word(vec('Beijing') + capital))
print(find_closest_word(vec('Lisbon') + capital))

doc = "Spain petroleum city king"
vdoc = [vec(x) for x in doc.split(" ")]
print(vdoc)
doc2vec = np.sum(vdoc, axis = 0)
print(doc2vec)
