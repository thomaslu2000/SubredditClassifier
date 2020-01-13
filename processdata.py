from nltk import word_tokenize, pos_tag, WordNetLemmatizer, download
from nltk.corpus import wordnet, stopwords

stopwords = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer() 

def get_wordnet_pos(tag):
    """Map POS tag to first character lemmatize() accepts"""
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def clean(text):
    text = text.lower()
    tokens = ["num" if w.isnumeric() else w for w in word_tokenize(text) if w not in stopwords]
    return " ".join(lemmatizer.lemmatize(w, pos=get_wordnet_pos(pos)) for w, pos in pos_tag(tokens))


subs = ["worldnews", "technology", "gaming", "travel"]

for sub in subs:
    with open('clean_%s.txt' % sub, 'w') as wf, open('%s.txt' % sub, 'r') as rf:
        line = rf.readline()
        while line:
            wf.write("%s\n" % clean(line))
            line = rf.readline()
