import itertools as it
import random as rnd

fileTrain = open('train-processed.tab', 'r') # file contenente gli esempi di training
datasetTrain = fileTrain.read().split("\n")
trainData = []
testData = []
fileTest = open('test-processed.tab', 'r') # file contenente gli esempi di test
datasetTest = fileTest.read().split("\n")
Alltarget = [] # lista contenente tutte le classi possibili con cui un documento può essere classificato

# formatto il training set per renderlo compatibile con l'algoritmo di apprendimento
for i in datasetTrain:
    t = []
    splitted = i.split()
    cat = splitted[0]
    doc = splitted[1].split(',')
    t.append(doc)
    t.append(cat)
    trainData.append(t)
    if cat not in Alltarget:
        Alltarget.append(cat)

rnd.shuffle(trainData)

# formatto il test set per renderlo compatibile con l'algoritmo di classificazione
for i in datasetTest:
    t = []
    splitted = i.split()
    cat = splitted[0]
    doc = splitted[1].split(',')
    t.append(doc)
    t.append(cat)
    testData.append(t)

rnd.shuffle(testData)

# funzione che preleva tutte le parole distinte che appaiono in tutti i documenti del dataset
# input: insieme di documenti
# output: vocabolario di tutte le parole che appaiono nei documenti

def fetch_words(examples):
    voc = []
    for i in range(len(examples)):
        for j in range(len(examples[i][0])):
            if examples[i][0][j] not in voc:
                voc.append(examples[i][0][j])
    return voc

# funzione che dato un insieme di documenti e una categoria vj, restituisce la lista dei documenti di classe vj
# Inputs:
# examples: insieme di documenti
# vj: categoria di documento
# Output:
# docs: lista di documenti di classe vj presenti in examples

def subset(examples, vj):
    docs = []
    for i in range(len(examples)):
        if examples[i][1] == vj:
            docs.append(examples[i][0])
    return docs

# funzione che dato un testo restituisce il numero di tutte le parole distinte che occorrono nel testo
# input:
# text: testo di un documento
# output:
# numero di parole distinte all'interno del documento passato

def total(text):
    distinct = []
    for i in text:
        if i not in distinct:
            distinct.append(i)
    return len(distinct)

# funzione che data una parola di vocabolario conta quante volte essa appare nel documento text
# input:
# wk: parola all'interno del vocabolario
# text: testo di un documento
# output:
# numero di occorrenze di wk in text

def occurs(wk, text):
    tot = 0
    for i in text:
        if wk == i:
            tot += 1
    return tot

def getIndexes(doc, vocabulary):
    indexes = []
    for i in doc:
        if i in vocabulary:
            indexes.append(doc.index(i))
    return indexes


def product(doc, positions, vj, priorWk):
    p = 1
    for i in positions:
        k = doc[i]
        p *= priorWk[vj][k]
    return p

# funzione che definisce l'algoritmo di apprendimento di naive bayes. La funzione apprende le probabilità P(wk|vj) e P(vj)
# inputs:
# Examples: insieme di documenti su cui cui fare il training. Ogni documento è associato ad una categoria (target value)
# V: insieme di tutte le possibile categorie a cui i documenti possono appartenere
# Outputs:
# Vocabulary: vocabolario di tutte le parole che compaiono nei documenti del training set
# Prior: lista delle probabilità P(vj)
# PriorWk: lista delle probabilità P(wk|vj)

def learn_naive_bayes_text(Examples, V):
    vocabulary = [] # è il vocabolario dei termini che compaiono nei documenti
    prior = []  # lista delle probabilità P(vj)
    priorWk = []  # lista delle probabilità P(wk|vj)
    vocabulary = fetch_words(Examples)

    for vj in V: # per ognuna di tutte le possibili categorie a cui un documento può appartenere...
        docs = subset(Examples, vj)
        prior.append(len(docs) / len(Examples)) # calcolo della probabilità P(vj)
        text = list(it.chain.from_iterable(docs)) # text è un singolo documento contenente la concatenazione di tutti i documenti presenti nella lista docs
        n = total(text)
        d = {}
        for wk in vocabulary: # per ogni parola all'interno del vocabolario...
            nk = occurs(wk, text)
            k = wk # uso wk come chiave all'interno di un dizionario delle probabilità P(wk|vj) data la classe vj
            d[k] = (nk + 1) / (n + len(vocabulary)) # calcolo della probabilità P(wk|vj)
        priorWk.append(d)
    return vocabulary, prior, priorWk

# funzione che ritorna la classe stimata per il documento passato, basandosi sulle probabilità apprese durante la fase di apprendimento
# inputs:
# doc: documento di cui stimare la classe
# Alltarget: lista di tutte le possibili categorie di documento
# vocabuary: vocabolario delle parole appreso durante l'apprendimento
# prior: lista delle probabilità P(vj) apprese durante l'apprendimento
# priorWk: lista delle probabilità P(wk|vj) apprese durante l'apprendimento
# Output:
# bestClass: classe stimata da naive bayes per il documento passato

def classify_naive_bayes_text(doc, Alltarget, vocabulary, prior, priorWk):
    positions = getIndexes(doc, vocabulary)
    max = -1000
    for i in range(len(Alltarget)):
        nb = prior[i] * product(doc, positions, i, priorWk)
        if max < nb:
            max = nb
            bestClass = Alltarget[i]
    return bestClass

vocabulary, prior, priorWk =  learn_naive_bayes_text(trainData,Alltarget)

c = 0
for i in range(0,len(testData)):
    clas = classify_naive_bayes_text(testData[i][0], Alltarget, vocabulary, prior, priorWk)
    if clas == testData[i][1]:
        c += 1
print(c/len(testData)) # stampo lìaccurattezza dell'algoritmo
