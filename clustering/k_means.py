import sys
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris() # carico il dataset iris
X = iris.data # inserisco il dataset in una matrice

# funzione che calcola la differenza tra due vettori passati, differenza elemento per elemento
# inputs:
# o: primo vettore
# c: secondo vettore
# output: lista risultato della differenza tra o e c
def diff(o, c):
    list = []
    for i in range(0, len(o)):
        list.append(o[i] - c[i])
    return list


# funzione che inserisce l'esempio passato nel cluster corretto, in base alla distanza euclidea dell'esempio dal centroide di ogni cluster
# inputs:
# object: esempio di training da inserire in un cluster
# clusters: lista dei clusters
def assign(object, clusters):
    cluster = []
    min = sys.maxsize # min viene inizializzato al MAX_INT
    for i in clusters: # per ogni cluster...
        dist = np.linalg.norm(diff(object, i['centroid'])) # calcolo la distanza euclidea tra l'esempio passato e il centroide del cluster
        if min > dist:
            min = dist # aggiornamento della distanza minima
            cluster = i['objects'] # aggiornamento del cluster dove inserire l'esempio passato
    if object not in cluster:
        cluster.append(object) # inserimento dell'esempio nel cluster corretto


# funzione che aggiorna il centroide di tutti i cluster con la media dei punti all'interno del cluster corrispondente
def updateCentroids(clusters):
    for i in clusters:
        i['centroid'] = np.mean(np.array(i['objects']),axis=0).tolist()


# funzione che restituisce la lista dei centroidi dei clusters passati
def takeCentroids(clusters):
    centroids = []
    for i in clusters:
        centroids.append(i['centroid'])
    return centroids


# funzione che svuota i clusters passati
def empty(clusters):
    for i in clusters:
        i['objects'] = []

# funzione che clusterizza il training set passato, applicando l'algoritmo kMeans
# inputs:
# k: numero di cluster da produrre in output
# D: training set
# output: suddivisione degli esempi di training in k cluster

def kMeans(k, D):
    centroids = random.sample(D, k) # seleziono k centroidi iniziali a partire dalla lista degli esempi di training
    clusters = []
    # formatto la lista dei cluster. Ogni elemento rappresenta un cluster
    # la chiave objects contiene gli esempi inseriti nel cluster
    # la chiave centroid contiene il centroide del cluster
    # la chiave label contiene l'etichetta del cluster (da usare per il test)
    for i in centroids:
        dict = {}
        dict['objects'] = []
        dict['centroid'] = i
        dict['label'] = centroids.index(i)
        clusters.append(dict)

    ok = True

    while ok == True: # fino a quando i centroidi non si stabilizzano continuo ad iterare
        for i in D: # inserisco ogni esempio di training nel cluster il cui centroide dista meno dall'esempio
            assign(i, clusters)
        before = takeCentroids(clusters) # ottengo la lista dei centroidi prima di aggiornarli
        updateCentroids(clusters)
        after = takeCentroids(clusters) # ottengo la lista dei centroidi dopo averli aggiornati
        if before == after: # se i centroidi dopo l'aggiornamento non sono cambiati, allora devo fermare l'algoritmo e restituire la clusterizzazione
            ok = False
        else:
            empty(clusters) # se non cambiano allora resetto i cluster per riassegnare di nuovo gli esempi
    return clusters


result = kMeans(3, X.tolist())

color = ['b','grey','black'] # lista di colori per il plot
for i in result: # per ogni cluster...
    for j in i['objects']: # per ogni esempio dentro il cluster...
        x = j[2]
        y = j[3]
        plt.scatter(x, y, c=color[result.index(i)]) # plot dei punti nei cluster corrispondenti
    plt.scatter(i['centroid'][2], i['centroid'][3], c='r') # plot dei centroidi

labels = []
# ciclo per ottenere le labels da usare per il calcolo della adjusted rand score
for i in X.tolist():
    for j in result:
        for z in j['objects']:
            if i == z:
                labels.append(j['label'])

y = iris.target
print(metrics.adjusted_rand_score(y, np.array(labels))) # calcolo della adjusted rand score
plt.show()