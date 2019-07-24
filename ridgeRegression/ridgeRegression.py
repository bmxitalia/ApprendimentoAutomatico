# importazione pacchetti
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# definizione della funzione target da approssimare con l'interpolazione polinomiale
def f(x):
	return 3*x+5*np.sin(2*x) # funzione abbastanza irregolare

# genero i punti e ne prendo un sottoinsieme andando a costituire un training set
x_sin = np.linspace(0,10,100) # genero 100 numeri equidistanti tra 0 e 10, usati per rappresentare la funzione target
x_training = np.linspace(0,10,100) # questi vengono invece usati per rappresentare il training set
random = np.random.RandomState(0)
random.shuffle(x_training) # mescolo casualmente i punti generati
x_training = np.sort(x_training[:20]) # prendo i primi 20 e li riordino
y_training = f(x_training) # ottengo le ordinate dei punti

# genero la versione matriciale degli array costruiti
X_training = x_training[:, np.newaxis]
X_sin = x_sin[:, np.newaxis]

graphPosition = 0 # è la posizione del grafico all'interno della griglia che viene visualizzata

for count1, degree in enumerate([1,5,10]): # visualizzo tre gradi di polinomio
	graphPosition = 231 + count1
	for count, alpha in enumerate([0,7]): # provo due valori di alpha
		model = make_pipeline(PolynomialFeatures(degree),Ridge(alpha)) # costruisco una pipeline, prima genero il polinomio con un grado fissato e poi applico la ridge regression per trovare il valore dei pesi
		model.fit(X_training,y_training) # alleno il modello sugli esempi di training, in pratica vengono trovati i pesi che minimizzano l'errore empirico e si cerca di approssimare la funzione target
		y_sin = model.predict(X_sin) # predice il valore per i nuovi punti, così vedo quanto la funzione appresa si avvicina alla funzione target. Il modello ha predetto una funzione e la funzione predict restituisce le ordinate delle ascisse passate in base alla funzione appresa
		plt.subplot(graphPosition)
		graphPosition = graphPosition + 3
		#plt.plot(x_sin,f(x_sin),color='green',label='target function') # visualizzo la funzione target
		plt.scatter(x_training,y_training,color='red',marker='o',label='training points',linewidth='0.1') # visualizzo i punti che corrispondono al mio training set
		plt.plot(x_sin,y_sin, color='grey',label="degree %d alpha %d" %(degree,alpha)) # disegno la funzione predetta, con le ascisse originarie e le ordinate predette dal modello
		plt.legend(loc='upper left') # visualizzo la legenda in alto a sinistra

y2 = f(x_sin)
plt.plot(x_sin, y2, color='g')
plt.show()
