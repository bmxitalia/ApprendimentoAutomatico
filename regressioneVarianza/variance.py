import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import explained_variance_score, mean_squared_error
import random


# definizione della funzione target da approssimare con l'interpolazione polinomiale
def f(x):
  return 3 * x + 5 * np.sin(2 * x)  # funzione complessa da interpolare


x_sin = np.linspace(0, 10,
                    10000)  # genero 10000 numeri equidistanti tra 0 e 10, usati per rappresentare la funzione target
y_sin = f(x_sin)
figure0 = plt.figure(0)
plt.plot(x_sin, y_sin, label="target function")
plt.title("Figura 1")
plt.legend()
figure0.show()  # la prima figura mostra la funzione target

# genero la versione matriciale degli array costruiti
X_sin = x_sin[:, np.newaxis]

# utilizzo kfold cross validation con 200 split
kf = KFold(n_splits=200, random_state=42, shuffle=True)
i = 0
w = []
figure1 = plt.figure(1)
for train_index, test_index in kf.split(X_sin):
  i += 1
  X_train, X_test = X_sin[train_index], X_sin[test_index]
  y_train, y_test = y_sin[train_index], y_sin[test_index]
  model = make_pipeline(PolynomialFeatures(15), Ridge(1.0, solver='svd'))
  model.fit(X_test,
            y_test)  # in ognuno dei 200 modelli abbiamo a disposizione solamente 50 punti di questi 10000 per fare training e questi vengono utilizzati interamente per il training facendo overfitting per forza di cose
  y_sinp = model.predict(X_sin)
  plt.subplot(20, 10, i)
  plt.scatter(X_test, y_test, color='red', marker='o', label='training points',
              linewidth='0.1')  # visualizzo per il modello corrente i punti di training utilizzati
  plt.plot(x_sin, y_sinp,
           color='grey')  # visualizzo per il modello corrente la funzione predetta dal modello su tutti i punti del dataset, compresi i 50 su cui ho fatto training
  w.append(model.steps[1][1].coef_)

figure1.set_figheight(30)
figure1.set_figwidth(30)
figure1.show()  # visualizzo la figura contenente i grafici dei 200 modelli appresi

meanW = np.mean(w, axis=0)  # calcolo la media dei pesi dei 200 modelli appresi

poly = np.poly1d(np.flip(meanW,
                         -1))  # poly è il polinomio che interpola i punti, ovvero sarebbe il modello costruito a partire dalla media dei pesi di tutti i 200 modelli costruiti
y3 = poly(x_sin)
figure2 = plt.figure(2)
plt.title("Figura 2")
plt.scatter(x_sin, y_sin, color='red', label="target function")  # disegno la funzione target in rosso
plt.plot(x_sin, y3, color='grey', label="mean weights model")  # disegno il polinomio interpolatore in grigio
plt.legend()
figure2.show()  # questa figura mostra come effettuando la media dei pesi venga abbattuta la varianza. Si può notare infatti che essa risulta nulla ma che la funzione presenta un bias rispetto alla funzione target

# calculate accuracy
# 1 - u/v

u = ((y_sin - y3) ** 2).sum()
v = ((y_sin - y_sin.mean()) ** 2).sum()
customModelAccuracy = 1 - u / v

# genero 50 numeri random unici
randomIndexes = random.sample(range(0, 10000), 50)

partialX_sin = X_sin[randomIndexes]
partialy_sin = y_sin[randomIndexes]

i = 0
scores = []
k = 10

for j in range(2, 10):  # provo da 2 a 10 split e cerco poi quello ottimo
  partialkf = KFold(n_splits=j, random_state=42, shuffle=True)
  for train_index, test_index in partialkf.split(partialX_sin):
    i += 1
    model = make_pipeline(PolynomialFeatures(10), Ridge(1.0, solver='svd'))
    X_train, X_test = partialX_sin[train_index], partialX_sin[test_index]
    y_train, y_test = partialy_sin[train_index], partialy_sin[test_index]
    model.fit(X_train, y_train)
  scores.append({"index": j, "score": model.score(X_sin, y_sin), })

figure3 = plt.figure(3)

partialy = model.predict(X_sin)


def getBestSplit(scores):  # ricerca dello split ottimo
  max = -1
  tmpK = -1
  for score in scores:
    if (score["score"] > max):
      max = score["score"]
      tmpK = score["index"]
  return tmpK


bestK = getBestSplit(scores)

# Individuato il k migliore alleno il modello definitivo con il k trovato
definitiveModel = make_pipeline(PolynomialFeatures(10), Ridge(1.0, solver='svd'))

defpartialkf = KFold(n_splits=bestK, random_state=42, shuffle=True)
for train_index, test_index in defpartialkf.split(partialX_sin):
  i += 1
  X_train, X_test = partialX_sin[train_index], partialX_sin[test_index]
  y_train, y_test = partialy_sin[train_index], partialy_sin[test_index]
  definitiveModel.fit(X_train, y_train)

definitivePartialy = model.predict(X_sin)

figure4 = plt.figure(4)
plt.title("Figura 3")
plt.plot(x_sin, y_sin, color='red', label="target function")  # funzione target
plt.scatter(x_sin, definitivePartialy, color='grey', label="predicted function")  # funzione appresa con k migliore
plt.legend()
figure4.show()