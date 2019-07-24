import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# definizione della retta che separa gli esempi di training
def f(x):
    return (7*x-2)/5


# generazione di un training set linearmente separabile
X = np.random.rand(100,2) # genera casualmente 100 tuple di valori
x = np.linspace(0,1,10)
y = f(x)
plt.subplot(131) # grafico che mostra la separabilità lineare del training set, con esempi positivi in verde e negativi in rosso
plt.plot(x,y) # visualizzazione della retta che separa gli esempi di training
points = [] # contiene gli esempi di training
c1 = 0
c2 = 0
for i in range(len(X)):
    if (7*X[i,0]-2)<5*X[i,1]:
        plt.scatter(X[i,0],X[i,1],color='green', marker='o', label='positive example' if c1 == 0 else "") # uso come ascisse la prima colonna e come ordinate la seconda
        points.append({"x": np.array([X[i, 0],X[i, 1]]), "t": 1}) # se l'esempio è a sinistra della retta allora lo classifico positivo
        c1 = c1 + 1
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', marker='o', label='negative example' if c2 == 0 else "")
        points.append({"x": np.array([X[i, 0],X[i, 1]]), "t": -1}) # se l'esempio è a destra della retta allora lo classifico negativo
        c2 = c2 + 1

plt.legend(loc='lower right')
plt.title("Pos/Neg Examples")

plt.subplot(132) # grafico che distingue il training set dal validation set. Esempi di training in giallo ed esempi di test in nero
plt.plot(x,y) # visualizzazione della retta che separa gli esempi di training

c1 = 0
c2 = 0

# disegno il training set e lo formatto per poter utilizzare il Perceptron di scikit-learn
trainingX = []
trainingY = []
for i in range(len(points)-30):  # uso 70 esempi per il training set e 30 esempi per il validation set
    plt.scatter(points[i]['x'][0],points[i]['x'][1] , color='yellow', marker='o', label='training example' if c1 == 0 else "")
    c1 = c1 +1
    trainingX.append(points[i]['x'])
    trainingY.append(points[i]['t'])

# disegno il validation set e lo formatto per poter utilizzare il Perceptron di scikit-learn
validationX = []
validationY = []
for i in range(70,len(points)):
    plt.scatter(points[i]['x'][0], points[i]['x'][1], color='black', marker='o', label='test example' if c2 == 0 else "")
    c2 = c2 +1
    validationX.append(points[i]['x'])
    validationY.append(points[i]['t'])

# faccio il fit dell'algoritmo Perceptron implementato da scikit-learn
model = Perceptron()
model.fit(trainingX,trainingY)
print("Accuratezza = ",model.score(validationX,validationY)) # stampa dell'accuratezza
validationPredict = model.predict(validationX) # eseguo la predizione

plt.legend(loc='lower right', scatterpoints = 1)
plt.title("Training/Test sets")

plt.subplot(133) # grafico che distingue le classificazioni corrette e quelle scorrette sul test set. Corrette in arancione e scorrette in blu
plt.plot(x,y) # visualizzazione della retta che separa gli esempi di training

c1 = 0
c2 = 0

for i in range(len(validationY)):
    if validationY[i] == validationPredict[i]:
        plt.scatter(validationX[i][0], validationX[i][1], color='orange', marker='o', label='Classified example' if c1 == 0 else "")
        c1 = c1 +1
    else:
        plt.scatter(validationX[i][0], validationX[i][1], color='blue', marker='o', label='Missclassified example' if c2 == 0 else "")
        c2 = c2 +1

plt.legend(loc='lower right', numpoints = 1)
plt.title("Classified/Misclassified test data")


plt.show() # visualizzo i tre grafici costruiti