import numpy as np
import matplotlib.pyplot as plt

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
        points.append({"x": np.array([X[i, 0],X[i, 1],1]), "t": 1}) # se l'esempio è a sinistra della retta allora lo classifico positivo
        c1 = c1 + 1
    else:
        plt.scatter(X[i, 0], X[i, 1], color='red', marker='o', label='negative example' if c2 == 0 else "")
        points.append({"x": np.array([X[i, 0],X[i, 1],1]), "t": -1}) # se l'esempio è a destra della retta allora lo classifico negativo
        c2 = c2 + 1

plt.legend(loc='lower right')
plt.title("Pos/Neg Examples")

plt.subplot(132) # grafico che distingue il training set dal validation set. Esempi di training in giallo ed esempi di test in nero
plt.plot(x,y) # visualizzazione della retta che separa gli esempi di training

n = 1 # coefficiente eta impostato a 1

# inizio algoritmo gradient-descent
# genero il vettore dei pesi casualmente, con valori piccoli
w = []
deltaW = []
for i in range(3):
    w.append(np.random.uniform(-0.05,0.05))
    deltaW.append(0)

c1 = 0
c2 = 0

################### ciclo di aggiornamento dei pesi della regola delta con attivazione sigmoidale -- fase di fit dell'algoritmo
for j in range(100):
    for z in range(len(deltaW)):
        deltaW[z] = 0
    for i in range(len(points)-30):  # uso 70 esempi per il training set e 30 esempi per il validation set
        plt.scatter(points[i]['x'][0],points[i]['x'][1] , color='yellow', marker='o', label='training example' if c1 == 0 else "")
        c1 = c1 + 1
        o = 1 / (1 + (np.exp(-np.dot(points[i]['x'],w)))) # dot esegue il prodotto sclare tra vettori
        for j in range(len(deltaW)): # calcolo del valore per aggiornare i pesi
            deltaW[j] = deltaW[j] + n*(points[i]['t'] - o)*o*(1-o)*points[i]['x'][j] # valore delta da aggiungere ai vari pesi
    for z in range(len(w)):
        w[z] = w[z] + deltaW[z]
#################### fine fit algoritmo
for i in range(70,len(points)):
    plt.scatter(points[i]['x'][0], points[i]['x'][1], color='black', marker='o', label='test example' if c2 == 0 else "")
    c2 = c2 +1

plt.legend(loc='lower right', scatterpoints = 1)
plt.title("Training/Test sets")

plt.subplot(133) # grafico che distingue le classificazioni corrette e quelle scorrette sul test set. Corrette in arancione e scorrette in blu
plt.plot(x,y) # visualizzazione della retta che separa gli esempi di training

c1 = 0
c2 = 0

######### predizione delle etichette: gli output della sigmoide maggiori o uguali a 0.50 verranno etichettati positivamente (1), mentre quelli minori negativamente (-1)
predictLabels = []
for i in range(70,len(points)):
    if (1 / (1 + np.exp(-np.dot(points[i]['x'], w)))) >= 0.50:
        predictLabels.append(1)
    else:
        predictLabels.append(-1)
################ fine predizione etichette ###########################################################################################################
accuracy = 0

############### predict sul validation set --> applicazione del Perceptron con pesi appresi dalla regola delta con attivazione sigmoidale ###############################
for i in range(70,len(points)):
    if predictLabels[i-70] == points[i]['t']:
        plt.scatter(points[i]['x'][0], points[i]['x'][1], color='orange', marker='o', label='Classified example' if c1 == 0 else "")
        c1 = c1 +1
        accuracy = accuracy + 1
    else:
        plt.scatter(points[i]['x'][0], points[i]['x'][1], color='blue', marker='o', label='Missclassified example' if c2 == 0 else "")
        c2 = c2 +1

accuracy /= 30
print(accuracy)
############## fine predizione #################################################################################################################################

plt.legend(loc='lower right', numpoints = 1)
plt.title("Classified/Misclassified test data")

plt.show() # visualizzo i tre grafici costruiti