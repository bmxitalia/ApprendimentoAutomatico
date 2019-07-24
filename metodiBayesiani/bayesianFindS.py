import itertools as it # serve per effettuare il prodotto cartesiano

# spazio delle istanze
X = [['0','Sunny','Cloudy','Rainy','?'],['0','Warm','Cold','?'],['0','Normal','High','?'],['0','Strong','Weak','?'],['0','Warm','Cool','?'],['0','Same','Change','?']]
H = list(it.product(*X)) # spazio delle ipotesi
# dataset
D = [['Sunny','Warm','Normal','Strong','Warm','Same','Yes'],['Sunny','Warm','High','Strong','Warm','Same','Yes'],['Rainy','Cold','High','Strong','Warm','Change','No'],['Sunny','Warm','High','Strong','Cool','Change','Yes']] # training set
C = [] # array binario delle probabilità P(D|h)
P = [] # array delle probabilità stimate per ogni ipotesi P(h)

# funzione che dati lo spazio delle ipotesi, il dataset e l'array C, inserisce in C le probabilità P(D|h) calcolate per ogni ipotesi sulla base del dataset fornito
def setPriorProb(H,C,D):
    out = False
    for i in range(len(H)):
        out = False
        for j in range(len(D)):
            if out == False and D[j][6] == 'Yes': # se l'esempio è positivo ha senso andare avanti con i controlli
                for z in range(6):
                    if H[i][z] != D[j][z] and H[i][z] != '?' and out == False:
                        C.append(0)
                        out = True
        if out == False:
            C.append(1)

# dato un numero di ? all'interno dell'ipotesi, restituisce la sua probabilità P(h). Probabilità più alte sono state date a ipotesi più specifiche
def setProb(cont):
    if cont == 6: return 0
    if cont == 5: return 0.1
    if cont == 4: return 0.2
    if cont == 3: return 0.3
    if cont == 2: return 0.4
    if cont == 1: return 0.5
    if cont == 0: return 0.6

# dati lo spazione delle ipotesi e l'array P, inserisce in P le probabilità P(h) per ogni ipotesi. Tali probabilità si basano sulla specificità/generalità delle ipotesi
def setProbOrder(H, P):
    for i in range(len(H)):
        c = 0
        for j in range(6):
            if H[i][j] == '?':
                c += 1
        P.append(setProb(c))

# funzione che implementa la formula di Bayes. c è una probabilità di tipo P(D|h) e p è una probabilità di tipo P(h). Entrambe le probabilità sono riferite all'ipotesi h
def bayes(c, p):
    return c*p

# funzione che restituisce l'ipotesi hMAP dati lo spazio delle ipotesi e gli array C e P
def hMAP(H, C, P):
    max = -1000
    for i in range(len(H)):
        b = bayes(C[i], P[i])
        if max < b:
            max = b
            maxH = H[i]
    return maxH

setPriorProb(H, C, D) # stimo le probabilità a priori, le ipotesi consistenti con tutti gli esempi di training hanno probabilità 1, mentre tutte le altre 0
setProbOrder(H, P) # ipotesi più specifiche avranno probabilità più alte rispetto alle ipotesi più generali
print(hMAP(H, C, P)) # stampo l'ipotesi più specifica consistente con tutti gli esempi di training