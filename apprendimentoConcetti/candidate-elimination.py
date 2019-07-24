def removeInconsistentG(esempio,g): # dato un esempio di training rimuove da g tutte le ipotesi inconsistenti con l'esempio
	rimosso = False
	for h in g:
		rimosso = False
		for attribute in range(0,6):
			if rimosso == False:
				if h[attribute] != '?': # controllo solo gli attributi diversi da ? perché ? è sempre consistente
					if h[attribute] != list(esempio.values())[attribute]: # se trovo un attributo di h in g che rende negativo l'esempio positivo, allora rimuovo tale h da g
						g.remove(h)
						rimosso = True
	return g

def removeInconsistentS(esempio,s): # dato un esempio di training rimuove da s tutte le ipotesi inconsistenti con l'esempio
	rimosso = False
	for h in s:
		rimosso = False
		for attribute in range(0,6):
			if rimosso == False:
				if h[attribute] != '?' and (list(esempio.values())[attribute] == h[attribute]):
					if attribute == 5:
						s.remove(h)
						rimosso = True
	return s


def inconsistentS(esempio,s): # dato un esempio di training restituisce l'insieme delle ipotesi in s inconsistenti con l'esempio
	aggiunto = False
	inconsistentList = []
	for h in s:
		aggiunto = False
		for attribute in range(0,6):
			if aggiunto == False:
				if h[attribute] != list(esempio.values())[attribute]:
					inconsistentList.append(h)
					aggiunto = True
	return inconsistentList

def inconsistentG(esempio,g): # dato un esempio di training restituisce l'insieme delle ipotesi in g inconsistenti con l'esempio
	inconsistentList = []
	for h in g:
		for attribute in range(0,6):
			if h[attribute] == list(esempio.values())[attribute] or h[attribute] == '?':
				if attribute == 5:
					inconsistentList.append(h)
	return inconsistentList

def lessGeneral(sh,g): # data una ipotesi e g restituisce true se l'ipotesi è meno generale di tutti gli argomenti di G
	for h in g:
		for attribute in range(0,6):
			if h[attribute] != "?" and sh[attribute] == "?":
				return False
	return True			

def addHGeneralized(sh,esempio,s,g): # data una ipotesi inconsistente e un esempio, rende l'ipotesi consistente con l'esempio generalizzandola
	for attribute in range(0,6):
		if sh[attribute] == "0":
			sh[attribute] = list(esempio.values())[attribute]
		else:
			if sh[attribute]!=list(esempio.values())[attribute]:
				sh[attribute] = "?"
	if lessGeneral(sh,g):
		s.append(sh)
	return s

values = [ # lista dei possibili valori degli attributi delle istanze
['sunny','cloudy','rainy'],
['warm','cold'],
['normal','high'],
['strong','weak'],
['warm','cool'],
['same','change']
]

def takeOpposite(attribute,index): # ritorna un attributo differente ad attribute per l'indice di dizionario values passato
	for i in range(0,len(values[index])):
		if values[index][i] != attribute:
			return values[index][i]

def consistent(copyGh,s,attribute): # controlla se una specializzazione di G è consistente con S
	sh = s[0] # prendo l'ultima ipotesi in lista, che è sempre l'unica in lista
	if sh[attribute] != '?' and copyGh[attribute] == sh[attribute]:
		return True
	return False

def	addGSpecification(gh,esempio,g,s): # idea: genero tutte le specializzazioni di gh che sono consistenti con l'esempio di training e le aggiungo a G solo se sono consistenti con lo specific boundary
	for attribute in range(0,6):
		if gh[attribute] == '?': # devo rendere più specifico l'attributo ma che sia consistente con l'ipotesi in s
			copyGh = gh.copy()
			copyGh[attribute] = takeOpposite(list(esempio.values())[attribute], attribute) # costruisco una specializzazione di G che sia consistente con l'esempio di training
			if consistent(copyGh,s,attribute): # se la specializzazione generata è consistente con s allora la posso aggiungere a G
				g.append(copyGh)
	return g

def listWithoutH(h,s):
	l = s.copy() # per evitare side effect
	l.remove(h)
	return l

def removeMoreGeneral(s): # rimuove in s tutte le ipotesi più generali di ogni ipotesi, in pratica fa in modo che s contenga solo le ipotesi più specifiche
	for h in s:
		for h2 in listWithoutH(h,s):
			for attribute in range(0,6):
				if h2[attribute] == "?" and h[attribute] != "?":
					s.remove(h2)
				if h2[attribute] != "?" and h[attribute] == "?":
					s.remove(h)
	return s

def removeMoreSpecific(g): # rimuove in g tutte le ipotesi più specifiche di ogni ipotesi, in pratica fa in modo che g contenga solo le ipotesi più generali
	for h in g:
		for h2 in listWithoutH(h,g):
			for attribute in range(0,6):
				if h2[attribute] != "?" and h[attribute] == "?":
					g.remove(h2)
				if h2[attribute] == "?" and h[attribute] != "?":
					g.remove(h)
	return g

# funzione di stampa di un'ipotesi

def printH(h):
	string = "<"
	for attribute in range(0,len(h)):
		if attribute == len(h)-1:
			string = string + h[attribute]
		else:
			string = string + h[attribute] + ","
	string = string + ">"
	print(string)

# costruzione training set come da esempio

listaEsempi = []
listaEsempi.append({'sky':'sunny','airtemp':'warm','humidity':'normal','wind':'strong','water':'warm','forecast':'same','enjoysport':'yes'})
listaEsempi.append({'sky':'sunny','airtemp':'warm','humidity':'high','wind':'strong','water':'warm','forecast':'same','enjoysport':'yes'})
listaEsempi.append({'sky':'rainy','airtemp':'cold','humidity':'high','wind':'strong','water':'warm','forecast':'change','enjoysport':'no'})
listaEsempi.append({'sky':'sunny','airtemp':'warm','humidity':'high','wind':'strong','water':'cool','forecast':'change','enjoysport':'yes'})

# inizializzazione dello specific boundary

s = []
s.append(['0','0','0','0','0','0']) # s contiene s1 che rappresenta tutte le ipotesi più specifiche

# inizializzazione dello general boundary

g = []
g.append(['?','?','?','?','?','?']) # g contiene g1 che rappresenta tutte le ipotesi più generali 

# algoritmo candidate elimination


for esempio in listaEsempi:
	if esempio['enjoysport'] == 'yes': # caso esempio positivo
		g=removeInconsistentG(esempio,g) # rimozione da G delle ipotesi inconsistenti con l'esempio corrente
		for sh in inconsistentS(esempio,s): # per ogni ipotesi in S inconsistente con l'esempio corrente...
			s.remove(sh) # rimuovo l'ipotesi da s
			s=addHGeneralized(sh,esempio,s,g) # aggiungo ad s la minima generalizzazione dell'ipotesi che è consistente con l'esempio
			s=removeMoreGeneral(s) # faccio in modo che s contenga solo le ipotesi più specifiche
	else: # caso esempio negativo
		s=removeInconsistentS(esempio,s) # rimozione da S delle ipotesi inconsistenti con l'esempio corrente
		for gh in inconsistentG(esempio,g): # per ogni ipotesi in G inconsistente con l'esempio corrente...
			g.remove(gh) # rimuovo l'ipotesi da g
			g=addGSpecification(gh,esempio,g,s) # aggiungo a g la minima specializzazione dell'ipotesi che è consistente con l'esempio
			#g=removeMoreSpecific(g) # faccio in modo che g contenta solo le ipotesi più generali


print(s)
print(g)
