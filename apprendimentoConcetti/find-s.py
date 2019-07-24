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

# inizializzazione di h come ipotesi più specifica, 0 significa che nessun valore è accettabile e ? che tutti i valori sono accettabili per un certo attributo

h = ['0','0','0','0','0','0']

# stampa della prima ipotesi

printH(h)

# algoritmo find s

for esempio in listaEsempi:
	if esempio['enjoysport'] == 'yes': # considero solo gli esempi positivi
		for attributo in range(0,6):
			# entra qui solamente per inizializzare l'ipotesi iniziale con gli attributi del primo esempio positivo
			if h[attributo] == '0':
				h[attributo] = list(esempio.values())[attributo]
			else:
				# generalizzazione dell'ipotesi corrente
				if h[attributo] != list(esempio.values())[attributo]:
					h[attributo] = '?'
	printH(h) # stampa dell'ipotesi corrente
