import csv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import numpy as np
from sklearn import svm

# costruzione del dataset a partire dal dataset fornito da kaggle
x = []
y = []
temp = []
input_file = csv.DictReader(open("train.csv"))
for row in input_file:
    temp = []
    for i in row:
        if i == 'Survived':
            y.append(int(row[i]))
        else:
            if i == 'Age' or i == 'Pclass' or i == 'Parch' or i == 'Fare' or i == 'SibSp':
                if row[i] == '':
                    row[i] = -1
                row[i] = float(row[i])
                temp.append(row[i])
            if i == "Embarked":
                if row[i] == '':
                    row[i] = -1
                if row[i] == 'S':
                    row[i] = 1
                if row[i] == 'Q':
                    row[i] = 3
                if row[i] == 'C':
                    row[i] = 2
                temp.append(row[i])
            if i == "Sex":
                if row[i] == 'male':
                    row[i] = 1
                else:
                    row[i] = 0
                temp.append(row[i])
    x.append(temp)

# divisione del dataset tra training set e test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# per l'imbarco occorre imputare i due valori mancanti, mentre per l'età i 177 mancanti -> è stato usato most_frequent
imp = SimpleImputer(missing_values=-1.0, strategy='most_frequent')
imp.fit(x_train)
x_train = imp.transform(x_train)
x_test = imp.transform(x_test)

# occorre codificare le variabili categoriche (classe, sesso, imbarco) -> sono già codificate


# occorre ora standardizzare le variabili continue (età, sibsp, parch, fare)
x_train_continue_feature = x_train[:, [2,3,4,5]]
x_test_continue_feature = x_test[:, [2,3,4,5]]
scaler = preprocessing.StandardScaler().fit(x_train_continue_feature)
x_train_continue_feature = scaler.transform(x_train_continue_feature)
x_test_continue_feature = scaler.transform(x_test_continue_feature)
x_train = np.concatenate((x_train[:,[0,1,6]],x_train_continue_feature), axis=1)
x_test = np.concatenate((x_test[:,[0,1,6]],x_test_continue_feature), axis=1)

'''
# utilizzo binarizer per dividere le variabili continue

x_train_price = x_train[:, [5]]
x_test_price = x_test[: ,[5]]
bin_price = preprocessing.Binarizer(100.0).fit(x_train_price)
x_train_price = bin_price.transform(x_train_price)
x_test_price = bin_price.transform(x_test_price)

x_train_age = x_train[:, [2]]
x_test_age = x_test[: ,[2]]
bin_age = preprocessing.Binarizer(60.0).fit(x_train_age)
x_train_age = bin_age.transform(x_train_age)
x_test_age = bin_age.transform(x_test_age)

x_train_s = x_train[:, [3]]
x_test_s = x_test[: ,[3]]
bin_s = preprocessing.Binarizer(1.0).fit(x_train_s)
x_train_s = bin_s.transform(x_train_s)
x_test_s = bin_s.transform(x_test_s)

x_train_p = x_train[:, [4]]
x_test_p = x_test[: ,[4]]
bin_p = preprocessing.Binarizer(1.0).fit(x_train_p)
x_train_p = bin_p.transform(x_train_p)
x_test_p = bin_p.transform(x_test_p)

x_train = np.concatenate((x_train[:,[0,1,6]],x_train_age,x_train_s,x_train_p,x_train_price), axis=1)
x_test = np.concatenate((x_test[:,[0,1,6]],x_test_age,x_test_s,x_test_p,x_test_price), axis=1)

enc = preprocessing.OneHotEncoder(categories='auto').fit(x_train)
x_train = enc.transform(x_train)
x_test = enc.transform(x_test)
'''

# costruzione del modello di predizione
svc = svm.SVC(kernel="rbf", gamma=0.1)
svc.fit(x_train,y_train)
print(svc.score(x_test,y_test))


'''
# istruzioni utilizzate per l'analisi dei dati
f = 0
m = 0
min = 0
mag = 0
ad = 0
st = 0
sd = 0
rd = 0
c = s = q = 0
z = 0
spos = 0

# calcolo numero donne e uomini
for i in range(len(x)):
    if x[i][3] == "female":
        f += 1
    if x[i][3] == "male":
        m += 1

# codice per capire la correlazione tra i dati
for i in range(len(x)):
    if x[i][3] == "female" and y[i] == 1: # numero donne sopravvisute
        f += 1
    if x[i][3] == "male" and y[i] == 1: # numero uomini sopravvisuti
        m += 1
    if x[i][4] <= 17.9  and y[i] == 1: # conteggio minorenni
        min += 1
    if x[i][4] >= 18.0 and x[i][4] <= 65.0 and y[i] == 1: # conteggio adulti
        ad += 1
    if x[i][4] >= 65 and y[i] == 1: # conteggio anziani
        mag += 1
    if x[i][1] == 1 and y[i] == 1:
        st += 1
    if x[i][1] == 2 and y[i] == 1:
        sd += 1
    if x[i][1] == 3 and y[i] == 1:
        rd += 1
    if x[i][10] == 'C' and y[i] == 1:
        c += 1
    if x[i][10] == 'S' and y[i] == 1:
        s += 1
    if x[i][10] == 'Q':
        q += 1
    if x[i][8] <= 100:
        z += 1
    if x[i][6] < 1.0:
        spos += 1
'''