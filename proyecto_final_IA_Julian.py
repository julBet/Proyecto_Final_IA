"""
Desarrollado por: Julian David Betancourt Marin
Proyecto Final IA.
Objetivos:  .    
"""
#%% Librerias 
import numpy as np
import cv2
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from IPython import get_ipython
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
#%% Limpiar pantalla y variables
print('\014')
get_ipython().magic('reset -sf')

#%% Funciones
# Funcion Invertir Fotos
def Invertir(Imagen,Rotacion=1):
    return cv2.flip(Imagen,Rotacion)
# Funcion Leer Bases de datos
def Leer_DB(Ruta, Formato,Nombre,h = 500,w = 290):
    """ 
    Inmporta una base de datos de imagenes desde una carpeta del PC 
    Paramtros: Ruta y Formato
    Ruta: Ruta de la carpeta que contiene las imagenes
    Formato: Formato de todas las imagenes
    Nombre: Nombre DataBase
    h, w dimensiones imagenes (default = 500 x 290)
    """
    import pathlib
    import cv2
    Imagen = pathlib.Path(Ruta)
    Formato="*."+ str(Formato)
    Base = {
        Nombre:list(Imagen.glob(Formato))
        }
    X = []
    for label, images in Base.items():
        for image in images:
            img = cv2.imread(str(image)) # Reading the image
            if img is not None:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (h, w))
                X.append(img)
    return X
# Funcion identificacion de las ROI's
def IdentificacionROI(Imagen,h,w,areaRoi,CteProp=1,Tmin=0,Umbral=0):
    import cv2
    from numpy import mean,setdiff1d
    hsv = cv2.cvtColor(Imagen,cv2.COLOR_BGR2HSV)
    H = hsv[:,:,0]
    H=H[h-areaRoi:h+areaRoi+1,w-areaRoi:w+areaRoi+1]
    H=setdiff1d(H,0)
    H=mean(H)
    return (H-Umbral)*CteProp+Tmin
#%% Importar Base de datos desde el computador
Sanos=Leer_DB(r'C:\Users\Julian\OneDrive - Pontificia Universidad Javeriana\Documents\Universidad\IA\DB\Sanos', 'png', 'Pacientes_Sanos',65,168)
Enfermos=Leer_DB(r'C:\Users\Julian\OneDrive - Pontificia Universidad Javeriana\Documents\Universidad\IA\DB\Enfermos', 'png', 'Pacientes_Sanos',65,168)
#%% Procesamiento de los datos
ROIs=[[25,20],[50,57],[50,10],[150,25]]
#%% invierte los pies Izquierdos para normalizar las fotos y crea los vectores de 10 caracteristicas
PSanos,PEnfermos,Datos=np.empty([0, 2*np.size(ROIs,axis=0)]),np.empty([0, 2*np.size(ROIs,axis=0)]),[]
for i in range(np.size(Enfermos,axis=0)):
    if i%2==0:
        Enfermos[i]=Invertir(Enfermos[i])
    for j in range(np.size(ROIs,axis=0)):
        Datos.append(IdentificacionROI(Enfermos[i], ROIs[j][0], ROIs[j][1], 4))
    if i%2==1:
        PEnfermos=np.concatenate((PEnfermos,np.array(Datos).reshape(1,2*np.size(ROIs,axis=0))),axis=0)
        Datos=[]
for i in range(np.size(Sanos,axis=0)):
    if i%2==0:
        Sanos[i]=Invertir(Sanos[i])
    for j in range(np.size(ROIs,axis=0)):
        Datos.append(IdentificacionROI(Sanos[i], ROIs[j][0], ROIs[j][1], 4))
    if i%2==1:
        PSanos=np.concatenate((PSanos,np.array(Datos).reshape(1,2*np.size(ROIs,axis=0))),axis=0)
        Datos=[]
        
print(np.isnan(np.sum(PSanos)))
print('Hola')
print(np.isnan(np.sum(PEnfermos)))     

# Conjuntos Entrenamiento, Validacion y Prueba por clase 70/15/15
# Creacion Etiquetas
YSanos = np.zeros((len(PSanos), 1))
YEnfermos = np.ones((len(PEnfermos),1))
# Concatenacion Etiquetas
PSanos=np.concatenate((PSanos,YSanos),axis=1)
PEnfermos=np.concatenate((PEnfermos,YEnfermos),axis=1)
# Conjuntos Lo mas balanceados que se puede
TrainingS, ValidS = model_selection.train_test_split(PSanos, test_size = int(0.3*len(PSanos)), train_size = int(0.7*len(PSanos)))
ValidationS, TestingS = model_selection.train_test_split(ValidS, test_size = int(0.5*len(ValidS)), train_size = int(0.5*len(ValidS)))

TrainingE, ValidE = model_selection.train_test_split(PEnfermos, test_size = int(0.3*len(PEnfermos)), train_size = int(0.7*len(PEnfermos)))
ValidationE, TestingE = model_selection.train_test_split(ValidE, test_size = int(0.5*len(ValidE)), train_size = int(0.5*len(ValidE)))

Training=np.concatenate((TrainingS,TrainingE),axis=0)
Testing=np.concatenate((TestingS,TestingE),axis=0)
Validation=np.concatenate((ValidationS,ValidationE),axis=0)
# Se borran los conjuntos sobrantes
del TrainingS,TrainingE
del TestingS,TestingE
del ValidationS,ValidationE
del ValidS,ValidE
# Etiquetas Categoricas
Y_Train = Training[:,np.size(Training,axis=1)-1]
Y_Valid = Validation[:,np.size(Training,axis=1)-1]
Y_Test = Testing[:,np.size(Training,axis=1)-1]
Y_Test = np.float64(Y_Test)
Y_Train_Dummies = pd.get_dummies(Y_Train)
Y_Valid_Dummies = pd.get_dummies(Y_Valid)
# Matrices Ya acondicionadas
Train=Training[:,0:np.size(Training,axis=1)-1]
Valid=Validation[:,0:np.size(Training,axis=1)-1]
Test=Testing[:,0:np.size(Training,axis=1)-1]

d=2*np.size(ROIs,axis=0)
Red = Sequential() # Se crea un modelo
Red.add(Dense(d, activation = 'sigmoid', input_shape = (d,))) #Capa Entrada
Red.add(Dense(50, activation = 'sigmoid'))                    #Capa Oculta
Caract=np.size(Y_Train_Dummies,axis=1)
Red.add(Dense(2, activation = 'softmax'))                #Capa Salida
# Optimizacion
Red.compile(optimizer = 'adam',
                  loss = 'mean_squared_error', #categorical_crossentropy #mean_squared_error
                  metrics = 'categorical_accuracy')
# Entrenamiento
Red.fit(Train,Y_Train_Dummies, epochs = 250,
              verbose = 1 , workers = 4 , use_multiprocessing=True,
              validation_data = (Valid,Y_Valid_Dummies))

Out_Prob = Red.predict(Test)
Out_Testing = Out_Prob.round()

Out_Testing = pd.DataFrame(Out_Testing)
Out_Testing = Out_Testing.values.argmax(1)

MatrC=confusion_matrix(Out_Testing,Y_Test)
print(f1_score(Out_Testing,Y_Test),
      '\n',
      MatrC)
# Se obtienen los pesos
class_weights = {0:len(PSanos)/(len(PSanos) + len(PEnfermos)),
                 1:len(PEnfermos)/(len(PSanos) + len(PEnfermos))}


# Datos globales (C-means)
Datos=np.concatenate((PSanos,PEnfermos),axis=0)
"""
Datos=np.concatenate((PSanos,PEnfermos),axis=0)        
K=np.cov(np.transpose(Datos))   
EigVals, EigVects = np.linalg.eig(K)
#Porcentajes
PorcAc = 100*np.cumsum(EigVals)/sum(EigVals)
PorcInd=[]
for i in range (len(EigVals)):
    PorcInd.append(100*EigVals[i,]/sum(EigVals))    
"""
Labels=np.concatenate((np.zeros((len(PSanos), 1)), np.ones((len(PEnfermos),1))),axis=0)
Datos=np.concatenate((Datos,Labels),axis=1)