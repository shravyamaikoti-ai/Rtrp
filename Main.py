from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from keras_dgl.layers import GraphCNN #loading GNN class
import keras.backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten
import os
from sklearn.preprocessing import StandardScaler


main = Tk()
main.title("NetFense: Adversarial Defenses against Privacy Attacks on Neural Networks for Graph Data")
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, y_train, y_test, labels, nodes, labels, source, target, edges

def uploadDataset(): 
    global filename, X, Y, nodes, labels, source, target, edges
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n")
    nodes = pd.read_csv("Dataset/nodes.csv")
    edges = pd.read_csv("Dataset/edges.csv")

    source = edges['source'].ravel()
    target = edges['target'].ravel()
    labels = nodes['label'].ravel()
    text.insert(END,"Nodes Dataset\n")
    text.insert(END,str(nodes)+"\n\n")
    text.insert(END,"Edges Dataset\n")
    nodes = nodes['nodeID'].ravel()
    text.insert(END,str(edges)+"\n\n")
    text.insert(END,"Total edges found in Normal Dataset : "+str(edges.shape[0]))
    
def getLabel(src):
    label = 0
    for i in range(len(nodes)):
        if nodes[i] == src:
            label = labels[i]
            break
    return label    

def purturbedData():
    global nodes, labels, source, target, edges, X, Y
    text.delete('1.0', END)
    random_edges = np.random.randint(low = 1, high = np.max(source), size = 30)
    perturbed = []
    for i in range(len(target)):
        if target[i] not in random_edges:
            perturbed.append([source[i], target[i], getLabel(source[i])])    
    random_source = np.random.randint(low = 1, high = np.max(source), size = 80)    
    random_target = np.random.randint(low = 1, high = np.max(target), size = 80)
    for i in range(len(random_source)):
        perturbed.append([random_source[i], random_target[i], getLabel(random_source[i])])
    perturbed = pd.DataFrame(perturbed, columns=['source', 'edge', 'label'])
    perturbed.to_csv("perturbed.csv", index = False)
    text.insert(END,"Perturbed Data\n\n")
    text.insert(END,str(perturbed)+"\n\n")
    perturbed = perturbed.values
    X = perturbed[:,0:perturbed.shape[1]-1]
    Y = perturbed[:,perturbed.shape[1]-1]
    text.insert(END,"Total edges found in Normal Dataset : "+str(edges.shape[0])+"\n")
    text.insert(END,"Total edges found after Perturbed Data : "+str(perturbed.shape[0]))

def trainGNN():
    global X_train, X_test, y_train, y_test, X, Y
    text.delete('1.0', END)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #train GNN model by uisng GraphCNN class from keras
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    gcnn_model = Sequential()
    #define GNN layer to build graph from input features
    gcnn_model.add(GraphCNN(128, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    gcnn_model.add(GraphCNN(64, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    gcnn_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    #defining output layer
    gcnn_model.add(Dense(units = 256, activation = 'relu'))
    gcnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    gcnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #compile and train the model
    if os.path.exists("model/gcnn_weights.h5") == False:
        hist = gcnn_model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data = (X_test, y_test), verbose=1)
        gcnn_model.save_weights("model/gcnn_weights.h5")
        f = open('model/gnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        gcnn_model.load_weights("model/gcnn_weights.h5")
    #define predict function to perform prediction on test data and calculate accuracy and other metrics
    pred = []
    for i in range(len(X_test)):
        temp = []
        temp.append(X_test[i])
        temp = np.asarray(temp)
        predict = gcnn_model.predict(temp, batch_size=1)
        predict = np.argmax(predict)
        pred.append(predict)
    predict = np.asarray(pred)    
    y_test1 = np.argmax(y_test, axis=1)  
    a = accuracy_score(y_test1,predict)
    text.insert(END,"GNN Accuracy on Perturbed Data = "+str(1 - a)+"\n")
    text.insert(END,"GNN Accuracy affected margin on Perturbed Data = "+str(a)+"\n\n")
    
def extensionGNN():
    global X_train, X_test, y_train, y_test
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    text.insert(END,"Extension Encrypted Features : "+str(X_train)+"\n\n")
    #train GNN model by uisng GraphCNN class from keras
    graph_conv_filters = np.eye(1)
    graph_conv_filters = K.constant(graph_conv_filters)
    extension_model = Sequential()
    #define GNN layer to build graph from input features
    extension_model.add(GraphCNN(128, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    extension_model.add(GraphCNN(64, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    extension_model.add(GraphCNN(1, 1, graph_conv_filters, input_shape=(X_train.shape[1],), activation='elu', kernel_regularizer=l2(5e-4)))
    #defining output layer
    extension_model.add(Dense(units = 256, activation = 'relu'))
    extension_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    extension_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    #compile and train the model
    if os.path.exists("model/extension_weights.h5") == False:
        hist = extension_model.fit(X_train, y_train, batch_size=1, epochs=10, validation_data = (X_test, y_test), verbose=1)
        extension_model.save_weights("model/extension_weights.h5")
        f = open('model/extension_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        extension_model.load_weights("model/extension_weights.h5")
    #define predict function to perform prediction on test data and calculate accuracy and other metrics
    pred = []
    for i in range(len(X_test)):
        temp = []
        temp.append(X_test[i])
        temp = np.asarray(temp)
        predict = extension_model.predict(temp, batch_size=1)
        predict = np.argmax(predict)
        pred.append(predict)
    predict = np.asarray(pred)    
    y_test1 = np.argmax(y_test, axis=1)  
    a = accuracy_score(y_test1,predict)
    text.insert(END,"Extension GNN Accuracy on Encrypted & Perturbed Data = "+str(1 - a)+"\n")
    text.insert(END,"Extension GNN Accuracy affected margin on Encrypted & Perturbed Data = "+str(a)+"\n\n")

def graph():
    f = open('model/gnn_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    gnn_loss = data['val_loss']
    f = open('model/extension_history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    extension_loss = data['val_loss']
    plt.figure(figsize=(8,5))
    plt.grid(True)
    plt.xlabel('EPOCH')
    plt.ylabel('Training Loss')
    plt.plot(gnn_loss, 'ro-', color = 'green')
    plt.plot(extension_loss, 'ro-', color = 'blue')
    plt.legend(['GNN Perturbed Loss', 'Extension Encrypted Perturbed Loss'], loc='upper left')
    plt.title('Propose Perturbed & Extension Loss Graph')
    plt.show()

def close():
    main.destroy()

font = ('times', 15, 'bold')
title = Label(main, text='NetFense: Adversarial Defenses against Privacy Attacks on Neural Networks for Graph Data')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload CiteSeer Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Perturbed Dataset", command=purturbedData)
processButton.place(x=20,y=150)
processButton.config(font=ff)

splitButton = Button(main, text="Trained GNN on Perturbed Data", command=trainGNN)
splitButton.place(x=20,y=200)
splitButton.config(font=ff)

rfButton = Button(main, text="Extension GNN on Perturbed & Encrypted", command=extensionGNN)
rfButton.place(x=20,y=250)
rfButton.config(font=ff)

predictButton = Button(main, text="Comparison Graph", command=graph)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=350)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=360,y=100)
text.config(font=font1)

main.config(bg='forestgreen')
main.mainloop()
