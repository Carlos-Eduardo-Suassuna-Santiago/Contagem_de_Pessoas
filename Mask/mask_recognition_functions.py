import os
import warnings #biblioteca usada para filtrar alguns alertas indesejados
import cv2 as cv
import numpy as np
import pandas as pd #biblioteca para manipulação de controle de dados
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier #biblioteca usada para fazer o treinamento do modelo
from sklearn.model_selection import train_test_split, GridSearchCV

warnings.filterwarnings('ignore')

def load_dataframe():
    """Carrega um dataframe Pandas com as imagens para o treinamento do modelo"""
    
    """Dicionário que receberá uma lista vazia que seré preenchida com os referentes dados"""
    dados = {
        "ARQUIVO":[],
        "ROTULO":[],
        "ALVO":[],
        "IMAGEM":[],
    }
    
    """caminhos onde as imagens do banco de dados estão armazenadas"""
    #caminho_com_mascara = "images\maskon"
    #caminho_sem_mascara = "images\maskoff"
    
    """Listar para ambos os conjuntos os arquivos contidos nos diretórios"""
    com_mascara = os.listdir("images\maskon")
    sem_mascara = os.listdir("images\maskoff")
    
    """Percorrer os arquivos e salvar suas caracteristicas no dicionario de dados"""
    for arquivo in com_mascara:
        dados["ARQUIVO"].append(f"images\maskon\{arquivo}")
        dados["ROTULO"].append(f"Com mascara")
        dados["ALVO"].append(1)
        img = cv.cvtColor(cv.imread(f"images\maskon\{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
    
    for arquivo in sem_mascara:
        dados["ARQUIVO"].append(f"images\maskoff\{arquivo}")
        dados["ROTULO"].append(f"Sem mascara")
        dados["ALVO"].append(0)
        img = cv.cvtColor(cv.imread(f"images\maskoff\{arquivo}"), cv.COLOR_BGR2GRAY).flatten()
        dados["IMAGEM"].append(img)
    
    """Usando a variavel dados como parametro para criar um dataframe do pandas"""
    dataframe = pd.DataFrame(dados)
    
    """Retorna o dataframe criado"""
    return dataframe

def train_test(dataframe):
    """Divide o dataframe em conjunto de treino e teste"""
    X = list(dataframe["IMAGEM"])
    y = list(dataframe["ALVO"])
    
    return train_test_split(X,y, train_size=0.4, random_state=13)

def pca_model(X_train):
    """Função onde é passada um conjunto de caracteristicas de treino e treinamos o modelo PCA o retornando no final do processo"""
    
    #PCA para retirada de features das imagens
    pca = PCA(n_components=30)
    pca.fit(X_train)
    
    return pca

def knn(X_train, y_train):
    """Função com o treinamento do modelo KNN, onde passamos o conjunto de treino e treinamos o modelo"""
    
    warnings.filterwarnings("ignore")
    
    #Modelo k-Nearest Neighbors
    grid_params = {
    "n_neighbors": [2, 3, 5, 11, 19, 23, 29],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattam", "cosine", "l1", "l2"]
    }
    
    knn_model = GridSearchCV(KNeighborsClassifier(), grid_params, refit=True)
    knn_model.fit(X_train, y_train)
    
    return knn_model