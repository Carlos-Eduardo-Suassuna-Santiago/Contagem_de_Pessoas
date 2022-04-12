import cv2 as cv
import mask_recognition_functions


cam = cv.VideoCapture(0) #permite que o opencv se conecte a webcan do pc
classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml") #Modelo para reconhecer faces

dataframe = mask_recognition_functions.load_dataframe() #Carregando dataframe com as imagens para treinamento

X_train, X_test, y_train, y_test = mask_recognition_functions.train_test(dataframe) #Dividindo conjuntos de treino e teste
pca = mask_recognition_functions.pca_model(X_train) #Modelo PCA para extração de features da imagem

X_train = pca.transform(X_train) #Conjunto de treino com features extraídas
X_test = pca.transform(X_test) #Conjunto de teste com features extraídas

knn = mask_recognition_functions.knn(X_train, y_train) #Treinando modelo classificatório KNN

#Rótulo das classificações
label = {
    0: "Sem mascara",
    1: "Com mascara"
}

if cam.isOpened():#testa se o python se conectou a webcan
    validacao, frame = cam.read()#lê a informação que está na camera
    
    while validacao:#loop para manter a camera capturando as imagens até que o comando seja encerrado
        validacao, frame = cam.read() 
        
        gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(gray_frame)
        
        for x,y,w,h in faces:
            gray_face = gray_frame[y:y+h, x:x+w] #Recortando região da face

            if gray_face.shape[0] >= 100 and gray_face.shape[1] >= 100:
                gray_face = cv.resize(gray_face, (160,160)) #Redimensionando
                vector = pca.transform([gray_face.flatten()]) #Extraindo features da imagem

                pred = knn.predict(vector)[0] #Classificando a imagem
                classification = label[pred]

                #Desenhando retangulos em torno da face
                if pred == 0:
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
                elif pred == 1:
                    cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
                
                #Escrevendo classificação e quantidade de faces vistas
                cv.putText(frame, classification, (x - 20,y + h + 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 2, cv.LINE_AA)
        
        cv.imshow("Video capture", frame)#mostra na tela uma janela com o titulo e a imagem da camera
        key = cv.waitKey(1)#faz com que a imagem gerada fique um tempo na tela/armazena uma tecla
        
        if key == 27:#encerrar a janela de exibição quando a tecla ESC for pressionada
            break

cam.release()#encerra a conexão com a camera
cv.destroyAllWindows()#garante que a imagem que estava sendo exibida na janela seja fechada