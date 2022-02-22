import numpy as np
import cv2
import os

# função para calcular o centro do contorno
def center(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# permite que o opencv se conecte a webcan do pc
cap = cv2.VideoCapture('1.mp4')
# comando usado para remover o fundo do video
fgbg = cv2.createBackgroundSubtractorMOG2()

# variavel para armazenar as pessoas que estao sendo identificadas
detects = []

# variaveis para traçar as linhas
posL = 150  # posição da linha na vertical(localização na tela)
ofsset = 30 # a distancia em ambas as posiçoes(cima/baixo) a partir da linha central

# variaveis com a posição das linhas
xy1 = (20, posL)
xy2 = (300, posL)  # largura da linha

# variaveis para saber a quantidade de pessoas que foram, voltaram e o total
up = 0
down = 0
total = 0

if cap.isOpened():  # testa se o python se conectou a webcan
    validacao, frame = cap.read()  # lê a informação que está na variavel cap

    while validacao:  # loop para manter a camera capturando as imagens até que o comando seja encerrado
        validacao, frame = cap.read()

        # Alterar a cor do frame pra escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #variavel que vai retirar a mascara, ou seja a diferença entre o frame anterior e o proximo
        fgmask = fgbg.apply(gray)
        # variavel para remover as sombras da mascara e deixar a imagem mais limpa dando um cv2.THRESH_BINARY para deixar so em preto e branco
        retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        # variavel criada para auxiliar no funcionamento da variavel opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # variavel para limpar ainda mais a imagem com a mascara removendo todos os resquicios de noise "sujeira" da imagem
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        # variavel que vai dilatar os objetos da imagem deixando eles maiores
        dilation = cv2.dilate(opening, kernel, iterations=8)
        # variavel para remover o noise de dentro da figura na imagem
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=8)
        # remover os contornos da imagem binaria do closing
        contours, hierachy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # adicionar a linha no centro da imagem com cor e espeçura
        cv2.line(frame, xy1, xy2, (255, 0, 0), 3)
        # traçar as linhas do ofsset
        cv2.line(frame, (xy1[0], posL-ofsset),(xy2[0], posL-ofsset), (255, 255, 0), 2) #linha de cima 
        cv2.line(frame, (xy1[0], posL+ofsset),(xy2[0], posL+ofsset), (255, 255, 0), 2) #linha de baixo

        # variavel pra contar o id de quem esta la dentro
        i = 0
        # laço paraa o aray com os contornos
        for cnt in contours:
            # pegando o tamanho e a area dos contornos
            (x, y, w, h) = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)

            # condição para ignorar alguns pequenos ruidos na imagem que possam atrapalhar na detecção das pessoas
            if int(area) > 3000:
                # centro do contorno
                centro = center(x, y, w, h)
                # comando para atribuir o id as pessoas e diferencialas
                cv2.putText(frame, str(i), (x+5, y+15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.circle(frame, centro, 4, (0, 0, 255), -1) 
                # contorno na imagem
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if len(detects) <= i:  # se o tamanho da lista for menor ou igual ao i
                    detects.append([])  # cria o item atual

                # condição para garantir que a detecção seja feita so entre as linhas do ofsset
                if centro[1] > posL-ofsset and centro[1] < posL+ofsset:
                    detects[i].append(centro)
                else:
                    detects[i].clear()
                i += 1

        # condição que checa se não tem nenhum elemento na imagem
        if len(contours) == 0:
            detects.clear()  # caso não tenha, a lista é esvaziada
        # se não, a lista vai ser percorrida para identificar a direção que a pessoa foi e armazenar esse dado
        else:
            for detect in detects:
                for (c, l) in enumerate(detect):  # c = count, l = line
                    # condição para saber se a pessoa passou pra parte de cima
                    if detect[c-1][1] < posL and l[1] > posL:
                        detect.clear()  # limpa o cache sempre que a condiçao é iniciada
                        up += 1  # adiciona +1 ao contador de pessoas que subiu
                        total += 1  # aciona +1 ao total
                        # traça uma linha verde
                        cv2.line(frame, xy1, xy2, (0, 255, 0), 5)
                        continue  # contnuar executando o codigo

                    # condição para saber se a pessoa passou pra parte de baixo
                    if detect[c-1][1] > posL and l[1] < posL:
                        detect.clear()  # limpa o cache sempre que a condiçao é iniciada
                        down += 1  # adiciona +1 ao contador de pessoas que desceu
                        total += 1  # aciona +1 ao total
                        # traça uma linha azul
                        cv2.line(frame, xy1, xy2, (0, 0, 255), 5)
                        continue  # contnuar executando o codigo

                    # condição para identificar quando a linha central for cruzada
                    if c > 0:
                        cv2.line(frame, detect[c-1], l, (0, 0, 255), 1)

        # configuração com os textos e a contagem referente a cada opção
        cv2.putText(frame, "TOTAL: "+str(total), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, "SUBINDO: "+str(up), (10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "DESCENDO: "+str(down), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # mostra na tela uma janela com o titulo e a imagem da camera
        cv2.imshow("Video", frame)
        '''
        cv2.imshow("Video de teste cinza", gray)
        cv2.imshow("Video de teste com mascara", fgmask)   
        cv2.imshow("Video de teste com threshold", th) 
        cv2.imshow("Video de teste com Opening", opening)
        cv2.imshow("Video de teste com dilatação", dilation)
        cv2.imshow("Video de teste com closing", closing)
        '''
        key = cv2.waitKey(30) #faz com que a imagem gerada ficque um tempo na tela/armazena uma tecla

        if key == 27: #encerrar a janela de exibição quando a tecla ESC for pressionada
            break

cap.release() #encerra a conexão com a cap
cv2.destroyAllWindows()# garante que a imagem que estava sendo exibida na janela seja fechada
os.system('cls')