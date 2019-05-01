from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

#DATASET EXTRACTED FROM https://danbruno.net/writing/ocarina/

def vertical(img):

    # Binarizamos a imagem
    binarized = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    # Criamos uma cópia
    vertical = np.copy(binarized)
    # Pegamos a dimensão do array
    rows = vertical.shape[0]
    # O valor 20 foi através de experimentação e nos deu um melhor resultado final, quanto menor o valor maior foi a abstração do resultado final.
    # Para a imagem 4 o melhor valor ficou entre 90~100
    # um bom trabalho futuro seria a predição do melhor valor.
    verticalsize = rows / 20
    # Pegamos a estrutura do elemento/imagem
    verticalStructure = cv.getStructuringElement(cv.MORPH_CROSS, (1, int(verticalsize)))
    # Aplicamos a erosão na imagem (valorizamos os pixels com os menores valores, no caso, entre 0 e 1 sempre será escolhido o 0)
    vertical = cv.erode(vertical, verticalStructure)
    # Em seguida aplicamos a dilatação que é o inverso da Erosão, justamente para realçarmos os pontos das notas.
    vertical = cv.dilate(vertical, verticalStructure)
    # RESULTADO
    cv.imshow("FINAL", vertical)
    
    cv.waitKey(0)

def horizontal(img):
    bw = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols / 10
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (int(horizontal_size), 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    cv.imshow("FINAL", horizontal)
    cv.waitKey(0)

colored = cv.imread('dataset/1.png', cv.IMREAD_COLOR)
img = cv.cvtColor(colored, cv.COLOR_BGR2GRAY)
gray = cv.bitwise_not(img)

vertical(gray)
# Extra Horizontal
# horizontal(gray)
