#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto2 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
import numpy as np
import skimage.io as io
import skimage.exposure as expo
from scipy.signal import correlate2d
from skimage.color import rgb2gray
from skimage.filters import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
import os
import cv2
##con la función indicada en la guía se crea filtro gaussiano
def gaussian_kernel(size, sigma):
    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/(2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
    return g
def MyNormCCorrelation_201719942_201822262(image, kernel, boundary_condition="fill"):
    """
    Función para la cross-correlación normalizada de una imagen y un kenerl dados. Se aplica la condición de frontera  deseada por el usuario
    :param image: arreglo de la imagen
    :param kernel: arreglo del filtro con el cual se realizará la cross-correlación normalizada
    :param boundary_condition: str "fill", "valid" o "symm" que determinará la condición de frontera que se le aplicará en la cross-correlación normalizada
    :return: arreglo de la imagen una vez realizada la cross-correlación normalizada
    """
    normalized_ccorrelation=0 # se inicializa variable respuesta de crosscorrelación que se modificará según el método de frontera
    a=round((len(kernel)-1)/2) # cálculo de a y b
    b=round((len(kernel[0])-1)/2)
    I0 = np.mean(kernel)
    I0_desvest = np.std(kernel)
    if boundary_condition=="fill": # se realizan serie de condicionales los cuales se aplicarán según la condición de frontera que igresa por parámetro
        fill_image = np.pad(image.copy(), a, mode="constant") #copia de la imagen para generar bordes en la imagen
        normalized_ccorrelation = np.zeros((len(image)+a*2, len(image[0])+b*2)) # se crea matriz para almacenar cross-correlación con tamaño dependiente de a y b          #print(CCorrelation.shape)
        for filas in range(0+a,len(fill_image)-a): # recorrido para realizar la cross-correlación empezando sobre pixel central dado por a y b
            for columnas in range(0+b,len(fill_image[0])-b):
                mini_matriz = fill_image[filas-a:filas+a+1, columnas-b:columnas+b+1]
                I1 = np.mean(mini_matriz)
                I1_desvest = np.std(mini_matriz)
                i_fila=filas-a # contador de filas para tomar vecinos de pixel central
                factor_1_num = 0
                factor_2_num = 0
                for multi_i in range(len(kernel)): # recorrido por el tamaño del que kernal para sacar prom del pixelcentral evaluda
                    j_column=columnas-b # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += kernel[multi_i][multi_j] - I0
                        factor_2_num += fill_image[i_fila][j_column] - I1
                        j_column+=1 #aumento contador columnas
                    i_fila+=1 # aumento contador filas
                normalized_ccorrelation[filas][columnas] = (factor_1_num*factor_2_num)/(I1_desvest*I0_desvest) # cálculo cross-correlación normalizada
    elif boundary_condition=="symm": # para condición de frontera symm # se realiza arreglo de la entrega pasada de esta condición de frontera
        imagen_marco = np.pad(image.copy(), a, mode="symmetric") # reflejo de bordes con ayuda de librería numpy pad y modo symmetric del tamaño que indique el kernel de entrada
        normalized_ccorrelation = np.zeros((len(image) , len(image[0]) ))  # se crea matriz para almacenar cross-correlación con tamaño dependiente de a y b          #print(CCorrelation.shape)
        for filas in range(0 + a, len(imagen_marco) - a):  # recorrido para realizar la cross-correlación empezando sobre pixel central dado por a y b
            for columnas in range(0 + b, len(imagen_marco[0]) - b):
                mini_matriz = imagen_marco[filas - a:filas + a + 1, columnas - b:columnas + b + 1]
                I1 = np.mean(mini_matriz)
                I1_desvest = np.std(mini_matriz)
                factor_1_num = 0
                factor_2_num = 0
                i_fila = filas - a  # contador de filas para tomar vecinos de pixel central
                for multi_i in range(len(kernel)):  # recorrido por el tamaño del que kernal para sacar prom del pixelcentral evaluda
                    j_column = columnas - b  # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += kernel[multi_i][multi_j] - I0
                        factor_2_num += imagen_marco[i_fila][j_column] - I1
                        j_column += 1  # aumento contador columnas
                    i_fila += 1  # aumento contador filas
                normalized_ccorrelation[filas][columnas] = (factor_1_num * factor_2_num) / ( I1_desvest * I0_desvest)  # cálculo cross-correlación normalizada
        normalized_ccorrelation= np.pad(normalized_ccorrelation, a, mode="symmetric") # se añaden bordes finales reflejados de la crosscorrelación con ayuda de la función de numpy utilizada para el marco inicial
    elif boundary_condition=="valid": # método de frontera valid
        normalized_ccorrelation=np.zeros((len(image)-a*2,len(image[0])-b*2)) # matriz para almacenar respuesta
        for filas in range(0+a,len(image)-a): # recorrido para cálculo crosscorrelación como en métodos anterioes
            for columnas in range(0+b,len(image[0])-b):
                mini_matriz = image[filas - a:filas + a + 1, columnas - b:columnas + b + 1]
                I1 = np.mean(mini_matriz)
                I1_desvest = np.std(mini_matriz)
                factor_1_num = 0
                factor_2_num = 0
                i_fila=filas-a
                for multi_i in range(len(kernel)):
                    j_column=columnas-b
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += kernel[multi_i][multi_j] - I0
                        factor_2_num += image[i_fila][j_column] - I1
                        j_column+=1
                    i_fila+=1
                normalized_ccorrelation[filas - a][columnas - b] = (factor_1_num * factor_2_num) / ( I1_desvest * I0_desvest)  # cálculo cross-correlación normalizada
    return normalized_ccorrelation
def error_cuadrado(imageref,imagenew):
    """
    Calculo error cruadrático medio
    :param imageref: arreglo con imagen de referencia
    :param imagenew: arreglo con imagen nueva para la cual se desea conocer el error con respecto a la de referencia. Mismo tamaño de imagen de referencia
    :return: error cuadrpatico medio
    """
    suma_error=0 # variable para almacenar suma de diferencia de cuadrados
    for i in range(len(imageref)): #recorrido por las dimensiones de la imagen de referencia que
        for j in range(len(imageref[0])): #recorrido por columnas
            suma_error+=(imageref[i][j]-imagenew[i][j])**2 # suma a la variable suma_error la resta al cuadrado de la posición evaluada en ambas imagenes
    error=suma_error/(len(imageref)*len(imageref[0])) # división de la suma de restas al cuadrado calculada previamente entre las dimensiones de la imagen (cantidad de pixeles)
    return error
kernel_a=np.array([[1,1,1],[1,1,1],[1,1,1]])
rosas=io.imread("roses.jpg")
#rosas_noise=io.imread("noisy_roses.jpg")
rosas=rgb2gray(rosas) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
prueba_ka_s=MyNormCCorrelation_201719942_201822262(rosas,kernel_a,boundary_condition="fill")
prueba_scipy_s=correlate2d(rosas,kernel_a,boundary="fill")
error_ka_s=error_cuadrado(prueba_scipy_s, prueba_ka_s)
print("error kernel a symm")
print(error_ka_s)
print("\n",prueba_scipy_s)
print("\n",prueba_ka_s)
print(prueba_ka_s.shape)
print(prueba_scipy_s.shape)
filtro_Gauss_punto3=gaussian_kernel(5,1) # se crea filtro de Gauss con función proporcionada en el enunciado. Filtro de 5x5b con sigma de 1
cross_filtroGauss=MyNormCCorrelation_201719942_201822262(rosas,filtro_Gauss_punto3) # cross-correlación con condición de frontera fill de imagen con ruido y filtro de Gauss creado previamente
plt.figure("Original_kernel_b_Gauss") # figura para mostrar imagen original e imagen filtrada con filtro de gauss decrito previamente  y kernel b
plt.subplot(1,3,1) # cada imagen tiene su respectivo subplot, remoción de ejes, título y es visualizado con mapa de color de rises
plt.title("Imagen original escala grises")
plt.imshow(prueba_ka_s,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con kernel b")
plt.imshow(prueba_scipy_s,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n5x5 y σ = 1")
plt.imshow(rosas,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
##PROBLEMA BIOMÉDICA
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
reference1=io.imread("reference1.jpg") # carga de las diferentes imágenes a trabajar en el problema biomédico
reference2=io.imread("reference2.jpg")
reference3=io.imread("reference3.jpeg")
parasitized=io.imread("Parasitized.png")
uninfected=io.imread("Uninfected.png")
plt.figure("HistogramasMalaria")
plt.subplot(2,3,1)
plt.title("Imagen reference1.jpg")
plt.imshow(rgb2gray(reference1),cmap="gray")
plt.axis("off")
plt.subplot(2,3,4)
plt.title("Histograma reference1.jpg")
plt.hist(rgb2gray(reference1).flatten(),bins=256)
plt.tight_layout()
plt.subplot(2,3,2)
plt.title("Imagen reference2.jpg")
plt.imshow(rgb2gray(reference2),cmap="gray")
plt.axis("off")
plt.subplot(2,3,5)
plt.title("Histograma reference2.jpg")
plt.hist(rgb2gray(reference2).flatten(),bins=256)
plt.tight_layout()
plt.subplot(2,3,3)
plt.title("Imagen reference3.jpeg")
plt.imshow(rgb2gray(reference3),cmap="gray")
plt.axis("off")
plt.subplot(2,3,6)
plt.title("Histograma reference3.jpeg")
plt.hist(rgb2gray(reference3).flatten(),bins=256)
plt.tight_layout()
plt.tight_layout()
plt.show()
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
def myImagePreprocessor(image, target_hist, action="show"):
    """
    Preprocesamiento de imagen de entrada con imagen target
    :param image: Imagen para procesar
    :param target_hist: imagen con histograma de intereés para procesar imagen deseada
    :param action: str "show" o "save" que definirá el usuario según lo que desee
    :return: arreglo de imagen preprocesada (especificada)
    """
    image=rgb2gray(image) # se convierten imágenes a blanco y negro para ser procesadas
    target_hist=rgb2gray(target_hist)
    matched_image=expo.match_histograms(image,target_hist) # especificación de la imagen con función match_histograms. 1er parámetro imagen a especificar, 2do paarámetro imagen que tiene histograma con el cual se especificará imagen
    equa_ref=expo.equalize_hist(target_hist)  # ecualización de imagen con histograma deseado
    equa_image=expo.equalize_hist(image) # ecualización de imagen para processar con equalize_hist
    plt.figure() # figura para imágenes originales, ecualizadas y la especificada con sus respectivos histogramas
    plt.subplot(5,2,1) # subplots áta mostrar las diferentes imágenes con un color map gris con su respectivo título y sin ejes
    plt.title("Imagen original")
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    plt.subplot(5,2,3)
    plt.title("Imagen original ecualizada")
    plt.imshow(equa_image,cmap="gray")
    plt.axis("off")
    plt.subplot(5,2,5)
    plt.title("Imagen referencia")
    plt.imshow(target_hist,cmap="gray")
    plt.axis("off")
    plt.subplot(5,2,7)
    plt.title("Imagen referencia ecualizada")
    plt.imshow(equa_ref,cmap="gray")
    plt.axis("off")
    plt.subplot(5, 2,9)
    plt.title("Imagen especificada")
    plt.imshow(matched_image,cmap="gray")
    plt.axis("off")
    plt.subplot(5,2,2) # serie de subplots para las diferentes imágenes a mostrar. Se indica su título, se utiliza .flatten para realizar los histogramas y se indica el número de bins a utilizar para el histograma (256)
    plt.title("Histograma original")
    plt.hist(image.flatten(),bins=256)
    plt.subplot(5,2,4)
    plt.title("Histograma original ecualizada")
    plt.hist(equa_image.flatten(),bins=256)
    plt.subplot(5,2,6)
    plt.title("Histograma referencia")
    plt.hist(target_hist.flatten(),bins=256)
    plt.subplot(5,2,8)
    plt.title("Histograma referencia ecualizada")
    plt.hist(equa_ref.flatten(),bins=256)
    plt.subplot(5, 2, 10)
    plt.title("Histograma especificada")
    plt.hist(matched_image.flatten(),bins=256)
    plt.tight_layout()
    if action=="show": # si por parámetro se indica que se desea mostrar la figura se realiza un plt.show()
        plt.show()
    elif action=="save": # si por parámetro se indica que se quiere guardar la imagen esta se guarda con plt.savefig y se cierra con plt.close()
        plt.savefig("Preprocesamiento")
        plt.close()
    return matched_image
ref1=myImagePreprocessor(parasitized,reference1,action="save")
ref1_show=myImagePreprocessor(parasitized,reference1,action="show")
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
ref2=myImagePreprocessor(parasitized,reference2,action="save")
ref2_show=myImagePreprocessor(parasitized,reference2,action="show")
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
ref3=myImagePreprocessor(parasitized,reference3,action="save")
ref3_show=myImagePreprocessor(parasitized,reference3,action="show")
#input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
mejor_especif=myImagePreprocessor(parasitized,reference2,action="save")
mejor_especif_show=myImagePreprocessor(uninfected,reference2,action="show")

##
#img1 = io.imread(os.path.abspath("2/Proyecto2_Entrega2/noisy1.jpg"))
#img2 = io.imread(os.path.abspath("2/Proyecto2_Entrega2/noisy2.jpg"))
img1=io.imread("noisy1.jpg")
img2=io.imread("noisy2.jpg")
imag_ruido1 = rgb2gray(img1) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
imag_ruido2 = rgb2gray(img2) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
plt.figure()
plt.subplot(2,1,1)
plt.title("Noisy 1")
plt.imshow(imag_ruido1, cmap="gray")
plt.subplot(2,1,2)
plt.title("Noisy 2")
plt.imshow(imag_ruido2, cmap="gray")
plt.tight_layout
plt.show()
##
# Filtro medio adaptativo
def MyAdaptMedian_201719942_201822262(gray_image, window_size, max_window_size):
    imagen_marco=np.pad(gray_image.copy(),int(max_window_size)//2,mode="symmetric")# copia de la imagen con bordes de modo symm del tamaño máximo que podría llegar a tener la ventana, se utiliza la función de numpy pad
    filas,columnas=len(gray_image),len(gray_image[0])
    filtered_image=np.zeros((filas, columnas))
    desfase=int(max_window_size)//2
    for fila in range(filas):
        for columna in range(columnas):
            fila_pix_original,column_pix_original=fila+desfase,columna+desfase
            size_actual = window_size
            while size_actual <= max_window_size:
                marco_actual = int(size_actual) // 2
                mini_matriz = imagen_marco[fila_pix_original - marco_actual:   (fila_pix_original) + marco_actual + 1,column_pix_original - marco_actual:    column_pix_original + marco_actual + 1]
                z_min = np.min(mini_matriz.flatten())  # variable para el mínino, se usa la funcion de numpy min
                z_max = np.max(mini_matriz.flatten())  # variable para el máximo, se usa la funcion de numpy max
                z_med = np.median(mini_matriz.flatten())  # variable para la mediana, se usa la función de numpy median
                A1 = z_med - z_min
                A2 = z_med - z_max
                if A1>0 and A2<0:
                    centro=imagen_marco[fila_pix_original,column_pix_original]
                    B1=centro-z_min
                    B2=centro-z_max
                    if B1>0 and B2<0:
                        filtered_image[fila][columna] = centro
                        #print("centro")
                    else:
                        filtered_image[fila][columna] = z_med
                        #print("opc2")
                    break
                else:
                    if size_actual==max_window_size:
                        filtered_image[fila][columna]=z_med
                        #print("opcmax")
                    size_actual+=1
    return filtered_image
#median(imag_ruido1)
## Pruebe con 3 tamaños de ventana diferentes aplicar su función de filtro mediano adaptativo a las imágenes con ruido. Muestre sus experimentos en un subplot con títulos las imágenes con ruido y las respuestas después del filtrado.
plt.figure()
plt.subplot(1,3,1)
plt.title("Filtro mediano adaptativo\ntamaño ventana = 3")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,3,35), cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Filtro mediano adaptativo\ntamaño ventana = 9")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,9,35), cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Filtro mediano adaptativo\ntamaño ventana = 19")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,19,35), cmap="gray")
plt.axis("off")
plt.tight_layout
plt.show()
##Pruebe filtrar ambas imágenes con distintos kernels Gaussianos, que varíen en tamaño y valor de σ
filtro1_Gauss,filtro2_Gauss , filtro3_Gauss , filtro4_Gauss, filtro5_Gauss , filtro6_Gauss=gaussian_kernel(3,1) ,gaussian_kernel(3,50) ,  gaussian_kernel(3,100) , gaussian_kernel(3,5) , gaussian_kernel(5,5) , gaussian_kernel(7,5)# creación de filtros Gaussianos con tamaño constante y sigma variable  #se crean tres diferentes filtros de Gauss con función dada variando el tamaño y mateniendo el sigma constante
plt.figure("R1_G1")
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro1_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure("R2_G2}1")
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro1_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro2_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro2_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro3_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro3_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro4_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro4_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro5_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro5_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido1,filtro6_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyCCorrelation_201719942_201822262(imag_ruido2,filtro6_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")

##Muestre en un subplot el resultado del mejor filtrado para cada imagen y diga con qué filtro y qué parámetros se produce la mejor imagen.
plt.figure()
plt.subplot(1,2,1)
plt.title("noisy1: Filtro mediano adaptativo tamaño ventana = 3")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,3,5), cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("noisy2: Filtro Gauss 5x5 y σ = 5")
plt.imshow(MyCCorrelation_201719942_201822262(imag_ruido2,filtro5_Gauss,boundary_condition="fill"), cmap="gray")
plt.axis("off")
plt.tight_layout
plt.show()





##PRUEBAS
print(error_cuadrado(median(imag_ruido1,mode="reflect"),MyAdaptMedian_201719942_201822262(imag_ruido1,3,51)))
print(error_cuadrado(median(imag_ruido1,mode="mirror"),MyAdaptMedian_201719942_201822262(imag_ruido1,3,51)))
##plt.figure()
plt.subplot(2,1,1)
plt.title("MyAdapt")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,3,5), cmap="gray")
plt.subplot(2,1,2)
plt.title("Teoría")
plt.imshow(median(imag_ruido1,mode="reflect"), cmap="gray")#cv2.medianBlur(imag_ruido1,3), cmap="gray")
plt.tight_layout
plt.show()
##
kernel_c=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
c=np.pad(kernel_c,2,mode="symmetric")
print(c[   ( 2 ) -   1     :   ( 2 )  +  1  +1,2  -   1     :    2   + 1 + 1])

print(kernel_c)
print(c)
##