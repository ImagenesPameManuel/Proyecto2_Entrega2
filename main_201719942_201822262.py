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
from skimage.feature import match_template
import matplotlib.pyplot as plt
import os
#con la función indicada en la guía se crea filtro gaussiano
def gaussian_kernel(size, sigma):
    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/(2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
    return g
def MyNormCCorrelation_201719942_201822262(image, kernel, boundary_condition="fill"): # función de la entrega pasada modificada con cambios en el método de frontera symm y la f+órmula de crosscorrelación de manera que el retorno sea una crosscorrelación normalizada
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
    I0 = np.mean(kernel) # se calculan la media y la desviación estandar del kernel para el uso en el cálculo de la crosscorrelación normalizada con ayuda de la librería numpy
    I0_desvest = np.std(kernel)
    if boundary_condition=="fill": # se realizan serie de condicionales los cuales se aplicarán según la condición de frontera que igresa por parámetro
        fill_image = np.pad(image.copy(), a, mode="constant") #copia de la imagen para generar bordes de 0s en la imagen con ayuda de librería numpy ccon modo constante
        normalized_ccorrelation = np.zeros((len(image)+a*2, len(image[0])+b*2)) # se crea matriz para almacenar cross-correlación con tamaño dependiente de a y b          #print(CCorrelation.shape)
        for filas in range(0+a,len(fill_image)-a): # recorrido para realizar la cross-correlación empezando sobre pixel central dado por a y b
            for columnas in range(0+b,len(fill_image[0])-b):
                mini_matriz = fill_image[filas-a:filas+a+1, columnas-b:columnas+b+1] # se extrae la sección de la imagen que se trabajará en esta parte del recorrido
                I1 = np.mean(mini_matriz) # se calculan la media y la desviación estandar de la sección de la imagen trabajada para el uso en el cálculo de la crosscorrelación normalizada con ayuda de la librería numpy
                I1_desvest = np.std(mini_matriz)
                i_fila=filas-a # contador de filas para tomar vecinos de pixel central
                factor_1_num = 0 # se inicializa variable para almacenar el factor del numerador para la crosscorrelación normalizada
                for multi_i in range(len(kernel)): # recorrido por el tamaño del que kernal para sacar prom del pixelcentral evaluda
                    j_column=columnas-b # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += (kernel[multi_i][multi_j] - I0) * (fill_image,[i_fila][j_column] - I1) # suma a la variable designada para almacenar la suma que sorrerponde al numerador
                        j_column+=1 #aumento contador columnas
                    i_fila+=1 # aumento contador filas
                if I1_desvest==0: # condicionales en el caso de que las desviaciones estandar sean 0 para evitar divisiones entre 0 y no salga como valor nan
                    I1_desvest=1
                if I0_desvest==0:
                    I0_desvest=1
                normalized_ccorrelation[filas][columnas] = (factor_1_num) / ( I1_desvest * I0_desvest)  # cálculo cross-correlación normalizada
    elif boundary_condition=="symm": # para condición de frontera symm # se realiza arreglo de la entrega pasada de esta condición de frontera
        imagen_marco = np.pad(image.copy(), a, mode="symmetric") # reflejo de bordes con ayuda de librería numpy pad y modo symmetric del tamaño que indique el kernel de entrada
        normalized_ccorrelation = np.zeros((len(image) , len(image[0]) ))  # se crea matriz para almacenar cross-correlación con tamaño dependiente de a y b          #print(CCorrelation.shape)
        fila_norma = 0 # indicador de fila de la imagen que almacenará la crosscorrelación normalizada
        for filas in range(0 + a, len(imagen_marco) - a-1):  # recorrido para realizar la cross-correlación empezando sobre pixel central dado por a y b
            columna_norma = 0 # indicador de columna de la imagen que almacenará la crosscorrelación normalizada
            for columnas in range(0 + b, len(imagen_marco[0]) - b-1):
                mini_matriz = imagen_marco[filas - a:filas + a + 1, columnas - b:columnas + b + 1] # se extrae la sección de la imagen que se trabajará en esta parte del recorrido
                I1 = np.mean(mini_matriz)# se calculan la media y la desviación estandar de la sección de la imagen trabajada para el uso en el cálculo de la crosscorrelación normalizada con ayuda de la librería numpy
                I1_desvest = np.std(mini_matriz)
                factor_1_num = 0# se inicializa variable para almacenar el factor del numerador para la crosscorrelación normalizada
                i_fila = filas - a  # contador de filas para tomar vecinos de pixel central
                for multi_i in range(len(kernel)):  # recorrido por el tamaño del que kernal para sacar prom del pixelcentral evaluda
                    j_column = columnas - b  # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += (kernel[multi_i][multi_j] - I0) * (imagen_marco[i_fila][j_column] - I1) # suma a la variable designada para almacenar la suma que sorrerponde al numerador
                        j_column += 1  # aumento contador columnas
                    i_fila += 1  # aumento contador filas
                if I1_desvest==0: # condicionales en el caso de que las desviaciones estandar sean 0 para evitar divisiones entre 0 y no salga como valor nan
                    I1_desvest=1
                if I0_desvest==0:
                    I0_desvest=1
                normalized_ccorrelation[fila_norma][columna_norma] = (factor_1_num) / ( I1_desvest * I0_desvest)  # cálculo cross-correlación normalizada
                columna_norma += 1 # aumento en indicador columnas
            fila_norma += 1 # aumento en indicador de filas
        normalized_ccorrelation= np.pad(normalized_ccorrelation, a, mode="symmetric") # se añaden bordes finales reflejados de la crosscorrelación con ayuda de la función de numpy utilizada para el marco inicial
    elif boundary_condition=="valid": # método de frontera valid
        normalized_ccorrelation=np.zeros((len(image)-a*2,len(image[0])-b*2)) # matriz para almacenar respuesta
        for filas in range(0+a,len(image)-a): # recorrido para cálculo crosscorrelación como en métodos anterioes
            for columnas in range(0+b,len(image[0])-b):
                mini_matriz = image[filas - a:filas + a + 1, columnas - b:columnas + b + 1]# se extrae la sección de la imagen que se trabajará en esta parte del recorrido
                I1 = np.mean(mini_matriz) # se calculan la media y la desviación estandar de la sección de la imagen trabajada para el uso en el cálculo de la crosscorrelación normalizada con ayuda de la librería numpy
                I1_desvest = np.std(mini_matriz)
                factor_1_num = 0 # se inicializa variable para almacenar el factor del numerador para la crosscorrelación normalizada
                i_fila=filas-a # contador de filas para tomar vecinos de pixel central
                for multi_i in range(len(kernel)):
                    j_column=columnas-b # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        factor_1_num += (kernel[multi_i][multi_j] - I0) * (image[i_fila][j_column] - I1) # suma a la variable designada para almacenar la suma que sorrerponde al numerador
                        j_column+=1
                    i_fila+=1
                if I1_desvest==0: # condicionales en el caso de que las desviaciones estandar sean 0 para evitar divisiones entre 0 y no salga como valor nan
                    I1_desvest=1
                if I0_desvest==0:
                    I0_desvest=1
                normalized_ccorrelation[filas - a][columnas - b] = (factor_1_num) / ( I1_desvest * I0_desvest)  # cálculo cross-correlación normalizada
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
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
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
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
# Filtro medio adaptativo
def MyAdaptMedian_201719942_201822262(gray_image, window_size, max_window_size):
    """
    función para cálculo con filtro medio adaptativo
    :param gray_image: imagen a filtrar en escala de grises
    :param window_size: tamaño inicial de la ventana
    :param max_window_size: tamaño máximo para la venta
    :return: imagen filtrada
    """
    imagen_marco=np.pad(gray_image.copy(),int(max_window_size)//2,mode="symmetric")# copia de la imagen con bordes de modo symm del tamaño máximo que podría llegar a tener la ventana, se utiliza la función de numpy pad
    filas,columnas=len(gray_image),len(gray_image[0]) #se definen filas y columnas con las dimensiones de la imagen
    filtered_image=np.zeros((filas, columnas)) #se crea un arreglo de ceros para almacenar la nueva imagen
    desfase=int(max_window_size)//2 # variable para determinar en donde está el primer pixel de la imagen
    for fila in range(filas):
        for columna in range(columnas):
            fila_pix_original,column_pix_original=fila+desfase,columna+desfase #variables para determinar fila y columna
            size_actual = window_size #variable con el tamaño de la imagen
            while size_actual <= max_window_size: #iteración hasta que la imagen exceda el tamaño maximo de ventana
                marco_actual = int(size_actual) // 2 # definición del tamaño del marco utilizando el tamaño actual
                mini_matriz = imagen_marco[fila_pix_original - marco_actual:   (fila_pix_original) + marco_actual + 1,column_pix_original - marco_actual:    column_pix_original + marco_actual + 1] # extracción de sección a trabajar de la imagen según el tmaño del filtro
                # se extrae una matriz pequeña utilizando la imagen con marco como referencia
                z_min = np.min(mini_matriz.flatten())  # variable para el mínino, se usa la funcion de numpy min
                z_max = np.max(mini_matriz.flatten())  # variable para el máximo, se usa la funcion de numpy max
                z_med = np.median(mini_matriz.flatten())  # variable para la mediana, se usa la función de numpy median
                A1 = z_med - z_min # variable para saber quien es mayor, la media o el minimo
                A2 = z_med - z_max # variable para saber quien es mayor, la media o el máximo
                if A1>0 and A2<0:
                    centro=imagen_marco[fila_pix_original,column_pix_original] # se ubica el centro de la imagen con marvo
                    B1=centro-z_min # comparacion del centro con el minimo
                    B2=centro-z_max # comparacion del centro con el máximo
                    if B1>0 and B2<0:
                        filtered_image[fila][columna] = centro # se llenan los valores para la imagen filtrada dependiendo de si es un impulso o no
                    else:
                        filtered_image[fila][columna] = z_med # se llenan los valores para la imagen filtrada dependiendo de si es un impulso o no
                    break
                else:
                    if size_actual==max_window_size: # si el tamaño de la ventana ya llegó al máximo
                        filtered_image[fila][columna]=z_med # se asigna valor mediano (no se asegura que no sea un impulso realmente)
                    size_actual+=1 # aumento en el tamaño de la ventana
    return filtered_image #se retorna la imagen
#Pruebe con 3 tamaños de ventana diferentes aplicar su función de filtro mediano adaptativo a las imágenes con ruido. Muestre sus experimentos en un subplot con títulos las imágenes con ruido y las respuestas después del filtrado.
plt.figure() # figura para probar distintos tamaños iniciales de ventana en cada subplot se muestra imagen filtrada con su respectivo título y remoción de ejes. Se visualiza con color map gray
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
#Pruebe filtrar ambas imágenes con distintos kernels Gaussianos, que varíen en tamaño y valor de σ
# se crean y evaluán distintos filtros de Gaauss para identificar cuál genera un mejor filtrado, se cambian tamaños y desviaciones estandar
filtro1_Gauss,filtro2_Gauss , filtro3_Gauss , filtro4_Gauss, filtro5_Gauss , filtro6_Gauss=gaussian_kernel(3,1) ,gaussian_kernel(3,50) ,  gaussian_kernel(3,100) , gaussian_kernel(3,5) , gaussian_kernel(5,5) , gaussian_kernel(7,5)# creación de filtros Gaussianos con tamaño constante y sigma variable  #se crean tres diferentes filtros de Gauss con función dada variando el tamaño y mateniendo el sigma constante
"""plt.figure("R1_G1")
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro1_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure("R2_G2}1")
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro1_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro2_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro2_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro3_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro3_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro4_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro4_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro5_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro5_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido1,filtro6_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")
plt.figure()
prueba_ka_s=MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro6_Gauss,boundary_condition="fill")
plt.imshow(prueba_ka_s, cmap="gray")"""
#Muestre en un subplot el resultado del mejor filtrado para cada imagen y diga con qué filtro y qué parámetros se produce la mejor imagen.
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure() # figura de imágens filtradas con el filtro que mejor resultados pressentó. Para cada una se genera su respectivo subplot con título y remoción de ejes, visualizando en colormap gray
plt.subplot(1,2,1)
plt.title("noisy1: Filtro mediano adaptativo tamaño ventana = 3")
plt.imshow(MyAdaptMedian_201719942_201822262(imag_ruido1,3,5), cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
#se comentan las siguientes líneas ya que el subplot se realiza con la función sin haber modificado la función de crosscorrelación para que ahora fuese normalizada, lara visualizar imagen ir a informe
"""plt.title("noisy2: Filtro Gauss 5x5 y σ = 5") 
plt.imshow(MyNormCCorrelation_201719942_201822262(imag_ruido2,filtro5_Gauss,boundary_condition="fill"), cmap="gray")
plt.axis("off")"""
plt.tight_layout
plt.show()
#PROBLEMA BIOMÉDICA
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
reference1=io.imread("reference1.jpg") # carga de las diferentes imágenes a trabajar en el problema biomédico
reference2=io.imread("reference2.jpg")
reference3=io.imread("reference3.jpeg")
parasitized=io.imread("Parasitized.png")
uninfected=io.imread("Uninfected.png")
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
#Selección del kernel
"""plt.figure()
plt.subplot(2,2,1)
plt.title("Imagen Parasitized.png")
plt.imshow(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),rgb2gray(parasitized[7:20,33:54]),boundary_condition="symm"), cmap="gray")
plt.axis("off")
plt.subplot(2,2,2)
plt.title("Imagen Parasitized.png")
plt.imshow(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),rgb2gray(parasitized[12:15,39:42]),boundary_condition="symm"), cmap="gray")
plt.axis("off")
plt.subplot(2,2,3)
plt.title("Imagen Parasitized.png")
plt.imshow(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),rgb2gray(parasitized[10:15,40:45]),boundary_condition="symm"), cmap="gray")
plt.imshow(match_template(rgb2gray(parasitized),rgb2gray(parasitized[10:15,40:45]),mode="symmetric"), cmap="gray")
plt.axis("off")
print(np.mean(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),rgb2gray(parasitized[10:15,40:45]),boundary_condition="symm")))
print(rgb2gray(parasitized[10:15,40:45]))
print(len(rgb2gray(parasitized[7:20,33:54])))
print(len(rgb2gray(parasitized[7:20,33:54])[0]))"""
# carga de imágenes de test
test1, test2, test3,test4,test5,test6,test7,test8,test9,test10=io.imread(os.path.join("test","malaria1.png")),io.imread(os.path.join("test","malaria2.png")),io.imread(os.path.join("test","malaria3.png")),io.imread(os.path.join("test","malaria4.png")),io.imread(os.path.join("test","malaria5.png")),io.imread(os.path.join("test","malaria6.png")),io.imread(os.path.join("test","malaria7.png")),io.imread(os.path.join("test","malaria8.png")),io.imread(os.path.join("test","malaria9.png")),io.imread(os.path.join("test","malaria10.png"))
def etiqueta(image,template): # función para asignar etiquetas por medio de prints dependiendo del porcentaje
    """
    asignación de etiquetas para las diferente imágenes
    :param image:  imagen a la que se aplicará cierto template
    :param template: que se utilizará para identificar objetos de interés
    :return: porcentaje
    """
    normalizada=MyNormCCorrelation_201719942_201822262(image,template,boundary_condition="symm") # filtrado por cross correlación normalizada
    suma=0 #inicialización de variable para la suma
    filas=len(normalizada) # filas de la imagen
    column=len(normalizada[0]) # columnas de la imagen
    for fila in range(filas): # recorrido por la matriz de la imagen
        for columna in range(column):
            suma+=(normalizada[fila][columna]) # suma del valor del pixel actual a la variable suma
    porcentaje=abs(suma/(filas*column)) # se halla porcentaje al valor absoluto de la suma
    if porcentaje>0.02: # si el porcentaje es mayor a 0.02 (2%)
        print("Uninfected:", porcentaje) # se considera a la imagen no infectada. Se muestra al usuario la etiqueta y el porcentaje que arrojó
    else: # de lo contrario
        print("Parasitized:", porcentaje) # se considera a la imagen infectada . Se muestra al usuario la etiqueta y el porcentaje que arrojó
    return porcentaje # retorno del porcentaje
# imágenes especificadas
especificado6,especificado7,especificado8,especificado9,especificado10=myImagePreprocessor(test6,reference2,action="save"),myImagePreprocessor(test7,reference2,action="save"),myImagePreprocessor(test8,reference2,action="save"),myImagePreprocessor(test9,reference2,action="save"),myImagePreprocessor(test10,reference2,action="save")
# almacenamiento de kernels seleccionados tanto de imagen original como preprocesada
testkernel_1=rgb2gray(parasitized)[12:15,39:42]
testkernel_2=myImagePreprocessor(rgb2gray(parasitized),rgb2gray(reference2),action="save")[12:15,39:42]
plt.figure()
plt.subplot(1,2,1)
plt.title("Imagen Parasitized.png con kernel 1")
plt.imshow(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),testkernel_1,boundary_condition="symm"), cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Imagen Parasitized.png con kernel 2")
plt.imshow(MyNormCCorrelation_201719942_201822262(rgb2gray(parasitized),testkernel_2,boundary_condition="symm"), cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
# se muestran valores y etiquetas para las diferentes imágenes
print("imag1-5 kernel 1 SIN especificado")
etiqueta(rgb2gray(test1),testkernel_1)
etiqueta(rgb2gray(test2),testkernel_1)
etiqueta(rgb2gray(test3),testkernel_1)
etiqueta(rgb2gray(test4),testkernel_1)
etiqueta(rgb2gray(test5),testkernel_1)
print("imag6-10 kernel 1 CON especificado")
etiqueta(rgb2gray(especificado6),testkernel_1)
etiqueta(rgb2gray(especificado7),testkernel_1)
etiqueta(rgb2gray(especificado8),testkernel_1)
etiqueta(rgb2gray(especificado9),testkernel_1)
etiqueta(rgb2gray(especificado10),testkernel_1)
print("imag1-5 kernel 2 SIN especificado")
etiqueta(rgb2gray(test1),testkernel_2)
etiqueta(rgb2gray(test2),testkernel_2)
etiqueta(rgb2gray(test3),testkernel_2)
etiqueta(rgb2gray(test4),testkernel_2)
etiqueta(rgb2gray(test5),testkernel_2)
print("imag6-10 kernel 2 CON especificado")
etiqueta(rgb2gray(especificado6),testkernel_2)
etiqueta(rgb2gray(especificado7),testkernel_2)
etiqueta(rgb2gray(especificado8),testkernel_2)
etiqueta(rgb2gray(especificado9),testkernel_2)
etiqueta(rgb2gray(especificado10),testkernel_2)
