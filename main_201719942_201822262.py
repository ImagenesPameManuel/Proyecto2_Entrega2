#Pamela Ramírez González #Código: 201822262
#Manuel Gallegos Bustamante #Código: 201719942
#Análisis y procesamiento de imágenes: Proyecto2 Entrega1
#Se importan librerías que se utilizarán para el desarrollo del laboratorio
import numpy as np
import skimage.io as io
import skimage.exposure as expo
from scipy.signal import correlate2d
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#se crean kernels propuestos en la guía con arreglos de numpy
kernel_a=np.array([[1,1,1],[1,1,1],[1,1,1]])
kernel_b=(1/9)*kernel_a
kernel_c=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_d=np.array([[1,2,1],[0,0,0],[-1,2,-1]])
#con la función indicada en la guía se crea filtro gaussiano
def gaussian_kernel(size, sigma):
    size = int(size)//2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/(2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2))) * normal
    return g
def MyCCorrelation_201719942_201822262(image, kernel, boundary_condition="fill"):
    """
    Función para la cross-correlación de una imagen y un kenerl dados. Se aplica la condición de frontera  deseada por el usuario
    :param image: arreglo de la imagen
    :param kernel: arreglo del filtro con el cual se realizará la cross-correlación
    :param boundary_condition: str "fill", "valid" o "symm" que determinará la condición de frontera que se le aplicará en la cross-correlación
    :return: arreglo de la imagen una vez realizada la cross-correlación
    """
    CCorrelation=0 # se inicializa variable respuesta de crosscorrelación que se modificará según el método de frontera
    a=round((len(kernel)-1)/2) # cálculo de a y b
    b=round((len(kernel[0])-1)/2)
    if boundary_condition=="fill": # se realizan serie de condicionales los cuales se aplicarán según la condición de frontera que igresa por parámetro
        fill_image=image.copy() #copia de la imagen para generar bordes en la imagen
        for i in range(a): # recorrido para generar bordes de 0s en la imagen
            fill_image=np.insert(fill_image, 0, 0, axis=1) #se inserta marco de 0s en la imagen fiil_image con uso de np.insert indicando como parámetros el arreglo al cual se le insertarán los elementos indicados como 3er parámetro en la posición indicada en el 2do parámetro del eje indicado como 4to parámetro columnas (1) o filas (0)
            fill_image=np.insert(fill_image, 0, 0, axis=0)
            fill_image=np.insert(fill_image, fill_image.shape[0], 0, axis=0)
            fill_image=np.insert(fill_image, fill_image.shape[1], 0, axis=1)
        CCorrelation = np.zeros((len(image)+a*2, len(image[0])+b*2)) # se crea matriz para almacenar cross-correlación con tamaño dependiente de a y b          #print(CCorrelation.shape)
        for filas in range(0+a,len(fill_image)-a): # recorrido para realizar la cross-correlación empezando sobre pixel central dado por a y b
            for columnas in range(0+b,len(fill_image[0])-b):
                i_fila=filas-a # contador de filas para tomar vecinos de pixel central
                for multi_i in range(len(kernel)): # recorrido por el tamaño del que kernal para sacar prom del pixelcentral evaluda
                    j_column=columnas-b # contador pata columans para tomar vecinos del pixel central
                    for multi_j in range(len(kernel[0])):
                        CCorrelation[filas][columnas]+=fill_image[i_fila][j_column]*kernel[multi_i][multi_j] # cálculo cross-correlación
                        j_column+=1 #aumento contador columnas
                    i_fila+=1 # aumento contador filas
    elif boundary_condition=="symm": # para condición de frontera symm
        copia = image.copy() # copia para realizar reflejo de la imagen
        primera_fila = copia[0] # primera fila
        # - encuentro las primeras filas
        primeras_filas = copia[0, :]
        # - agrego primer y último valor a la fila
        primera_fila = np.insert(primera_fila, 0, np.array(copia[0][0])*a)
        # primera_fila = np.insert(primera_fila, len(primera_fila), primera_fila[len(primera_fila) - 1])
        primera_fila = np.append(primera_fila, np.array(primera_fila[len(primera_fila) - 1])*a)
        filas_secundarias = np.array([[]])
        filas_secundarias = np.insert(filas_secundarias, 0, primera_fila)
        # guardo los valores de las filas secundarias
        if (a > 1):
            for i in range(0, len(a)):
                # almacena la segunda fila añadiendo al inicio y al final el valor de la esquina correspondiente
                segundas_filas = copia[i + 1, :]
                segundas_filas = np.insert(segundas_filas, 0, np.array(copia[0][0])*a)  # con elemento al principio
                segundas_filas = np.append(primera_fila, np.array(copia[0][len(copia[0]) - 1])*a) #con elemento final
                filas_secundarias = np.append(filas_secundarias, segundas_filas)
        # -encuentro las ultimas filas
        ultima_fila = copia[len(copia) - 1, :] # se extrae ´última fila y se insertan valores extremos para
        ultima_fila = np.insert(ultima_fila, 0, np.array(copia[len(copia)-1][0])*a)
        ultima_fila = np.append(ultima_fila, np.array(copia[len(copia)-1][len(copia[0])-1])*a)
        filas_secundarias2 = np.array([[]])
        filas_secundarias2 = np.insert(filas_secundarias2, 0, ultima_fila)
        if (a > 1): # PARA KERNELS CON a>1
            for i in range(0, len(a)):
                segundas_filas = copia[len(copia) - i - 2, :] # se añaden segundas filas incluyendo en sus extremos el valor del pixel de la esquina correspondiente
                segundas_filas = np.insert(segundas_filas, 0, np.array(copia[len(copia)-1][0])*a)
                segundas_filas = np.insert(segundas_filas, len(segundas_filas), np.array(copia[len(copia)-1][len(copia[0]) - 1])*a)
                filas_secundarias2 = np.append(filas_secundarias2, segundas_filas)
        # -encuentro las primeras columnas
        primeras_columnas = copia[:, 0]
        columnas_secundarias = np.array([[]])
        columnas_secundarias = np.append(columnas_secundarias,primeras_columnas)
        if(a > 1):
            for i in range(0, len(a)):
                segundas_columnas = copia[:, i+1] # se agregan columans en una misma matriz
                columnas_secundarias = np.append(columnas_secundarias, segundas_columnas)
        # -encuentro las ultimas columnas
        ultimas_columnas = copia[:, len(copia[0])-1]
        columnas_secundarias2 = np.array([[]])
        columnas_secundarias2 = np.append(columnas_secundarias2, ultimas_columnas)
        if(a>1):
            for i in range(0, len(a)):
                segundas_columnas = copia[:, len(copia[0])-2-i] # se agregan columnas en una única matriz
                columnas_secundarias2 = np.append(columnas_secundarias2, segundas_columnas)
        # -se agrega el primer valor y el último valor al principio y final de la primera fila
        # - se agregan las columnas para la respuesta
        for i in range(0, a):
            if a > 1:
                copia = np.insert(copia, 0, columnas_secundarias[i], axis=1)
                copia = np.insert(copia, len(copia[0]), columnas_secundarias2[i], axis=1)
            else:
                copia = np.insert(copia, 0, columnas_secundarias, axis=1)
                copia = np.insert(copia, len(copia[0]), columnas_secundarias2, axis=1)
        for i in range(0, a): #se agregan filas
            if a > 1:
                copia = np.insert(copia, 0, filas_secundarias[i], axis=0)
                copia = np.insert(copia, len(copia), filas_secundarias2[i], axis=0)
            else:
                copia = np.insert(copia, 0, filas_secundarias, axis=0)
                copia = np.insert(copia, len(copia), filas_secundarias2, axis=0)
        CCorrelation_ceros = np.zeros((len(image) , len(image[0]))) # matriz para cross-correlación parcial sin bordes
        for filas in range(0+a, len(copia) - a):  # recorrido para realizar cross-correlación de forma similar a método fill
            for columnas in range(0+b, len(copia[0]) - b):
                i_fila = filas - a
                for multi_i in range(len(kernel)):
                    j_column = columnas - b
                    for multi_j in range(len(kernel[0])):
                        CCorrelation_ceros[filas-a][columnas-b] += copia[i_fila][j_column] * kernel[multi_i][multi_j] # cálculo suma para crosscorrelación
                        j_column += 1
                    i_fila += 1
        CCorrelation=CCorrelation_ceros.copy() # respuesta final en la cual tendrá bordes reflejjados de
        for reflejo in range(a): # recorrido para reflego de columnas
            CCorrelation=np.insert(CCorrelation,0,CCorrelation_ceros[:, reflejo],axis=1)
            CCorrelation = np.insert(CCorrelation, len(CCorrelation[0]), CCorrelation_ceros[:, len(CCorrelation_ceros[0])-1-reflejo], axis=1)
        for reflejofilas in range(a): # recorrido para reflejo filas
            fila_aux= np.append(CCorrelation_ceros[reflejofilas, :],np.array(CCorrelation_ceros[0][len(CCorrelation_ceros[0])-1])*a) # a cada fila se le agregan en sus extremos los valores de las esquinas correspondientes
            fila_aux=np.insert(fila_aux, 0, np.array(CCorrelation_ceros[0][0])*a)
            fila_aux_final=np.append(CCorrelation_ceros[len(CCorrelation_ceros)-1-reflejofilas, :],np.array(CCorrelation_ceros[0][len(CCorrelation_ceros)-1])*a)
            fila_aux_final=np.insert(fila_aux_final, 0, np.array(CCorrelation_ceros[len(CCorrelation_ceros)-1][0])*a)
            CCorrelation = np.insert(CCorrelation, 0,fila_aux, axis=0)
            CCorrelation = np.insert(CCorrelation, len(CCorrelation_ceros), fila_aux_final,axis=0)
    elif boundary_condition=="valid": # método de frontera valid
        CCorrelation=np.zeros((len(image)-a*2,len(image[0])-b*2)) # matriz para almacenar respuesta
        for filas in range(0+a,len(image)-a): # recorrido para cálculo crosscorrelación como en métodos anterioes
            for columnas in range(0+b,len(image[0])-b):
                i_fila=filas-a
                for multi_i in range(len(kernel)):
                    j_column=columnas-b
                    for multi_j in range(len(kernel[0])):
                        CCorrelation[filas-a][columnas-b]+=image[i_fila][j_column]*kernel[multi_i][multi_j] # cálculo cross-correlación
                        j_column+=1
                    i_fila+=1
    return CCorrelation
rosas=io.imread("roses.jpg")
rosas_noise=io.imread("noisy_roses.jpg")
rosas=rgb2gray(rosas) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
rosas_noise=rgb2gray(rosas_noise)
prueba_ka_s=MyCCorrelation_201719942_201822262(rosas,kernel_a,boundary_condition="symm")
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
#carga de imágenes con io.imread
rosas=io.imread("roses.jpg")
rosas_noise=io.imread("noisy_roses.jpg")
rosas=rgb2gray(rosas) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro
rosas_noise=rgb2gray(rosas_noise) #se le quita 3D a la imagen para convertirla en una imagen blanco-negro          #print(rosas.shape) #print(kernel_a.shape) print(len(rosas))
#Comparaciones de resultados función creada con función propia de scipy.signal: correlate2d
prueba_ka=MyCCorrelation_201719942_201822262(rosas,kernel_a)
prueba_scipy=correlate2d(rosas,kernel_a,boundary="fill")
prueba_ka_v=MyCCorrelation_201719942_201822262(rosas,kernel_a,boundary_condition="valid")
prueba_scipy_v=correlate2d(rosas,kernel_a,mode="valid")
prueba_ka_s=MyCCorrelation_201719942_201822262(rosas,kernel_a,boundary_condition="symm")
prueba_scipy_s=correlate2d(rosas,kernel_a,boundary="symm")
error_ka_s=error_cuadrado(prueba_scipy_s, prueba_ka_s)
error_ka=error_cuadrado(prueba_scipy,prueba_ka)
error_ka_v=error_cuadrado(prueba_scipy_v,prueba_ka_v)
print("error kernel a fill")
print(error_ka)
print("error kernel a symm")
print(error_ka_s)
print("error kernel a valid ")
print(error_ka_v)
#5.1.1. Función MyCCorrelation 2.1
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
plt.figure("original_funcionpython") #figura para mostrar imagen original, imagen con cross-correlación con scipy y con función creada
plt.subplot(1,3,1) # título y remoción de ejes para las diretentes imágenes visualizadas con mapa de color de grises
plt.title("Imagen original escala grises")
plt.imshow(rosas,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen correlate2d con kernel a y fill")
plt.imshow(prueba_scipy,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen MyCCorrelation con kernel a y fill")
plt.imshow(prueba_ka,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
error_ka=error_cuadrado(prueba_scipy,prueba_ka) # cálculo error cuadrático medio
#print(error_ka)
#5.1.2. Aplicaciones de Cross-Correlación 1.
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
noise_kernel_a=MyCCorrelation_201719942_201822262(rosas_noise,kernel_a) # cálculo cross-correlación de imagen con ruido con kernels a y b con función creada previamente
noise_kernel_b=MyCCorrelation_201719942_201822262(rosas_noise,kernel_b)
plt.figure("original_kernel_ab") # figura para visualizar imagen original y filtrada con kernels a y b con opciónn de frontera fill
plt.subplot(1,3,1) # subplotspara cada una de las imágenes descritas previamente con su título y remoción de ejes visualizadas con mapa de color de grises
plt.title("Imagen original escala grises")
plt.imshow(rosas_noise,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con kernel a")
plt.imshow(noise_kernel_a,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con kernel b")
plt.imshow(noise_kernel_b,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#5.1.2. Aplicaciones de Cross-Correlación 3.
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
filtro_Gauss_punto3=gaussian_kernel(5,1) # se crea filtro de Gauss con función proporcionada en el enunciado. Filtro de 5x5b con sigma de 1
cross_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro_Gauss_punto3) # cross-correlación con condición de frontera fill de imagen con ruido y filtro de Gauss creado previamente
plt.figure("Original_kernel_b_Gauss") # figura para mostrar imagen original e imagen filtrada con filtro de gauss decrito previamente  y kernel b
plt.subplot(1,3,1) # cada imagen tiene su respectivo subplot, remoción de ejes, título y es visualizado con mapa de color de rises
plt.title("Imagen original escala grises")
plt.imshow(rosas_noise,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con kernel b")
plt.imshow(noise_kernel_b,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n5x5 y σ = 1")
plt.imshow(cross_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#5.1.2. Aplicaciones de Cross-Correlación 5.
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
filtro1_Gauss_punto5=gaussian_kernel(3,1) # creación de filtros Gaussianos con tamaño constante y sigma variable
filtro2_Gauss_punto5=gaussian_kernel(3,50)
filtro3_Gauss_punto5=gaussian_kernel(3,100)
cross1P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro1_Gauss_punto5) # cálculo cross-correlación de imagen con ruido con los diferentes filtros de Gauss creados previamente
cross2P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro2_Gauss_punto5)
cross3P5_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro3_Gauss_punto5)
plt.figure("Gausstamanofijosigmavariable") #fgura para mostrar el efecto de losdistintos filtros de Gauss creados previamentes sobre la imagen con ruido
plt.subplot(1,3,1) # subplot para cada una de las imágenes filtradas cada uno tiene su respectivo título, remoción de ejes y es visualizada con color map gris
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 1")
plt.imshow(cross1P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 50")
plt.imshow(cross2P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 100")
plt.imshow(cross3P5_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#5.1.2. Aplicaciones de Cross-Correlación 7.  VA A BOTAR ERROR función no sirve para filtros con tamaño diferente 3x3
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
filtro1_Gauss_punto7=gaussian_kernel(3,5) #se crean tres diferentes filtros de Gauss con función dada variando el tamaño y mateniendo el sigma constante
filtro2_Gauss_punto7=gaussian_kernel(5,5)
filtro3_Gauss_punto7=gaussian_kernel(7,5)
cross1P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro1_Gauss_punto7) # Cross-correlación para los diferentes filtros aplicados a la imagen con ruidp
cross2P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro2_Gauss_punto7)
cross3P7_filtroGauss=MyCCorrelation_201719942_201822262(rosas_noise,filtro3_Gauss_punto7)
plt.figure("Gausstamanovariablesigmafijo") # se crea figura para mostrar el efecto de variar el tamaño del filtro de Gauss
plt.subplot(1,3,1) # subplot para mostrar cada una de las cross-correlaciones con los filtros de Gauss creados con anterioridad. Se inserta título, se remueven ejes y se visualiza con mapa de colores gris
plt.title("Imagen con filtro Gauss:\n3x3 y σ = 5")
plt.imshow(cross1P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,2)
plt.title("Imagen con filtro Gauss:\n5x5 y σ = 5")
plt.imshow(cross2P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.subplot(1,3,3)
plt.title("Imagen con filtro Gauss:\n7x7 y σ = 5")
plt.imshow(cross3P7_filtroGauss,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#5.1.2. Aplicaciones de Cross-Correlación 8.
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
prueba_kc_v=MyCCorrelation_201719942_201822262(rosas_noise,kernel_c,boundary_condition="valid") # cross-correlación con kernels c y d con imagen co ruido y condición de frontera valid
prueba_kd_v=MyCCorrelation_201719942_201822262(rosas_noise,kernel_d,boundary_condition="valid")
plt.figure("kernel_c_y_kernel_d") # figura ara mostrar cross-correlación de imagen con ruido con kernels c y d
plt.subplot(1,2,1) # subplot para cada imagen con su respectivo título, remoción de ejes y visualización con mapa de color gris
plt.title("Imagen con kernel c")
plt.imshow(prueba_kc_v,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Imagen con kernel d")
plt.imshow(prueba_kd_v,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#BONO
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
prueba_kc_BONO=np.absolute(MyCCorrelation_201719942_201822262(rosas,kernel_c,boundary_condition="valid"))
prueba_kd_BONO=np.absolute(MyCCorrelation_201719942_201822262(rosas,kernel_d,boundary_condition="valid"))
plt.figure("BONO")
plt.subplot(1,2,1)
plt.title("Valor absoluto de la\nimagen con kernel c")
plt.imshow(prueba_kc_BONO,cmap="gray")
plt.axis("off")
plt.subplot(1,2,2)
plt.title("Imagen con kernel d")
plt.imshow(prueba_kd_BONO,cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.show()
#PROBLEMA BIOMÉDICA
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
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
"""plt.subplot(2,4,4)
plt.title("Imagen Parasitized.png")
plt.imshow(parasitized)
plt.axis("off")
plt.subplot(2,4,8)
plt.title("Histograma Parasitized.png")
plt.hist(parasitized.flatten())"""
plt.tight_layout()
plt.show()
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
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
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
ref2=myImagePreprocessor(parasitized,reference2,action="save")
ref2_show=myImagePreprocessor(parasitized,reference2,action="show")
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
ref3=myImagePreprocessor(parasitized,reference3,action="save")
ref3_show=myImagePreprocessor(parasitized,reference3,action="show")
input("Press Enter to continue...") # input para continuar con el programa cuando usuario presione Enter cuando desee
mejor_especif=myImagePreprocessor(parasitized,reference2,action="save")
mejor_especif_show=myImagePreprocessor(uninfected,reference2,action="show")