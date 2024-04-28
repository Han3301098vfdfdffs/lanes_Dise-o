import numpy as np
import cv2
import pickle

def nothing(x): #No hacer nada, pasa por alto cualquier calculo u operación
    pass

def undistort (img, cal_dir='cal_pickle.p'):#Pasa o retorna la imagen definida en el argumento
    return img

"""def undistort(img, cal_dir='cal_pickle.p'):
     with open(cal_dir, mode='rb') as f:
         file = pickle.load(f)
     mtx = file['mtx']
     dist = file['dist']
     dst = cv2.undistort(img, mtx, dist, None, mtx)
     return dst"""

def colorFilter(img): #Filtrado de color 
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV) #Pasa de BGR a HSV separa información de tono, brillo e intensidad
    lowerYellow = np.array([18,94,140]) #Valores minimos HSV de amarillo
    upperYellow = np.array([48,255,255]) #Valores maximos HSV de amarrillo
    lowerWhite = np.array([0, 0, 200]) #Valores minimos HSV blanco
    upperWhite = np.array([255, 255, 255]) #Valores maximos HSV blanco
    maskedWhite= cv2.inRange(hsv,lowerWhite,upperWhite) #Máscara binaria identifica pixeles en el rango definido para blanco
    maskedYellow = cv2.inRange(hsv, lowerYellow, upperYellow) #Máscara binaria identifica pixeles en el rango definido para amarillo
    combinedImage = cv2.bitwise_or(maskedWhite,maskedYellow) #Combina las dos máscaras binarias
    return combinedImage

def thresholding(img): #Definición mejorada de bordes
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convierte a escales de grises 
    kernel = np.ones((5,5)) #kernel de convolución 5x5 
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0) #Filtro desenfoque Gaussiano, suavizar y reducir ruido
    imgCanny = cv2.Canny(imgBlur, 50, 100) #Detector de borden Canny 50 y 100 son los limites del umbral
    #imgClose = cv2.morphologyEx(imgCanny, cv2.MORPH_CLOSE, np.ones((10,10)))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1) #Dilatación en bordes (une regiones de borde en contornos más solidos)
    imgErode = cv2.erode(imgDial, kernel, iterations=1) #Erosión (eliminar detalles no desados en bordes)

    imgColor = colorFilter(img) #Se aplica el filtro anterios
    combinedImage = cv2.bitwise_or(imgColor, imgErode) #Se combina todo

    return combinedImage,imgCanny,imgColor

def initializeTrackbars(intialTracbarVals): #ventana Trackbars
    cv2.namedWindow("Trackbars") #ventana con barras deslizantes
    cv2.resizeWindow("Trackbars", 360, 240) #Se ajusta la ventana a 360x240 pixeles
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],50, nothing) #Barra deslizante variable Width Top de 0-50
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, nothing) #Barra deslizante variable Height Top de 0-100
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, nothing) #Barra deslizante variable Width Bottom de 0-50
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, nothing) #Barra deslizante variable Width Botton de 0-100

def valTrackbars(): #Se definen los valores iniciales de Trackbars
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                      (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    return src

def drawPoints(img,src): #Dibujar puntos para calibrar 
    img_size = np.float32([(img.shape[1],img.shape[0])]) #Se crea vector 
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    src = src * img_size #Escala las coordenadas por las dimensiones de la imagen
    for x in range( 0,4):
        cv2.circle(img,(int(src[x][0]),int(src[x][1])),15,(0,0,255),cv2.FILLED)#Se dibuja un circulo
    return img 

def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img) #Corregir la distorición de la imagen
    img = np.copy(img) #Se compia la imagen sin modificar la original
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float) #Convesrión de RGB a HLS 
    #hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float) #codigo para ver en cv2
    l_channel = hls[:, :, 1] #Separa canal iluminación
    s_channel = hls[:, :, 2] #Separa canal saturación
    h_channel = hls[:, :, 0] #Separa canal matriz
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  #Operador Sobel en iluminación toma la derivada en x
    abs_sobelx = np.absolute(sobelx)  # Derivada absoluta de x para acentuar las líneas alejadas de la horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx)) #Se normalizan los valores entre 0 y 255

    # Umbral x gradiente
    sxbinary = np.zeros_like(scaled_sobel) #Se crea imagen binaria aplicando umbral al gradiente en x
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1#Se resaltan los bordes verticales

    # Threshold color channel
    s_binary = np.zeros_like(s_channel) #Se crea imagen binaria con umbral de saturación
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1 #Resalta pixeles con una saturación dentro del rango

    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255 #Imagen a color combinando en los caneles verde y azul, rojo en ceros, multiplicados por 255

    combined_binary = np.zeros_like(sxbinary) #Se crea una imgen combinada de las binarias 
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])]) #Se calcula el tamaño de la imagen en float32
    src = src* img_size #Se escalan los puntos de origen multiplicandolos por el tamaño de la imagen
    # Para los puntos de destino, elijo arbitrariamente algunos puntos para ser
    # una buena opción para mostrar nuestro resultado distorsionado
    # nuevamente, no es exacto, pero lo suficientemente cercano para nuestros propósitos
    dst = dst * np.float32(dst_size) #
    # Dados los puntos src y dst, calcula la matriz de transformación de perspectiva
    M = cv2.getPerspectiveTransform(src, dst) #Utiliza puntos de destino para calcular matriz de transformación
    # Deformar la imagen usando OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size) #Se aplica transformación

    return warped

def inv_perspective_warp(img, #Su inversa
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0) #Calculo histograma
    return hist

left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):
    global left_a, left_b, left_c, right_a, right_b, right_c #Retoma las variables
    left_fit_ = np.empty(3) #Se inicializan para almacenar ajustes de lineas de carril
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255 #Imagen en blanco para ver el proceso

    histogram = get_hist(img) #Se calcula el histograma
    # encontrar picos de las mitades izquierda y derecha para ver lineas
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Establecer la altura de las ventanas deslizantes
    window_height = int(img.shape[0] / nwindows)
    window_height = np.int(img.shape[0] / nwindows)
    # Identificar las posiciones x & y de todos los píxeles distintos de cero en la imagen
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Posiciones actuales que se actualizarán para cada ventana
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Cree listas vacías para recibir índices de píxeles de los carriles izquierdo y derecho
    left_lane_inds = []
    right_lane_inds = []
    # Pasa por las ventanas una por una.
    for window in range(nwindows):
        # Identificar los límites de la ventana en x & y (y derecha e izquierda)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Dibujar las ventanas en la imagen de visualización.
        if draw_windows == True:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (100, 255, 255), 1)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (100, 255, 255), 1)
            # Identificar los píxeles distintos de cero en x & y dentro de la ventana
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Agregar estos índices a las listas.
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # Si encontró > píxeles minpix, vuelva a centrar la siguiente ventana en su posición media
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    #        if len(good_right_inds) > minpix:
    #            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))

    # Concatenar los arrays de índices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extraer las posiciones de los píxeles de las líneas izquierda y derecha
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if leftx.size and rightx.size:
        # Ajustar un polinomio de segundo orden a cada uno
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])

        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])

        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])

        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])

        # Generar valores x & y para trazar
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    else:
        return img,(0,0),(0,0),0
    
def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0]) #Crea un array de valores y equiespaciados desde 0 hasta el tamaño de la imagen en el eje y menos 1.
    y_eval = np.max(ploty) #Calcula el valor máximo de y en el array
    ym_per_pix = 1 / img.shape[0]  # Calcula la relación de metros por píxel en la dimensión y
    xm_per_pix = 0.1 / img.shape[0]  # Calcula la relación de metros por píxel en la dimensión x

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2) #Ajusta un polinomio de segundo grado en el espacio de mundo (metros) utilizando los puntos 
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)#Ajusta un polinomio de segundo grado para el carril derecho.
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])#Calcula el radio de curvatura del carril izquierdo en metros utilizando la fórmula de curvatura para un polinomio de segundo grado.
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])#Calcula el radio de curvatura del carril derecho de manera similar al anterior.

    car_pos = img.shape[1] / 2 #Calcula la posición horizontal del automóvil en la imagen.
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2] #Calcula la coordenada x en la parte inferior de la imagen para el carril izquierdo.
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2] #Calcula la coordenada x en la parte inferior de la imagen para el carril derecho.

    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2 #Calcula la posición del centro del carril tomando el promedio de las coordenadas x inferiores para el carril izquierdo y derecho.
    center = (car_pos - lane_center_position) * xm_per_pix / 10 #Calcula la desviación del centro del automóvil con respecto al centro del carril, convirtiendo de píxeles a metros y dividiendo por 10 para mayor claridad.
    # Now our radius of curvature is in meters

    return (l_fit_x_int, r_fit_x_int, center)

def draw_lanes(img, left_fit, right_fit,frameWidth,frameHeight,src):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])#Genera un array de puntos equidistantes en el eje y para representar los valores de y.
    color_img = np.zeros_like(img) #Crea una imagen de ceros del mismo tamaño y tipo que la imagen de entrada 

    left = np.array([np.transpose(np.vstack([left_fit, ploty]))]) #alcula las coordenadas (x, y) del carril izquierdo mediante la combinación de los valores de y generados por ploty con los valores x obtenidos del ajuste polinómico left_fit
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))]) #Similar al paso anterior, calcula las coordenadas (x, y) del carril derecho a partir del ajuste polinómico right_fit
    points = np.hstack((left, right)) #Combina las coordenadas de los carriles izquierdo y derecho en un único array para definir los puntos que formarán los polígonos de los carriles.

    cv2.fillPoly(color_img, np.int_(points), (0, 200, 255)) #Rellena el área entre los carriles izquierdo y derecho con un color específico en la imagen color_img. Esto crea una representación visual de los carriles en la imagen.
    inv_perspective = inv_perspective_warp(color_img,(frameWidth,frameHeight),dst=src)#Aplica una transformación de perspectiva inversa a la imagen coloreada de los carriles utilizando la función inv_perspective_warp. Esta transformación revierte la perspectiva aplicada anteriormente para mostrar los carriles en su perspectiva original.
    inv_perspective = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)#Combina la imagen original img con la imagen de los carriles transformados inv_perspective usando un peso específico para cada una. Esto se realiza para superponer los carriles dibujados en la imagen original.
    return inv_perspective

def textDisplay(curve,img):
    font = cv2.FONT_HERSHEY_SIMPLEX #Define el tipo de fuente para el texto que se va a mostrar en la imagen.
    cv2.putText(img, str(curve), ((img.shape[1]//2)-30, 40), font, 1, (255, 255, 0), 2, cv2.LINE_AA) #Agrega texto a la imagen img que representa el valor de curve
    directionText=' No lane '#Inicializa la variable directionText con el texto predeterminado ' No lane '.
    if curve > 10:
        directionText='Right'#Si es verdadero, establece directionText como 'Right' (Derecha).
    elif curve < -10:
        directionText='Left'#Si es verdadero, establece directionText como 'Left' (Izquierda).
    elif curve <10 and curve > -10:
        directionText='Straight'#Si es verdadero, establece directionText como 'Straight' (Recto).
    elif curve == -1000000:
        directionText = 'No Lane Found'#Establece directionText como 'No Lane Found' (No se encontró carril).
    cv2.putText(img, directionText, ((img.shape[1]//2)-35,(img.shape[0])-20 ), font, 1, (0, 200, 200), 2, cv2.LINE_AA)#Agrega el texto directionText a la imagen 
    #No es necesario un return, la imagen se modifica directamente

def textDisplay(curve,img):
    font = cv2.FONT_HERSHEY_SIMPLEX #Define el tipo de fuente para el texto que se va a mostrar en la imagen.
    cv2.putText(img, str(curve), ((img.shape[1]//2)-30, 40), font, 1, (255, 255, 0), 2, cv2.LINE_AA) #Agrega texto a la imagen img que representa el valor de curve
    directionText=' No lane '#Inicializa la variable directionText con el texto predeterminado ' No lane '.
    if curve > 10:
        directionText='Right'#Si es verdadero, establece directionText como 'Right' (Derecha).
    elif curve < -10:
        directionText='Left'#Si es verdadero, establece directionText como 'Left' (Izquierda).
    elif curve <10 and curve > -10:
        directionText='Straight'#Si es verdadero, establece directionText como 'Straight' (Recto).
    elif curve == -1000000:
        directionText = 'No Lane Found'#Establece directionText como 'No Lane Found' (No se encontró carril).
    cv2.putText(img, directionText, ((img.shape[1]//2)-35,(img.shape[0])-20 ), font, 1, (0, 200, 200), 2, cv2.LINE_AA)#Agrega el texto directionText a la imagen 
    #No es necesario un return, la imagen se modifica directamente

def stackImages(scale,imgArray):
    rows = len(imgArray) # Obtiene el número de filas en imgArray
    cols = len(imgArray[0]) # Obtiene el número de columnas en la primera fila de imgArray
    rowsAvailable = isinstance(imgArray[0], list) # Verifica si imgArray contiene sub-listas (filas disponibles)
    width = imgArray[0][0].shape[1] # Obtiene el ancho y alto de las imágenes en imgArray[0][0]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):  # Si hay filas disponibles (imgArray es una lista de listas)
            for y in range(0, cols):
                # Redimensiona las imágenes en imgArray a la escala dada
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]: 
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                # Convierte las imágenes en escala de grises a BGR si es necesario
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        # Crea una imagen en negro del tamaño de una imagen en imgArray[0][0]
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows # Crea una lista de imágenes en negro de tamaño de filas
        hor_con = [imageBlank]*rows
        # Combina horizontalmente las imágenes en cada fila de imgArray
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        # Combina verticalmente las imágenes horizontales en una sola imagen
        ver = np.vstack(hor)
    else:
        # Si imgArray no contiene filas (una lista plana de imágenes)
        for x in range(0, rows):
            # Redimensiona las imágenes a la escala dada
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            # Convierte las imágenes en escala de grises a BGR si es necesario
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        # Combina horizontalmente todas las imágenes en imgArray
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def drawLines(img,lane_curve):
    # Obtiene el ancho y alto de la imagen
    myWidth = img.shape[1]
    myHeight = img.shape[0]
    print(myWidth,myHeight) # Imprime el ancho y alto de la imagen

    # Dibuja líneas verticales para representar la curva del carril
    for x in range(-30, 30):
        # Define la longitud de cada segmento de línea en función del ancho de la imagen
        w = myWidth // 20
        # Dibuja la línea en la imagen
        cv2.line(img, (w * x + int(lane_curve // 100), myHeight - 30),
                 (w * x + int(lane_curve // 100), myHeight), (0, 0, 255), 2)
    # Dibuja una línea vertical adicional en el centro de la imagen
    cv2.line(img, (int(lane_curve // 100) + myWidth // 2, myHeight - 30),
             (int(lane_curve // 100) + myWidth // 2, myHeight), (0, 255, 0), 3)
    # Dibuja una línea horizontal en la parte inferior de la imagen
    cv2.line(img, (myWidth // 2, myHeight - 50), (myWidth // 2, myHeight), (0, 255, 255), 2)

    return img