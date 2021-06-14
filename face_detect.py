import cv2 #Importa la librería de OpenCV "Computer vision" para la detección de rostros en este caso
import sys #Importa sys, utilizado en linea 4

# Obteniendo los datos de entrada
imagePath = sys.argv[1] # Al ejecutarlo por la consola, el segundo argumento corresponde al directorio de la imagen que se utilizará (el primer argumento es el nombre del script)
cascPath = "haarcascade_frontalface_default.xml" # Directorio del archivo "cascada", es el clasificador encargado de la detección de rostros en este caso

# Seleciona el haar cascade a utilizar, en este caso el para detección de frontal de rostros
faceCascade = cv2.CascadeClassifier(cascPath)

# Leyendo la imagen y cambiando su color
image = cv2.imread(imagePath) #"imread" de matlab lee la imagen del archivo especificado
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #"cvtColor" cambia el espacio de color de una imagen 
                                            # En este caso desde BGR ("BlueGreenRed") a gris
                                        # Trabajar en escala de grises simplifica el escaneo de la imágen, contribuye a ahorrar recursos

# Detectando los rostros en las imágenes
faces = faceCascade.detectMultiScale( # Aplica sobre el haar seleccionado la función "detectMultiScale"
                                        # Detecta los objetos a buscar considerando ciertos parámetros como los que siguen:
    gray, # Llama a la variable que contiene la imagen procesada 
    scaleFactor=1.15, # Indica la escala de la imagen para su análisis, en este caso está redimensionando 
                # la imagen en un 10%, es decir analiza la imagen reduciendola hasta esa cantidad, crea una pirámide de imagenes
                # a partir de la imagen original hasta la escala seleccionada, 
                # en escalas más grandes (del orden de 1.5 (al 50%) aprox) posiblemente detecte menos rostros y en escalas pequeñas 
                # (del orden de 1.01 (al 1%) aprox) reduce la imagen hasta el 1%, mayor es la cantidad de imagenes que debe analizar 
                # y produce que se detecte una gran cantidad de falsos positivos
    minNeighbors=10, #minNeighbors define la cantidad de "cuadros verificadores" de rostros que debería tener cada
                    # rostro detectado, si se elige un número menor, es posible que de acuerdo al tamaño en pixeles del rostro
                    # detecte más de una vez el mismo rostro, pues considera que cada rostro que detecte el barrido es uno distinto
                    # y además detecte muchos falsos positivos, pues necesita realizar menos verificaciones para aceptar o no una
                    # parte de la imagen como un rostro
                    # en un rango mayor, por ejemplo 25, posiblemente no detecte algunos rostros 
                    # pues necesita que se verifice que los 25 barridos vecinos confirmen la presencia de ese mismo rostro
    minSize=(30, 30),
    maxSize=(300,300)
    #min y max size determinan el mínimo de pixeles que podría ser detectado para definir un rostro
    # menor a minSize se descartan, mayor a maxSize también, depende en gran medida el tipo de imagen 
    # que se está buscando analizar
)

print("Found {0} faces!".format(len(faces))) # Imprime la cantidad de rostros encontrados con la función anterior

# Dibuja los rectangulos sobre los rostros detectados
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image) #Abre la imagen con los resultados
cv2.waitKey(4000) # Mantiene abierta la imagen con el resultado la cantidad de 
                    # milisegundos que se le especifique, con 0 como argumento la imagen
                    # se mantendrá abierta hasta que sea cerrada por el usuario
                    # (no puede realizarse ninguna acción en la consola hasta que la imagen sea cerrada)


# Se ejecuta de la siguiente manera: "python face_detect.py nombre_imagen.jpg"