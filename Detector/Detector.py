from typing import Any,Callable, Dict, List, Optional, Protocol, Tuple, Union
import cv2
import numpy as np
import os 
from pathlib import Path

#Adquirimos las imagenes que queremos para los marcadores tambien es separado
# Diccionario con los IDs de los marcadores y las rutas de las imagenes asociadas, utilizando os, y pathlib para cargar las imagenes de manera dinamica
class OPENCV(Protocol):
    def __init__(self) -> None:
        ...
    def read_images_from_dir(self, directory:str)-> None:
        ...
    def verify_load(self)-> bool:
        ...
    def load_parameters(self) -> None:
        ...

class OpenCVImpl(OPENCV):
    """
    Implementación de la interfaz OPENCV

    Atributos:
    ----------
    `images_map`: Diccionario que mapea los IDs de los marcadores a las imágenes
    `default_dir`: Directorio por defecto para cargar las imágenes
    """
    def __init__(self) -> None:
        self.images_map:dict[int, Any] = {}
        self.default_dir:str = os.path.join(Path(__file__).parent, "Imagenes") # Directorio por defecto para las imagenes
        self.parameters:Any
        self.dict:Any
        self.camara:Any

    def read_images_from_dir(self, directory:Optional[str]) -> None:
        """
        Lee las imágenes desde el directorio especificado y las almacena en el diccionario images_map, mapeando los IDs de los marcadores a las imágenes correspondientes.
        Args:
        ----------
        `directory`: Ruta del directorio desde donde se cargarán las imágenes. Si es None, se utilizará el directorio por defecto.
        """
        if directory is None:
            directory = self.default_dir
        
        self.images_map:dict[int, Any] = {
            i: cv2.imread(os.path.join(directory, f"imagen{i+1}.jpg")) for i in range(len(os.listdir(directory)))
            }
    
    def verify_load(self) -> bool:
        """
        Verifica que las imágenes se hayan cargado correctamente en el diccionario images_map.
        Retorna:
        ----------
        `bool`: True si todas las imágenes se cargaron correctamente, False en caso contrario.
        """
        for img_id, img_data in self.images_map.items():
            if img_data is None:
                print(f"Error: No se puede cargar la imagen para la id {img_id}")
                return False
        return True

    def load_parameters(self) -> None:
        self.parameters = cv2.aruco.DetectorParameters()
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.camara = cv2.VideoCapture(0)

    def aruco_detection(self) -> None:
        while True:
            ret, frame = self.camara.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #Detectamos los marcadores en las imagenes
            detector = cv2.aruco.ArucoDetector(self.dict, self.parameters)
            esquinas, ids, candidatos_malos = detector.detectMarkers(gray)

            if ids is not None:

                aruco = cv2.aruco.drawDetectedMarkers(frame, esquinas)

                for i, marker_id in enumerate(ids.flatten()):
                #print("ID marker:", marker_id)
                    if marker_id not in self.images_map:
                        print("No existen las imagenes")
                        continue #Si no hay imagen asociada saltamos

                    imagen = self.images_map[marker_id]

                    #Extraemos los puntos de las esquinas en coordenadas 
                    c1 = (esquinas[i][0][0][0], esquinas[i][0][0][1])
                    c2 = (esquinas[i][0][1][0], esquinas[i][0][1][1])
                    c3 = (esquinas[i][0][2][0], esquinas[i][0][2][1])
                    c4 = (esquinas[i][0][3][0], esquinas[i][0][3][1])

                    #Extraemos el tamano de la imagen 
                    if self.images_map is not None:

                        #Nuevo
                        copy = frame.copy()

                        Tamano = imagen.shape

                        #Organizamos las coordenadas del aruco en una matriz
                        puntos_aruco = np.array([c1, c2, c3, c4], dtype=float)
                        #Organizamos las coordenadas de la imagen en otra matriz
                        puntos_imagen = np.array([
                        [0, 0],
                        [Tamano[1]-1, 0],
                        [Tamano[1]-1, Tamano[0]-1],
                        [0, Tamano[0]-1]
                        ], dtype=float)

                        #Realizamos la superposicion de la imgen (NAnografica)
                        h, estado = cv2.findHomography(puntos_imagen, puntos_aruco)

                        #Posiblemente lo remueva
                        if h is None:
                            continue

                        #Realizamos las transformaciones de perspectiva
                        perspectiva = cv2.warpPerspective(imagen, h, (copy.shape[1], copy.shape[0]))
                        mask = np.zeros((copy.shape[0], copy.shape[1]), dtype=np.uint8)
                        cv2.fillConvexPoly(mask, puntos_aruco.astype(int), 255)
                        mask_inv = cv2.bitwise_not(mask)
                        fondo = cv2.bitwise_and(copy, copy, mask=mask_inv)
                        frente = cv2.bitwise_and(perspectiva, perspectiva, mask=mask)

                        copy = fondo + frente
                        cv2.imshow("Realidad virtual", copy)
                        k = cv2.waitKey(1)
                    else:
                        print("No tenemos imagen")
                        k = cv2.waitKey(1)
            else:
                cv2.imshow("Realidad virtual", frame)
                k = cv2.waitKey(1)

                if k == 27:
                    print("Adios")
                    break   

    def end_session(self) -> None:
        self.camara.release()
        cv2.destroyAllWindows()

                