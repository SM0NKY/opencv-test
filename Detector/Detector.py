from typing import Any,Callable, Dict, List, Optional, Protocol, Tuple
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
    def load_parameters(self, camera_input:Optional[str]) -> None:
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

    def load_parameters(self, camera_input:Optional[str] = None) -> None:
        """
        Carga los parámetros necesarios para la detección de marcadores ArUco.
        
        Retorna:
        ----------
        `None`: No retorna ningún valor, pero inicializa los atributos necesarios para la detección de marcadores ArUco.

        Parameters:
        ----------
        - `camera_input`: Ruta de la cámara de video. Si es None, se utilizará la cámara por defecto.

        """

        self.parameters = cv2.aruco.DetectorParameters()
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        if camera_input is None:
            self.camara = cv2.VideoCapture(0)
        else:
            self.camara = cv2.VideoCapture(camera_input)
        if not self.camara.isOpened():
            raise RuntimeError("Error al inicializar la cámara")

    def aruco_detection(self) -> None:
        """
        Realiza la detección de marcadores ArUco en tiempo real utilizando la cámara. Superpone las imágenes correspondientes a los marcadores detectados en la transmisión de video.
        """
        if self.camara is None:
            raise RuntimeError("La cámara no ha sido inicializada. Llama a load_parameters() antes de aruco_detection().")
        detector = cv2.aruco.ArucoDetector(self.dict, self.parameters)

        while True:
            ___, frame = self.camara.read()
            
            if not ___ or frame is None:
                print("Error al capturar el video")
                break

            output = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            #Detectamos los marcadores en las imagenes
            esquinas, ids, _ = detector.detectMarkers(gray) #Donde _ son los candidatos malos

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(output,esquinas,ids)

                for i, marker_id in enumerate(ids.flatten()):
                #print("ID marker:", marker_id)
                    if int(marker_id) not in self.images_map:
                        print("No existen las imagenes")
                        continue #Si no hay imagen asociada saltamos
                        
                    imagen = self.images_map[int(marker_id)]

                    if imagen is None:
                        print(f"Error: No se puede cargar la imagen para la id {marker_id}")
                        continue
                    
                    #Extraccion de los puntos de las esquinas en coordenadas 
                    pts_aruco = esquinas[i].reshape((4, 2)).astype(np.float32)
                    alto, ancho = imagen.shape[:2]
                    
                    pts_imagen:np.array = np.array([
                        [0, 0],
                        [ancho - 1, 0],
                        [ancho - 1, alto - 1],
                        [0, alto - 1]
                    ], dtype=np.float32)
                    #Extraemos el tamaño de la imagen 
                    
                        #Realizamos la superposicion de la imgen (NAnografica)
                    h, _ = cv2.findHomography(pts_imagen, pts_aruco)

                        #Posiblemente lo remueva
                    if h is None:
                            continue

                        #Realizamos las transformaciones de perspectiva
                    perspectiva = cv2.warpPerspective(
                        imagen,
                        h,
                        (output.shape[1], output.shape[0])
                    )
                    
                    mask = np.zeros((output.shape[0], output.shape[1]), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, pts_aruco.astype(int), 255)
                    
                    mask_inv = cv2.bitwise_not(mask)
                    
                    fondo = cv2.bitwise_and(output, output, mask=mask_inv)
                    frente = cv2.bitwise_and(perspectiva, perspectiva, mask=mask)

                    output = cv2.add(fondo, frente)
                
            cv2.imshow("Realidad virtual", output)
            k = cv2.waitKey(1) & 0xFF
                
                
            if k == 27:
                print("Adios")
                break   

    def end_session(self) -> None:
        if self.camara is not None:
            self.camara.release()
        cv2.destroyAllWindows()

