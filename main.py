from Detector import OpenCVImpl

if __name__ == "__main__":
    opencv = OpenCVImpl()
    opencv.read_images_from_dir(None)
    if opencv.verify_load():
        opencv.load_parameters()
        opencv.aruco_detection()
    else:
        print("Error al cargar las imagenes")
    opencv.end_session()