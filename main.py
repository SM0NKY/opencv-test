from typing import Optional
from Detector import OpenCVImpl

def main(video_input:Optional[str] = None, input_dir:Optional[str] = None) -> None:
    opencv = OpenCVImpl()
    opencv.read_images_from_dir(input_dir)
    if opencv.verify_load():
        opencv.load_parameters(video_input)
        opencv.aruco_detection()
    else:
        print("Error al cargar las imagenes")
    opencv.end_session()

if __name__ == "__main__":
    main()
        