from typing import Any, List, Callable
import cv2
import insightface
import threading

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.typing import Face, Frame
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    """Carga el modelo de intercambio de rostros si aún no está inicializado."""
    global FACE_SWAPPER
    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_path = resolve_relative_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_path, providers=roop.globals.execution_providers)
    return FACE_SWAPPER


def clear_face_swapper() -> None:
    """Limpia la caché del modelo de face swapper."""
    global FACE_SWAPPER
    FACE_SWAPPER = None


def pre_check() -> bool:
    """Verifica si el modelo está disponible y lo descarga si es necesario."""
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    """Verifica que los archivos de entrada sean válidos."""
    if not is_image(roop.globals.source_path):
        update_status('❌ ERROR: Debes seleccionar una imagen para el source path.', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.source_path)):
        update_status('⚠️ Advertencia: No se detectó un rostro en la imagen de origen.', NAME)
        return False
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('❌ ERROR: Debes seleccionar una imagen o video para el target path.', NAME)
        return False
    return True


def post_process() -> None:
    """Limpia referencias y modelos después del procesamiento."""
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    """Realiza el intercambio de rostros asegurando que las caras sean válidas."""
    if source_face is None:
        print("❌ ERROR: No se detectó un rostro en la imagen de origen.")
        return temp_frame  # Devuelve el frame original sin modificaciones
    
    if target_face is None:
        print("⚠️ Advertencia: No se detectó una cara en el frame objetivo.")
        return temp_frame  # Devuelve el frame original sin modificaciones

    try:
        return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)
    except Exception as e:
        print(f"❌ ERROR en swap_face: {e}")
        return temp_frame  # Evita que el proceso falle


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    """Procesa un frame aplicando face swap si se detectan caras."""
    if source_face is None:
        print("❌ ERROR: No se detectó un rostro válido en la imagen de origen.")
        return temp_frame

    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)
    
    return temp_frame


def process_frames(source_path: str, temp_frame_paths: List[str], update: Callable[[], None]) -> None:
    """Procesa múltiples frames asegurando que source_face sea válido."""
    source_face = get_one_face(cv2.imread(source_path))
    if source_face is None:
        print(f"❌ ERROR: No se detectó un rostro en {source_path}. Omitiendo procesamiento.")
        return
    
    reference_face = None if roop.globals.many_faces else get_face_reference()
    
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if update:
            update()


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    """Procesa una imagen estática aplicando face swap."""
    source_face = get_one_face(cv2.imread(source_path))
    if source_face is None:
        print(f"❌ ERROR: No se detectó un rostro en {source_path}.")
        return False

    target_frame = cv2.imread(target_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)

'''

def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    """Procesa un video aplicando face swap en cada frame."""
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(temp_frame_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)

    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
'''
# Variable global para el estado de detección de caras
faces_detected_globally = None  # None = Imagen sin cara, False = Video sin cara, True = Todo OK

def process_video(source_path: str, temp_frame_paths: List[str]) -> bool:
    """Procesa un video aplicando face swap en cada frame y actualiza una variable global con el estado."""

    # Verificar si hay una cara en la imagen fuente
    source_face = get_one_face(cv2.imread(source_path))
    if source_face is None:
        print(f"❌ ERROR: No se detectó un rostro en la imagen fuente {source_path}.")

        return None  # Devuelve False para que no se procese el video

    # Contador para verificar si se detectan caras en el video
    faces_detected = False

    def update_detection():
        """Se llama cuando se detecta una cara en el video."""
        nonlocal faces_detected
        faces_detected = True

    # Procesar los frames del video
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

    # Si no se detectó ninguna cara en el video, actualizar la variable global y devolver False
    if not faces_detected:
        print(f"❌ ERROR: No se detectaron caras en el video procesado.")
        faces_detected_globally = False  # Video sin caras
        return False

    # Si todo está bien, actualizar la variable global y devolver True
    faces_detected_globally = True  # Detección exitosa
    return True
