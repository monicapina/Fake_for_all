# real_esrgan_batch.py

import os
import glob
import cv2
from multiprocessing import Process
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


class RealESRGANBatchProcessor:
    def __init__(self, model_name="realesr-general-x4v3", denoise_strength=0.3, outscale=1.1, tile=100, tile_pad=20,
                 fp32=True, face_enhance=True, suffix="upscaled", ext="png", gpu_id=None):
        self.model_name = model_name.split('.')[0]
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.tile = tile
        self.tile_pad = tile_pad
        self.fp32 = fp32
        self.face_enhance = face_enhance
        self.suffix = suffix
        self.ext = ext
        self.gpu_id = gpu_id
        self.upsampler, self.face_enhancer = self.load_model()

    def load_model(self):
        if self.model_name == 'realesr-general-x4v3':
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]
        else:
            raise NotImplementedError(f"Modelo {self.model_name} no implementado.")

        model_path = os.path.join('weights', self.model_name + '.pth')
        if not os.path.isfile(model_path):
            ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
            for url in file_url:
                model_path = load_file_from_url(url=url, model_dir=os.path.join(ROOT_DIR, 'weights'),
                                                progress=True, file_name=None)

        # usar dni para controlar la denoising
        dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            half=not self.fp32,
            gpu_id=self.gpu_id
        )

        face_enhancer = None
        if self.face_enhance:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
                upscale=self.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)

        return upsampler, face_enhancer

    def process_image(self, input_path, output_path):
        basename, ext = os.path.splitext(os.path.basename(input_path))
        output_image = os.path.join(output_path, f"{basename}_{self.suffix}.{self.ext}")
        if os.path.exists(output_image):
            print(f"[GPU {self.gpu_id}] Saltando ya procesada: {output_image}")
            return

        os.makedirs(output_path, exist_ok=True)
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        try:
            if self.face_enhancer:
                _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            else:
                output, _ = self.upsampler.enhance(img, outscale=self.outscale)
        except RuntimeError as error:
            print(f"Error procesando {input_path}: {error}")
            return

        cv2.imwrite(output_image, output)
        print(f"[GPU {self.gpu_id}] Guardado: {output_image}")


def process_images_for_gpu(task_list, gpu_id):
    processor = RealESRGANBatchProcessor(gpu_id=int(gpu_id))
    for input_path, output_path in task_list:
        processor.process_image(input_path, output_path)


def assign_images_to_gpus(input_root, output_root, available_gpus):
    gpu_tasks = {gpu: [] for gpu in available_gpus}
    gpu_index = 0

    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, os.path.dirname(rel_path))
                assigned_gpu = available_gpus[gpu_index % len(available_gpus)]
                gpu_tasks[assigned_gpu].append((input_path, output_path))
                gpu_index += 1

    processes = []
    for gpu_id, task_list in gpu_tasks.items():
        p = Process(target=process_images_for_gpu, args=(task_list, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Carpeta raíz de entrada')
    parser.add_argument('--output', required=True, help='Carpeta raíz de salida')
    parser.add_argument('--gpus', nargs='+', default=['0'], help='Lista de GPUs disponibles (por índice)')
    args = parser.parse_args()

    assign_images_to_gpus(args.input, args.output, args.gpus)
