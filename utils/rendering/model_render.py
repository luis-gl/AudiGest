import cv2
import numpy as np
import os
import threading

from psbody.mesh import Mesh
from subprocess import call
from utils.rendering.rendering import render_mesh_helper

class ModelRender:
    def __init__(self, config: dict):
        self.template_mesh = Mesh(filename=config['files']['face'])
    
    def set_up(self, audio_path=None, out_folder=None, video_path=None):
        self.audio_path = audio_path
        self.out_folder = out_folder
        self.video_path = os.path.join(out_folder,video_path)

    def render_sequences(self, reconstructed, run_in_parallel=True):
        if run_in_parallel:
            thread = threading.Thread(target=self._render_helper, args=(reconstructed,))
            thread.start()
            thread.join()
        else:
            self._render_helper(reconstructed)

    def _render_helper(self, reconstructed):
            if not os.path.exists(self.out_folder):
                os.makedirs(self.out_folder)


            video_fname = f'{self.video_path}.wmv'
            temp_video_fname = f'{self.video_path}_tmp.wmv'
            self._render_sequences_helper(reconstructed, video_fname, temp_video_fname)

    def _render_sequences_helper(self, reconstructed, video_fname, temp_video_fname):
        num_frames = reconstructed.shape[0]

        if int(cv2.__version__[0]) < 3:
            print('cv2 < 3')
            writer = cv2.VideoWriter(temp_video_fname, cv2.cv.CV_FOURCC(*'wmv2'), 30, (800, 800), True)
        else:
            print('cv2 >= 3')
            writer = cv2.VideoWriter(temp_video_fname, cv2.VideoWriter_fourcc(*'wmv2'), 30, (800, 800), True)

        reconstructed = self.scale_face(reconstructed)

        for i_frame in range(num_frames):
            pred_img = render_mesh_helper(Mesh(reconstructed[i_frame], self.template_mesh.f))
            writer.write(pred_img)
        writer.release()
        
        cmd = (f'ffmpeg -i {self.audio_path} -i {temp_video_fname} -codec copy -ac 2 -channel_layout stereo {video_fname}').split()
        call(cmd)

        if os.path.exists(temp_video_fname):
            os.remove(temp_video_fname)
    
    def scale_face(self, reconstructed):
        reconstructed[:,:,0] *= 1.6
        reconstructed[:,:,1] *= 0.9
        reconstructed[:,:,2] *= 1.6
        return reconstructed