import os
import subprocess


def convert_to_wav(file, out_folder, clear_meta_data=False, out_folder_metadata=""):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    file_name = os.path.basename(file)
    file_name = file_name.split(".")[0] + ".wav"
    cmd = 'ffmpeg -i ' + file + " " + os.path.join(out_folder, file_name)
    subprocess.call(cmd, shell=False)
    print('WAV - Done')

    if clear_meta_data:
        clear_audio_metadata(out_folder, file_name, out_folder_metadata)


def clear_audio_metadata(wavs_path, new_audio_file, clear_path):
    if not os.path.exists(clear_path):
        os.makedirs(clear_path)

    cmd = 'ffmpeg -i ' + os.path.join(wavs_path, new_audio_file) + ' -map_metadata -1 -c:v copy -c:a copy '\
          + os.path.join(clear_path, new_audio_file)
    subprocess.call(cmd, shell=False)
    print('Metadata - Clear')


# Example

# audio_dir = 'MEAD/M003/audio/angry/level_2/001.m4a'
# output_dir = 'processed_data/M003/angry/level_2/audio'
# clean_dir = 'processed_data/M003/angry/level_2/clean_audio'
# convert_to_wav(audio_dir, output_dir, clear_meta_data=True, out_folder_metadata=clean_dir)
