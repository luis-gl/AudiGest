import os
from utils.files.save import save_pickle, load_pickle


def create_subject_dict(config: dict, project_root: str = None):
    phases = ['train', 'val', 'test']
    data_root = 'MEAD'
    emotions = config['emotions']
    files_dict = config['files']
    audio_common_path = 'audio'
    video_common_path = os.path.join('video', 'front')

    if project_root is not None:
        data_root = os.path.join(project_root, data_root)
    if not os.path.exists(data_root) or not os.path.isdir(data_root):
        raise ValueError('MEAD dataset is not in {} directory'.format(data_root))

    subject_dict = {}
    for phase in phases:
        phase_subjects = files_dict[phase]['subjects']

        subject_dict[phase] = {}
        for sbj in phase_subjects:
            audio_path = os.path.join(data_root, sbj, audio_common_path)
            video_path = os.path.join(data_root, sbj, video_common_path)

            emotion_dict = {}
            for e in emotions:
                audio_e_path = os.path.join(audio_path, e)
                video_e_path = os.path.join(video_path, e)
                levels = os.listdir(audio_e_path)

                lv_dict = {}
                for lv in levels:
                    audio_lv_path = os.path.join(audio_e_path, lv)
                    video_lv_path = os.path.join(video_e_path, lv)
                    audio_files = os.listdir(audio_lv_path)
                    video_files = os.listdir(video_lv_path)

                    lv_dict[lv] = {}
                    audio_list = []
                    video_list = []
                    for i in range(len(audio_files)):
                        audio_fpath = os.path.join(audio_lv_path, audio_files[i])
                        # Change '\\' by '/' if running on Linux
                        audio_fpath = audio_fpath.replace('\\', '/')
                        video_fpath = os.path.join(video_lv_path, video_files[i])
                        video_fpath = video_fpath.replace('\\', '/')
                        audio_list.append(audio_fpath)
                        video_list.append(video_fpath)
                    lv_dict[lv]['audio'] = audio_list
                    lv_dict[lv]['video'] = video_list

                emotion_dict[e] = lv_dict

            subject_dict[phase][sbj] = emotion_dict

        print(f'Using {len(subject_dict[phase])} {phase} subjects')

    return subject_dict


def get_data(config: dict, project_root: str = None):
    dict_fname = 'sbj_data_paths.pkl'
    try:
        data_dict = load_pickle(dict_fname)
    except ValueError:
        print(f'Creating new subject paths data.')
        data_dict = create_subject_dict(config=config, project_root=project_root)
        save_pickle(data_dict, dict_fname)
        print(f'Data saved on processed_data/{dict_fname}.')
        return data_dict
    else:
        print(f'Loading existing subject video paths data.')
        return data_dict


def print_subject_dict(data_dict: dict = None):
    if not data_dict or data_dict is None:
        raise ValueError('Dictionary is None or Empty')

    phases = ['train', 'val', 'test']
    for phase in phases:
        phase_dict = data_dict[phase]
        print(f'{phase}:')
        for sbj in phase_dict:
            sbj_dict = phase_dict[sbj]
            print(f'\t{sbj}:')
            for e in sbj_dict:
                e_dict = sbj_dict[e]
                print(f'\t\t{e}:')
                for lv in e_dict:
                    print(f'\t\t\t{lv}')
                    audio = e_dict[lv]['audio']
                    video = e_dict[lv]['video']
                    for i in range(len(audio)):
                        print(f'\t\t\t\taudio: {audio[i]}\tvideo: {video[i]}')
