import os
from utils.files.save import save_pickle, load_pickle


def get_valid_subjects_paths(project_root: str = '') -> dict:
    data_root = 'MEAD'
    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    data_path = os.path.join(project_root, data_root)
    if not os.path.exists(data_path) or not os.path.isdir(data_path):
        raise ValueError('MEAD dataset is not in {} directory'.format(project_root))

    subjects = [sbj for sbj in os.listdir(data_path)]

    audio_common_path = 'audio'
    video_common_path = os.path.join('video', 'front')

    emotions_set = set(emotions)
    sbj_dict = {}

    for sbj in subjects:
        audio_path = os.path.join(project_root, data_root, sbj, audio_common_path)
        video_path = os.path.join(project_root, data_root, sbj, video_common_path)

        if not os.path.exists(audio_path) or not os.path.exists(video_path):
            print(f'Subject {sbj} filtered by common path')
            subjects.remove(sbj)
            continue

        sbj_audio_emotions = os.listdir(audio_path)
        sbj_video_emotions = os.listdir(video_path)

        per_quantity = len(sbj_audio_emotions) != len(sbj_video_emotions)
        per_content = set(sbj_audio_emotions) != emotions_set or set(sbj_video_emotions) != emotions_set
        if per_quantity or per_content:
            print(f'Subject {sbj} filtered by emotion directories')
            subjects.remove(sbj)
            continue

        sbj_removed = False
        e_dict = {}

        for e in emotions:
            audio_e_path = os.path.join(audio_path, e)
            video_e_path = os.path.join(video_path, e)

            audio_levels = os.listdir(audio_e_path)
            video_levels = os.listdir(video_e_path)

            per_quantity = len(audio_levels) != len(video_levels)
            per_content = set(audio_levels) != set(video_levels)
            if per_quantity or per_content:
                print(f'Subject {sbj} filtered by {e} level directories')
                subjects.remove(sbj)
                sbj_removed = True
                break

            lv_dict = {}

            for lv in audio_levels:
                audio_lv_path = os.path.join(audio_e_path, lv)
                video_lv_path = os.path.join(video_e_path, lv)

                audio_files = os.listdir(audio_lv_path)
                video_files = os.listdir(video_lv_path)
                audio_fnames = [os.path.splitext(fname)[0] for fname in audio_files]
                video_fnames = [os.path.splitext(fname)[0] for fname in video_files]

                per_quantity = len(audio_fnames) != len(video_fnames)
                per_content = set(audio_fnames) != set(video_fnames)
                if per_quantity or per_content:
                    reason = 'quantity' if per_quantity else 'content'
                    print(f'Subject {sbj} filtered by {e}/{lv} {reason} files:')
                    subjects.remove(sbj)
                    sbj_removed = True
                    break

                lv_dict[lv] = {}
                lv_dict[lv]['audio'] = []
                lv_dict[lv]['video'] = []
                for i in range(len(audio_files)):
                    audio_fpath = os.path.join(audio_lv_path, audio_files[i])
                    audio_fpath = audio_fpath.replace('\\', '/')
                    video_fpath = os.path.join(video_lv_path, video_files[i])
                    video_fpath = video_fpath.replace('\\', '/')
                    lv_dict[lv]['audio'].append(audio_fpath)
                    lv_dict[lv]['video'].append(video_fpath)

            if sbj_removed:
                break

            e_dict[e] = lv_dict

        if sbj_removed:
            continue

        sbj_dict[sbj] = e_dict

    print(f'using {len(sbj_dict)}')
    print(f'{len(subjects)} subjects remain:\n{subjects}')

    return sbj_dict


def get_data(project_root: str = ''):
    dict_fname = 'sbj_data_paths.pkl'
    try:
        data_dict = load_pickle(dict_fname)
    except ValueError:
        print(f'Creating new subject paths data.')
        data_dict = get_valid_subjects_paths(project_root=project_root)
        save_pickle(data_dict, dict_fname)
        print(f'Data saved on processed_data/{dict_fname}, using {len(data_dict)} subjects.')
        return data_dict
    else:
        print(f'Loading existing subject video paths data, using {len(data_dict)} subjects.')
        return data_dict


def print_subject_dict(data_dict: dict = None):
    if not data_dict or data_dict is None:
        raise ValueError('Dictionary is None or Empty')

    for sbj in data_dict:
        print(f'{sbj}:')
        for e in data_dict[sbj]:
            print(f'\t{e}:')
            for lv in data_dict[sbj][e]:
                print(f'\t\t{lv}')
                audio = data_dict[sbj][e][lv]['audio']
                video = data_dict[sbj][e][lv]['video']
                for i in range(len(audio)):
                    print(f'\t\t\taudio: {audio[i]}\tvideo: {video[i]}')
