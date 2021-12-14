import os


def get_csv_elements(parent_dir: str):
    csv_line = ''
    for content in os.listdir(parent_dir):
        content_path = os.path.join(parent_dir, content)
        if not os.path.isdir(content_path):
            continue

        if not content.startswith('level'):
            csv_line += get_csv_elements(content_path)
            continue

        audio_path = os.path.join(content_path, 'audio')
        csv_elements = content_path.replace('processed_data\\', '')
        csv_elements = csv_elements.replace('train\\', '')
        csv_elements = csv_elements.replace('test\\', '')
        csv_elements = csv_elements.replace('\\', ',')
        csv_elements = csv_elements.replace('level_', '')
        for name in os.listdir(audio_path):
            name = name.split('.')[0]
            csv_line += f'{csv_elements},{name}\n'

    return csv_line


def main():
    root = 'processed_data'
    col_names = 'subject,emotion,level,audio\n'
    for state in ['train', 'val']:
        csv = get_csv_elements(os.path.join(root, state))
        with open(f'processed_data/{state}_dataset.csv', 'w') as file:
            file.write(col_names + csv)


if __name__ == '__main__':
    main()
