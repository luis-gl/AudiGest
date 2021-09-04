import os


def get_csv_elements(parent_dir: str):
    csv_line = ''
    for content in os.listdir(parent_dir):
        content_path = os.path.join(parent_dir, content)
        if os.path.isdir(content_path):
            if not content.startswith('level'):
                csv_line += get_csv_elements(content_path)
                continue
            names = [file.split('.')[0] for file in os.listdir(os.path.join(content_path, 'audio'))]
            csv_elements = content_path.replace('processed_data\\', '')
            csv_elements = csv_elements.replace('\\', ',')
            csv_elements = csv_elements.replace('level_', '')
            for name in names:
                csv_line += f'{csv_elements},{name}.wav,{name}.npy\n'
    return csv_line


def main():
    root = 'processed_data'
    col_names = 'subject,emotion,level,audio,landmarks\n'
    csv = get_csv_elements(root)
    with open('processed_data/data.csv', 'w') as file:
        file.write(col_names + csv)


if __name__ == '__main__':
    main()
