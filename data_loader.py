from data_process import load_dictionary


def main():
    face_landmarks = load_dictionary('sbj_face_landmarks.pkl')
    print(len(face_landmarks))


if __name__ == '__main__':
    main()
