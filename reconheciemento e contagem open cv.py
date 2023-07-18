import cv2

# Carrega o classificador pré-treinado para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializa a contagem de faces
count = 0

# Função para detectar e contar faces em uma imagem
def detect_and_count_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Realiza a detecção de faces na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Atualiza a contagem de faces
    count = len(faces)

    # Desenha os retângulos em volta das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Escreve a contagem de faces na imagem
    cv2.putText(image, f'Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    return image, count

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)  # Use 0 para a webcam, ou especifique o caminho para um arquivo de vídeo

while True:
    # Lê o próximo frame do vídeo
    ret, frame = cap.read()

    # Verifica se a captura de vídeo foi bem-sucedida
    if not ret:
        break

    # Realiza a detecção e contagem de faces no frame
    frame, count = detect_and_count_faces(frame)

    # Exibe o frame resultante com a contagem de faces
    cv2.imshow('Face Detection', frame)

    # Verifica se a tecla 'q' foi pressionada para encerrar o programa
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
