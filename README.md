import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Abrir cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Voltear imagen tipo espejo
    frame = cv2.flip(frame, 1)

    # Convertir BGR a RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar manos
    results = hands.process(rgb)

    h, w, _ = frame.shape

    # Guardar posiciones centrales
    centros = []

    if results.multi_hand_landmarks and results.multi_handedness:

        for hand_landmarks, hand_info in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            # Dibujar landmarks y conexiones
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Obtener tipo de mano
            lado = hand_info.classification[0].label

            # Landmark 0 = centro de la palma
            x = int(hand_landmarks.landmark[0].x * w)
            y = int(hand_landmarks.landmark[0].y * h)

            centros.append((x, y))

            # Color diferente para izquierda y derecha
            if lado == "Left":
                color = (255, 0, 0)  # Azul
                texto = "Izquierda"
            else:
                color = (0, 255, 0)  # Verde
                texto = "Derecha"

            # Dibujar circulo
            cv2.circle(frame, (x, y), 10, color, -1)

            # Mostrar texto
            cv2.putText(
                frame,
                texto,
                (x - 40, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # Unir las manos con una línea
        if len(centros) == 2:
            cv2.line(
                frame,
                centros[0],
                centros[1],
                (0, 0, 255),
                4
            )

    # Mostrar ventana
    cv2.imshow("Detector de Manos", frame)

    # Salir con ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
