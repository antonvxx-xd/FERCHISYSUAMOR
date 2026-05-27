import cv2
import mediapipe as mp
import time
import random

CAMARA_INDEX = 0

ANCHO = 1280
ALTO  = 720

MOSTRAR_LANDMARKS = True

COLOR_FACE = (0, 255, 180)
COLOR_HAND = (255, 150, 0)

mp_face_mesh = mp.solutions.face_mesh
mp_hands     = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def texto(img, txt, pos, color=(255,255,255), escala=0.6):

    cv2.putText(
        img,
        txt,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        escala,
        color,
        2,
        cv2.LINE_AA
    )

def detectar_emocion():

    emociones = [
        "Feliz ",
        "Neutral ",
        "Sorprendido ",
        "Triste "
    ]

    return random.choice(emociones)


def estimar_genero():

    return random.choice(["Hombre", "Mujer"])


def estimar_edad():

    edad = random.randint(18, 45)

    return edad


def dibujar_face_landmarks(frame, landmarks, w, h):

    puntos = []

    for lm in landmarks.landmark:

        x = int(lm.x * w)
        y = int(lm.y * h)

        puntos.append((x, y))

    # Dibujar puntos
    for x, y in puntos:

        cv2.circle(frame, (x, y), 1, COLOR_FACE, -1)

    # Bounding Box
    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]

    x1 = max(min(xs)-20, 0)
    y1 = max(min(ys)-20, 0)

    x2 = min(max(xs)+20, w)
    y2 = min(max(ys)+20, h)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        COLOR_FACE,
        2
    )

    return x1, y1, x2, y2


def dibujar_hands(frame, hand_landmarks, w, h):

    puntos = []

    for lm in hand_landmarks.landmark:

        x = int(lm.x * w)
        y = int(lm.y * h)

        puntos.append((x, y))

    # Dibujar conexiones simples
    for x, y in puntos:

        cv2.circle(frame, (x, y), 5, COLOR_HAND, -1)

    xs = [p[0] for p in puntos]
    ys = [p[1] for p in puntos]

    x1 = max(min(xs)-20, 0)
    y1 = max(min(ys)-20, 0)

    x2 = min(max(xs)+20, w)
    y2 = min(max(ys)+20, h)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        COLOR_HAND,
        2
    )

def main():

    cap = cv2.VideoCapture(CAMARA_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)

    if not cap.isOpened():

        print("ERROR: Cámara no disponible")
        return

    print("="*60)
    print(" SISTEMA DE RECONOCIMIENTO AVANZADO ")
    print("="*60)

    prev_time = time.time()

    fps = 0

    capturas = 0

    mostrar = MOSTRAR_LANDMARKS

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ─────────────────────────────
        # FACE
        # ─────────────────────────────

        resultado_face = face_mesh.process(rgb)

        if resultado_face.multi_face_landmarks:

            for face_landmarks in resultado_face.multi_face_landmarks:

                x1, y1, x2, y2 = dibujar_face_landmarks(
                    frame,
                    face_landmarks,
                    w,
                    h
                )

                # DATOS SIMULADOS
                edad   = estimar_edad()
                genero = estimar_genero()
                emocion = detectar_emocion()

                texto(
                    frame,
                    f"Edad: {edad}",
                    (x1, y1 - 60),
                    COLOR_FACE
                )

                texto(
                    frame,
                    f"Genero: {genero}",
                    (x1, y1 - 35),
                    COLOR_FACE
                )

                texto(
                    frame,
                    f"Emocion: {emocion}",
                    (x1, y1 - 10),
                    COLOR_FACE
                )

        # ─────────────────────────────
        # HANDS
        # ─────────────────────────────

        resultado_hands = hands.process(rgb)

        if resultado_hands.multi_hand_landmarks:

            for hand_landmarks in resultado_hands.multi_hand_landmarks:

                dibujar_hands(
                    frame,
                    hand_landmarks,
                    w,
                    h
                )

        # ─────────────────────────────
        # FPS
        # ─────────────────────────────

        current_time = time.time()

        fps = 0.9 * fps + 0.1 * (
            1 / max(current_time - prev_time, 1e-6)
        )

        prev_time = current_time

        texto(
            frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            (0,255,0),
            0.8
        )

        texto(
            frame,
            "Q=Salir  S=Captura  M=Landmarks",
            (20, 75),
            (200,200,200),
            0.6
        )

        # ─────────────────────────────
        # MOSTRAR
        # ─────────────────────────────

        cv2.imshow(
            "Reconocimiento Facial + Manos",
            frame
        )

        # ─────────────────────────────
        # TECLAS
        # ─────────────────────────────

        key = cv2.waitKey(1) & 0xFF

        if key in [ord('q'), 27]:

            break

        elif key == ord('s'):

            capturas += 1

            nombre = f"captura_{capturas:03d}.png"

            cv2.imwrite(nombre, frame)

            print(f"[INFO] Captura guardada: {nombre}")

        elif key == ord('m'):

            mostrar = not mostrar

    # ─────────────────────────────
    # CERRAR
    # ─────────────────────────────

    cap.release()

    cv2.destroyAllWindows()

    face_mesh.close()

    hands.close()

    print("Sistema finalizado")


if __name__ == "__main__":

    main()