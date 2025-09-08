import cv2

def main():
    #Esta es la funcion para abrir la camara(el 0 es la camara de la computadora)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("La camara no se encuentra activa o no se puede abrir")
        return

    #Con esto podemos cargar el archivo xml para reconocer caras
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            print("El frame no se logra leer")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Este es el detector de rostros
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor = 1.1, minNeighbors=5, minSize=(60, 60)
        )

        #Las cajas que identifican las caras
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        #Nombre de la ventana emergente
        cv2.imshow("Face Tracking UAV", frame)


        #Cerrar el programa con la tecla ESC o Q
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q") or ("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()