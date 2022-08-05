import cv2

# Take input of the person name

name = input("Enter name:  ")

# Create the videocapture object
cap = cv2.VideoCapture(0)

while True:

    # Read each frame
    success, frame = cap.read()

    # Show the output
    cv2.imshow("Frame", frame)

    # If 'c' key is pressed then click picture
    if cv2.waitKey(1) == ord('c'):
        filename = 'faces'+name+'.jpg'
        cv2.imwrite(filename, frame)
        print("Image Saved- ",filename)

    # if 'q' is pressed then break the loop and exit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Finally release the camera and destroy ll active windows
cap.release()
cv2.destroyAllWindows()