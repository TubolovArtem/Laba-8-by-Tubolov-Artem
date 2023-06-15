import cv2
import numpy as np

ref_image = cv2.imread('ref-point.jpg', cv2.IMREAD_GRAYSCALE)
orb = cv2.ORB_create()
keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_image, None)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

    matches = matcher.match(descriptors_ref, descriptors_frame)

    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:10]

    ref_center = np.array([ref_image.shape[1] / 2, ref_image.shape[0] / 2], dtype=np.float32)

    frame_center = np.zeros(2, dtype=np.float32)
    count = 0

    for match in good_matches:
        ref_point = keypoints_ref[match.queryIdx].pt
        frame_point = keypoints_frame[match.trainIdx].pt

        frame_center += frame_point
        count += 1

    if count > 0:
        frame_center /= count

    cv2.line(frame, (int(frame_center[0]), 0), (int(frame_center[0]), frame.shape[0]), (0, 255, 0), 2)
    cv2.line(frame, (0, int(frame_center[1])), (frame.shape[1], int(frame_center[1])), (0, 255, 0), 2)

    cv2.imshow('Tracked Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
