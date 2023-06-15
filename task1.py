import cv2

image_path = 'images/variant-8.jpg'
image = cv2.imread(image_path)

height, width = image.shape[:2]

left = (width - 400) // 2
top = (height - 400) // 2
right = left + 400
bottom = top - 400

cropped_image = image[top:bottom, left:right]

output_path = 'resFiles/task1_resImage.jpg'
cv2.imwrite(output_path, cropped_image)
