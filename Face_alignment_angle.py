
import cv2
import numpy as np

face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("./haarcascade_eye.xml")

for number in range(1, 395):
  for number2 in range(1,2):
    print(number, number2)
    try:
      img = cv2.imread(f'./test/{number}.jpg')
      img_original = cv2.imread(f'./test/{number}.jpg')

      #--------------------------------&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&--------------------------------------------------

      '''
      cv2.imshow('original_image', img)
      cv2.waitKey(0)
      '''
      # Converting the image into grayscale
      gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

      # Creating variable faces
      faces= face_cascade.detectMultiScale (gray, 1.03, 2)

      # Defining and drawing the rectangle around the face
      for(x , y,  w,  h) in faces:
        cv2.rectangle(img, (x,y) ,(x+w, y+h), (0,255,0), 3)

      '''
      cv2.imshow('face_image', img)
      cv2.waitKey(0)
      '''
      # Creating two regions of interest
      roi_gray=gray[y:(y+h), x:(x+w)]
      roi_color=img[y:(y+h), x:(x+w)]

      roi_gray = gray
      roi_color = img
      # Creating variable eyes
      eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 50) #1.08 20This parameter means, (image, scaleFactor, minNeighbors, flags, minSize, maxSize)
      index=0 
      # Creating for loop in order to divide one eye from another
      eye_2 = (0,0,0,0)
      
      for (ex , ey,  ew,  eh) in eyes:
        if index == 0:
          eye_1 = (ex, ey, ew, eh)
          #print(eye_1)
        elif index == 1:
          eye_2 = (ex, ey, ew, eh)
          #print(eye_2)
      # Drawing rectangles around the eyes
        cv2.rectangle(roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
        index = index + 1

      

      if eye_1[0] < eye_2[0]:
        left_eye = eye_1
        right_eye = eye_2
      else:
        left_eye = eye_2
        right_eye = eye_1

      # Calculating coordinates of a central points of the rectangles
      left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
      left_eye_x = left_eye_center[0] 
      left_eye_y = left_eye_center[1]
      
      right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
      right_eye_x = right_eye_center[0]
      right_eye_y = right_eye_center[1]
      
      if (index == 0):
        print('no eyes')
        cv2.imwrite(f'./0105_error/{number}_{number2}_e.jpg', img)
        cv2.imwrite(f'./0105_error_original/{number}_{number2}.jpg', img_original)
        #cv2.imshow('no_eye_image', img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue


      if (index == 1):
        print('no eye_2')
        cv2.imwrite(f'./0105_error/{number}_{number2}_e.jpg', img)
        cv2.imwrite(f'./0105_error_original/{number}_{number2}.jpg', img_original)
        #cv2.imshow('no_eye_image', img)
        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue


      if(abs(left_eye_x-right_eye_x) < abs(right_eye_y-left_eye_y)):
        print('python mistakes', abs(left_eye_x-right_eye_x), abs(right_eye_y-left_eye_y))
        #cdx = abs(left_eye_center[0] - 120)
        #cdy = abs(left_eye_center[1] - 159)
        #cropped_image_original = img_original[cdy:(cdy + 350), cdx: (cdx + 350)]
        cv2.imwrite(f'./0105_error/{number}_{number2}_w.jpg', img)
        cv2.imwrite(f'./0105_error_original/{number}_{number2}.jpg', img_original)
        cv2.imshow('eye_distance_image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue

      cv2.circle(roi_color, left_eye_center, 5, (255, 255, 0) , -1)
      cv2.circle(roi_color, right_eye_center, 5, (255, 255, 0) , -1)


      cv2.line(roi_color,right_eye_center, left_eye_center,(100,100,100),3)

      if left_eye_y > right_eye_y:
        A = (right_eye_x, left_eye_y)
        # Integer -1 indicates that the image will rotate in the clockwise direction
        direction = -1 
      else:
        A = (left_eye_x, right_eye_y)
        # Integer 1 indicates that image will rotate in the counter clockwise  
        # direction
        direction = 1 

      cv2.circle(roi_color, A, 5, (255, 255, 0) , -1)
      
      cv2.line(roi_color,right_eye_center, left_eye_center,(204,51,204),3)
      cv2.line(roi_color,left_eye_center, A,(204,51,204),3)
      cv2.line(roi_color,right_eye_center, A,(204,51,204),3)
      
      cv2.imshow('calculated_image', img)
      cv2.waitKey(0)
      
      delta_x = right_eye_x - left_eye_x
      delta_y = right_eye_y - left_eye_y
      angle=np.arctan(delta_y/delta_x)
      angle = (angle * 180) / np.pi

      # Width and height of the image
      h, w = img.shape[:2]
      # Calculating a center point of the image
      # Integer division "//"" ensures that we receive whole numbers
      center = (w // 2, h // 2)
      # Defining a matrix M and calling
      # cv2.getRotationMatrix2D method
      M = cv2.getRotationMatrix2D(center, (angle), 1.0)
      # Applying the rotation to our image using the
      # cv2.warpAffine method
      rotated = cv2.warpAffine(img, M, (w, h))

      original_rotated = cv2.warpAffine(img_original, M, (w,h))

      #cv2.imshow('rotated_image', rotated)
      #cv2.imshow('original_rotated', original_rotated)

      #------------------------------------------------------------------------------------------------------------------------------------
      # Converting the image into grayscale
      al_gray=cv2.cvtColor(original_rotated, cv2.COLOR_BGR2GRAY)

      # Creating variable faces
      al_faces= face_cascade.detectMultiScale (al_gray, 1.02, 3)

      # Defining and drawing the rectangle around the face
      '''
      cv2.imshow('face_image', img)
      cv2.waitKey(0)
      '''
      
      '''
      # Creating two regions of interest
      al_roi_gray=al_gray[y:(y+h), x:(x+w)]
      al_roi_color=original_rotated[y:(y+h), x:(x+w)]
      '''
      aligned_eyes = eye_cascade.detectMultiScale(al_gray, 1.08, 50) #This parameter means, (image, scaleFactor, minNeighbors, flags, minSize, maxSize)
      index2=0
      # Creating for loop in order to divide one eye from another
      for (eex , eey, eew,  eeh) in aligned_eyes:
        if index2 == 0:
          al_eye_1 = (eex, eey, eew, eeh)
          #print(al_eye_1)
        elif index2 == 1:
          al_eye_2 = (eex, eey, eew, eeh)
          #print(al_eye_2)
        index2 = index2 + 1

      if al_eye_1[0] < al_eye_2[0]:
        al_left_eye = al_eye_1
        al_right_eye = al_eye_2
      else:
        al_left_eye = al_eye_2
        al_right_eye = al_eye_1

      al_left_eye_center = (int(al_left_eye[0] + (al_left_eye[2] / 2)), int(al_left_eye[1] + (al_left_eye[3] / 2)))
      
      al_right_eye_center = (int(al_right_eye[0] + (al_right_eye[2]/2)), int(al_right_eye[1] + (al_right_eye[3]/2)))

      al_eyes_x_center = round((al_left_eye_center[0] + al_right_eye_center[0])/2)
      #print(al_eyes_x_center)
      
      cx = abs(al_eyes_x_center - 200) #110 / al_eyes_x_center - 200 al_left_eye_center[0] - 140
      cy = abs(al_left_eye_center[1] - 205)#149
      print(cx, cy)
      
      

      cropped_img = original_rotated[cy:(cy + 400), cx:(cx + 400)]
      
      #------------------------------------------------------------------------------------------------------------------------------------
      #print(['여기까지는 되는것?'])

      Resize_image = cv2.resize(cropped_img, (300,300), interpolation=cv2.INTER_AREA)
     

      #cv2.imwrite('./1_30_4.jpg', original_rotated)

      cv2.imwrite(f'./test/{number}_angle.jpg', Resize_image)
      '''
      print(left_eye_center[0])
      print(left_eye_center[1])
      cx = abs(left_eye_center[0] - 8) 
      cy = abs(left_eye_center[1] - 1)
      '''
      


      #--------------------------------&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&-밤에 추가-------------------------------------------------


    except:
      print('no picture')
      continue



'''
cv2.imshow('original_image', img)
cv2.waitKey(0)

# Converting the image into grayscale
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Creating variable faces
faces= face_cascade.detectMultiScale (gray, 1.03, 2)

# Defining and drawing the rectangle around the face
for(x , y,  w,  h) in faces:
  cv2.rectangle(img, (x,y) ,(x+w, y+h), (0,255,0), 3)


cv2.imshow('face_image', img)
cv2.waitKey(0)

# Creating two regions of interest
roi_gray=gray[y:(y+h), x:(x+w)]
roi_color=img[y:(y+h), x:(x+w)]

roi_gray = gray
roi_color = img
# Creating variable eyes
eyes = eye_cascade.detectMultiScale(roi_gray, 1.08, 20) #This parameter means, (image, scaleFactor, minNeighbors, flags, minSize, maxSize)
index=0
# Creating for loop in order to divide one eye from another
for (ex , ey,  ew,  eh) in eyes:
  if index == 0:
    eye_1 = (ex, ey, ew, eh)
    print(eye_1)
  elif index == 1:
    eye_2 = (ex, ey, ew, eh)
    print(eye_2)
# Drawing rectangles around the eyes
  cv2.rectangle(roi_color, (ex,ey) ,(ex+ew, ey+eh), (0,0,255), 3)
  index = index + 1

if eye_1[0] < eye_2[0]:
   left_eye = eye_1
   right_eye = eye_2
else:
   left_eye = eye_2
   right_eye = eye_1

# Calculating coordinates of a central points of the rectangles
left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
left_eye_x = left_eye_center[0] 
left_eye_y = left_eye_center[1]
 
right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
right_eye_x = right_eye_center[0]
right_eye_y = right_eye_center[1]
 
cv2.circle(roi_color, left_eye_center, 5, (255, 255, 0) , -1)
cv2.circle(roi_color, right_eye_center, 5, (255, 255, 0) , -1)


cv2.line(roi_color,right_eye_center, left_eye_center,(100,100,100),3)

if left_eye_y > right_eye_y:
   A = (right_eye_x, left_eye_y)
   # Integer -1 indicates that the image will rotate in the clockwise direction
   direction = -1 
else:
   A = (left_eye_x, right_eye_y)
  # Integer 1 indicates that image will rotate in the counter clockwise  
  # direction
   direction = 1 

cv2.circle(roi_color, A, 5, (255, 255, 0) , -1)
 
cv2.line(roi_color,right_eye_center, left_eye_center,(204,51,204),3)
cv2.line(roi_color,left_eye_center, A,(204,51,204),3)
cv2.line(roi_color,right_eye_center, A,(204,51,204),3)
'''
'''
cv2.imshow('calculated_image', img)
cv2.waitKey(0)
'''
'''
delta_x = right_eye_x - left_eye_x
delta_y = right_eye_y - left_eye_y
angle=np.arctan(delta_y/delta_x)
angle = (angle * 180) / np.pi

# Width and height of the image
h, w = img.shape[:2]
# Calculating a center point of the image
# Integer division "//"" ensures that we receive whole numbers
center = (w // 2, h // 2)
# Defining a matrix M and calling
# cv2.getRotationMatrix2D method
M = cv2.getRotationMatrix2D(center, (angle), 1.0)
# Applying the rotation to our image using the
# cv2.warpAffine method
rotated = cv2.warpAffine(img, M, (w, h))

original_rotated = cv2.warpAffine(img_original, M, (w,h))\

#cv2.imshow('rotated_image', rotated)
#cv2.imshow('original_rotated', original_rotated)

#------------------------------------------------------------------------------------------------------------------------------------
# Converting the image into grayscale
al_gray=cv2.cvtColor(original_rotated, cv2.COLOR_BGR2GRAY)

# Creating variable faces
al_faces= face_cascade.detectMultiScale (al_gray, 1.02, 3)

# Defining and drawing the rectangle around the face

cv2.imshow('face_image', img)
cv2.waitKey(0)

'''
'''
# Creating two regions of interest
al_roi_gray=al_gray[y:(y+h), x:(x+w)]
al_roi_color=original_rotated[y:(y+h), x:(x+w)]
'''
'''
aligned_eyes = eye_cascade.detectMultiScale(al_gray, 1.08, 20) #This parameter means, (image, scaleFactor, minNeighbors, flags, minSize, maxSize)
index2=0
# Creating for loop in order to divide one eye from another
for (eex , eey, eew,  eeh) in aligned_eyes:
  if index2 == 0:
    al_eye_1 = (eex, eey, eew, eeh)
    print(al_eye_1)
  elif index2 == 1:
    al_eye_2 = (eex, eey, eew, eeh)
    print(al_eye_2)
  index2 = index2 + 1

if al_eye_1[0] < al_eye_2[0]:
   al_left_eye = al_eye_1
   al_right_eye = al_eye_2
else:
   al_left_eye = al_eye_2
   al_right_eye = al_eye_1


al_left_eye_center = (int(al_left_eye[0] + (al_left_eye[2] / 2)), int(al_left_eye[1] + (al_left_eye[3] / 2)))
 
al_right_eye_center = (int(al_right_eye[0] + (al_right_eye[2]/2)), int(al_right_eye[1] + (al_right_eye[3]/2)))

cx = abs(al_left_eye_center[0] - 110)
cy = abs(al_left_eye_center[1] - 149)
print(cx, cy)

cropped_img = original_rotated[cy:(cy + 340), cx: (cx + 340)]
#------------------------------------------------------------------------------------------------------------------------------------


cv2.resize(cropped_img, (300,300))
cv2.imshow('cropped_image', cropped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./1_30_4.jpg', original_rotated)

cv2.imwrite(f'C:/Users/선종엽/Desktop/UGRP/dnnface/python_face_align/0912_angle_aligned/{number}_{number2}.jpg', cropped_img)
'''
'''
print(left_eye_center[0])
print(left_eye_center[1])
cx = abs(left_eye_center[0] - 8) 
cy = abs(left_eye_center[1] - 1)
'''
