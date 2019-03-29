import cv2
import re


annotation_folder = "/Users/sameerwatve/Desktop/ProjectCV/Original_Data/annotations"
image_folder = "/Users/sameerwatve/Desktop/ProjectCV/Original_Data/originalPics"
face_destination_folder = "train_face_images/"
nonface_destination_folder = "train_nonface_images/"

path_location = annotation_folder + "/FDDB-fold-"

number = '1'
file1 = open(path_location + number.zfill(2) +".txt", "r")
file_locations = file1.readlines()

for loc in range(len(file_locations)):
    path_image = image_folder + "/" + file_locations[loc][:-1]
    param_file = open (annotation_folder + "/FDDB-fold-" + number.zfill(2) +"-ellipseList.txt", "r")
    lineCount = 0
    for line in param_file:
        lineCount += 1
        if line.rstrip() == file_locations[loc][:-1] :
            print("Location", file_locations[loc][:-1])
            print("Found at: ", lineCount)
            break
    param_file.close()
    param_file = open (annotation_folder + "/FDDB-fold-" + number.zfill(2) +"-ellipseList.txt", "r")
    lines = param_file.readlines()
    num_faces = lines[lineCount]
    if (num_faces[:-1] == '1'):
        data = lines[lineCount + 1]
        print("Data = ", data)
        vals = re.findall("\d+\.\d+", data)
        vals = [float(i) for i in vals]
        print(vals)

        major_axis_radius = int(vals[0])
        minor_axis_radius =  int(vals[1])
        angle = int(vals[2])
        center_x = int(vals[3]) 
        center_y = int(vals[4])  

        img = cv2.imread(image_folder+"/"+file_locations[loc][:-1]+".jpg")
        [rows,cols,dim] = img.shape
        #img[r1:r1+height, c1:c1+width]
        img_cropped = img[center_y - major_axis_radius : center_y + major_axis_radius, 
                   center_x - minor_axis_radius : center_x + minor_axis_radius]
        img_non_face = img[center_y + major_axis_radius : rows , center_x + minor_axis_radius : cols]
        #cv2.imshow("Original",img)
        #cv2.imshow("Cropped",img_cropped)
        [r1, c1, d] = img_cropped.shape
        [r2, c2, d] = img_non_face.shape
         
        if (r1>=60 and c1>=60 and r2>=60 and c2>=60):              
            img_resized = cv2.resize(img_cropped, (10,10))
            img_nonface_resized = cv2.resize(img_non_face, (10,10))
            cv2.waitKey(1000)
            cv2.imwrite(face_destination_folder+str(number)+"_"+str(loc)+".jpg",img_resized)
            cv2.imwrite(nonface_destination_folder+str(number)+"_"+str(loc)+".jpg",img_nonface_resized)

cv2.destroyAllWindows()

