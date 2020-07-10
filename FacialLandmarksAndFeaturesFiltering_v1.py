import dlib
import cv2
import numpy as np
import utils as utils   # Custom


# Required functions..
def extract_facial_parts(imgToCropped, points, scaleFactor=1, masked=True, cropped=True):
    """
    This function will extract the facial part from the given image(imgToCropped) with the help of points(facial landmark points) sent.

    :param imgToCropped: The original image to be cropped
    :param points: The points of facial landmark detected
    :param scaleFactor: with what factor the cropped image to be enlarged or resized
    :param masked: If True, only that facial part that represent which points resemble are displayed by masking the rest portion, else not
            (NOTE: The size of the image remains same, just the required facial part gets visible and rest masked...!!).
    :param cropped: If True, the passed image gets cropped as per the points sent, else not.
            (NOTE: The size of the passed image changes to the size of facial_part (approzimately))
    :return: If both masked=True and cropped=True, then masked cropped image will be returned, if either one is true, that respective image is sent.
            (NOTE: If both are false returned a black image which is of the size of passed image,
                    so at least one or the either flag must be set to True.. to get the desired result..!!)

    WARNING:
    # NOTE: Atleast pass True either for masked or cropped..!! Else gets an error message, resolve this error..!!

    How its extended?
        Previously on one mask(Full black image) only one facial part is taken out by cv2.fillPoly(), now rather than taking the mask
            for each facial part, we used the mask of the previous facial part for the next one.
        After all the facial parts are completely done, returning the mask of all facial parts
    """
    # If in case, both the masked and cropped flags are set to False, called function receives a black image, else gets desired result..!!
    facial_part_mask = np.zeros_like(imgToCropped)
    if masked:
        # Create the mask image, so we can extract only the required part that resembles the passed set of facial landmark points..
        # This line of code is shifted to above this if condition for exception handling...
        # fill the points region with white, as when we perform bitwise_and(), only our target part region is visible and rest gets masked...
        cv2.fillPoly(img=facial_part_mask, pts=[points], color=(255, 255, 255))  # Make sure to pass points as [points] .... Else error as "cv2.error: OpenCV(4.3.0) ..\modules\imgproc\src\drawing.cpp:2374: error: (-215:Assertion failed) p.checkVector(2, CV_32S) >= 0 in function 'cv::fillPoly'"

    if cropped:
        # Perform bitwise_and() to make visible out the  required facial part..
        facial_part_mask = cv2.bitwise_and(imgToCropped, facial_part_mask)
        # If user would like to mask.. as well as crop..
        imgToCropped = facial_part_mask  # This will be used in the below section if cropped..
        # we need to extract the face_part from the given image and given points on image..
        x, y, width, height = cv2.boundingRect(points)
        # imgToCropped = cv2.rectangle(img=facial_part, pt1=(x, y), pt2=(x + width, y + height), color=(0, 255, 0), thickness=1)
        facial_part_cropped = imgToCropped[y: y + height, x: x+width]
        # As cropped image will be very small, lets resize it a bit bigger..
        facial_part_cropped = cv2.resize(src=facial_part_cropped, dsize=(0, 0), fx=scaleFactor, fy=scaleFactor)
        return facial_part_cropped

    # If the user would only like to mask..rather than cropping.. (NOTE: if even masked=False, returns a black image of size of the image sent)
    return facial_part_mask


def extract_facial_parts_extended(imgToCropped, multiple_facial_points, scaleFactor=1, masked=False, cropped=True):
    """
    This function is the extended version of "extract_face_part", with a extended version of extracting multiple facial parts rather than just
    single facial part by multiple facial landmark points

    :param imgToCropped: The original image to be cropped
    :param  points: The points of facial landmark detected
    :param scaleFactor: with what factor the cropped image to be enlarged or resized
    :param masked: If True, only that facial part that represent which points resemble are displayed by masking the rest portion, else not
            (NOTE: The size of the image remains same, just the required facial part gets visible and rest masked...!!).
    :param cropped: If True, the passed image gets cropped as per the points sent, else not.
            (NOTE: The size of the passed image changes to the size of facial_part (approzimately))
    :return: If both masked=True and cropped=True, then masked cropped image will be returned, if either one is true, that respective image is sent.
            (NOTE: If both are false returned a black image which is of the size of passed image,
                    so at least one or the either flag must be set to True.. to get the desired result..!!)

    WARNING:
    # NOTE: Atleast pass True either for masked or cropped..!! Else gets an error message, resolve this error..!!
    """
    # Mask image (A black image)
    facial_parts_mask = np.zeros_like(imgToCropped)
    for facial_part_points in multiple_facial_points:
        if masked:
            # Create the mask image, so we can extract only the required part that resembles the passed set of facial landmark points..
            # This line of code is shifted to above this if condition for exception handling...
            # fill the points region with white, as when we perform bitwise_and(), only our target part region is visible and rest gets masked...
            cv2.fillPoly(img=facial_parts_mask, pts=[facial_part_points], color=(255, 255, 255))  # Make sure to pass points as [points] .... Else error as "cv2.error: OpenCV(4.3.0) ..\modules\imgproc\src\drawing.cpp:2374: error: (-215:Assertion failed) p.checkVector(2, CV_32S) >= 0 in function 'cv::fillPoly'"

        if cropped:
            # Perform bitwise_and() to make visible out the  required facial part..
            facial_part_mask = cv2.bitwise_and(imgToCropped, facial_parts_mask)
            # If user would like to mask.. as well as crop..
            imgToCropped = facial_part_mask  # This will be used in the below section if cropped..
            # we need to extract the face_part from the given image and given points on image..
            x, y, width, height = cv2.boundingRect(facial_part_points)
            # imgToCropped = cv2.rectangle(img=facial_part, pt1=(x, y), pt2=(x + width, y + height), color=(0, 255, 0), thickness=1)
            facial_part_cropped = imgToCropped[y: y + height, x: x+width]
            # As cropped image will be very small, lets resize it a bit bigger..
            facial_part_cropped = cv2.resize(src=facial_part_cropped, dsize=(0, 0), fx=scaleFactor, fy=scaleFactor)
            # return facial_part_cropped --- loop again

    # Final Facial parts mask
    return facial_parts_mask


def change_facial_part_color(mask_crop_img, actual_image):
    """
    This function helps in changing the facial_part color by help of track_bars.
    :param mask_crop_img: masked and cropped image which is performed earlier than this step, if different one ambiguous results..
    :return: The colored facial part image
    """
    # Create the track bars..
    utils.createTrackbars()
    # Create the mask image of size as equal as the passed image(mask_crop_image)
    mask_image = np.zeros_like(mask_crop_img)
    while True:
        # fill the entire mask with the color selected by track bars (Basically, entire image will change to that desired color..)
        mask_image[:] = utils.adjust_color_by_tackbars()
        # Now perform the bitwise_and, so that only the part which is white will get the color set and rest gets masked..(Thats's how bitwise_and works right..!!)
        color_mask_crop_img = cv2.bitwise_and(mask_crop_img, mask_image)
        # Before merging both the original and mask image, make it blur a bit as it doesn't look good if edges are sharp..
        color_mask_crop_img = cv2.GaussianBlur(src=color_mask_crop_img, ksize=(9, 9), sigmaX=10)   # Playable parameters,, NOTE, ksize's values must be odd!!!

        # Now merge the original image(as it is, by passing alpha=1) and the masked one with a alpha (based on its weight, by passing beta=<adjustable_values>)
        color_mask_crop_img = cv2.addWeighted(actual_image, 1, color_mask_crop_img, 0.4, 0)
        # print("")
        cv2.imshow("Adjust Colors by track bars..", color_mask_crop_img)

        # when pressed escape,return the colors adjusted masked cropped image..
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            return color_mask_crop_img



""" Instances of some dlib classes.."""
# Get the face detector instance from dlib...
faces_detector = dlib.get_frontal_face_detector()
facial_landmarks_predictor = dlib.shape_predictor("Resources/shape_predictor_68_face_landmarks.dat")
facial_landmarks_count = 68  # This is based on the predictor, here we took above for 68, so assigned 68. If another, please assign the respective one...

""" As usual Process"""

# Required variables..
# A list to save all the facial landmark points to help in cropping in specific part of face..
# facial_landmarks_points = []

# Read the image.. to perform the iterations.. uncomment the below 2 lines by setting appropriate values and making changes at image names (And don't forget to indent the code below it..!!:)..)
# for name_count in range(2, 13):
#    for turn_count in range(1, 3):
# A list to save all the facial landmark points to help in cropping in specific part of face..
facial_landmarks_points = []
imageOriginal = cv2.imread("Resources/test_image_5.png")
# lets resize the image, its quite big..
# imageOriginal = cv2.resize(src=imageOriginal, dsize=(0, 0), dst=0.0, fx=0.5, fy=0.5)
img = imageOriginal.copy()
# Convert to grayscale..
grayImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# detect the faces via detector instance
detected_faces = faces_detector(grayImg)
# Drawing a bounding box around all the detected faces..
for face in detected_faces:
    # top-left corner
    x1, y1 = face.left(), face.top()
    # bottom-right corner
    x2, y2 = face.right(), face.bottom()
    # Draw the rectangle
    image = cv2.rectangle(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)

    # Get the facial landmarks..
    facial_landmarks = facial_landmarks_predictor(grayImg, face)
    # Now we need to draw all the landmarks..
    for count in range(facial_landmarks_count):
        # Get the origin of each landmark..
        x = facial_landmarks.part(count).x
        y = facial_landmarks.part(count).y
        # save the landmark points..
        facial_landmarks_points.append((x, y))
        # Draw a circle at that landmark detected..
        cv2.circle(img=img, center=(x, y), radius=3, color=(50, 50, 255), thickness=cv2.FILLED)
        # Lets also number each landmark for better understanding and also helps in cropping specific parts of image..
        cv2.putText(img=img, text=str(count), org=(x , y+10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1)
        # ---------------Observation of the count is below LOG, please refer..
        cv2.imshow("Facial Landmarks points sequence", img)
    # Convert to numpy array.. . as cv2.boundingRect() needs an array not list..
    facial_landmarks_points = np.array(facial_landmarks_points)
    # Extract the desired facial_part's land_mark points from below LOG. NOTE: the upper_bound of the range should be upper_bound+1 as slicing is done as upper_bound, if sent upper_bound+1, then we'll get till upper_bound. right..!! Simply slicing treats range as [lower_bound, upper_bound) (NOTE the usage of type of brackets here... used by keeping the mathematical view)
    leftEye = facial_landmarks_points[36:41 + 1]
    rightEye = facial_landmarks_points[42: 47+1]
    # lips = facial_landmarks_points[48: 61]
    upper_lip = facial_landmarks_points[48:54+1]
    lower_lip = facial_landmarks_points[55: 60+1]
    # extraction_points = np.vstack((leftEye, rightEye))
    extraction_points = [leftEye, rightEye, upper_lip, lower_lip]
    # Perform Extraction..
    # To extract only one feature at a time....................UN COMMENT THIS if would only like to extract only one facial part
    # extracted_facial_part = extract_facial_parts(imgToCropped=imageOriginal.copy(), points=lips, scaleFactor=5, masked=True, cropped=False)  # NOTE: Please atleast pass True either for masked or cropped..!! else gets a black image which will be of size of the image we pass here as a parameter..!!
    # To extract multiple features at a time...................UN COMMENT This if would like to extract multiple facial parts at a time..
    extracted_facial_part = extract_facial_parts_extended(imgToCropped=imageOriginal.copy(), multiple_facial_points=extraction_points, scaleFactor=5, masked=True, cropped=False)  # NOTE: Please atleast pass True either for masked or cropped..!! else gets a black image which will be of size of the image we pass here as a parameter..!!
    cv2.imshow("Extracted facial part", extracted_facial_part)
    # Change the colors of the extracted_facial_part
    # To work on grayImg, convert ot 3 channels from 1 channel else error as "cv2.error: OpenCV(4.3.0) ..\modules\core\src\arithm.cpp:669: error: (-209:Sizes of input arguments do not match) The operation is neither 'array op array' (where arrays have the same size and the same number of channels), nor 'array op scalar', nor 'scalar op array' in function 'cv::arithm_op'"
    grayImg = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2BGR)
    extracted_facial_part_colored = change_facial_part_color(mask_crop_img=extracted_facial_part, actual_image=grayImg)  # To get the result on color image, pass "actual_image=imageOriginal.copy()", for result on gray scale "actual_image=grayImg"

    # Display results

    cv2.imshow("Final Extracted facial Part + colored ", extracted_facial_part_colored)
    if cv2.waitKey(0) == ord('s'):
        cv2.imwrite("Results/test_image_gray_scale_5-left_eye+right_eye+lips.jpg", extracted_facial_part_colored)
        print("Image saved successfully..")
        cv2.destroyAllWindows()
        break



"""
LOG:
    Part-Name       |   Facial_landmark_count (Range - Inclusive both sides)
    ----------------------------------------
    Left Eyes       |   [36 .. 41]
    Right Eye       |   [42 .. 47]
    Nose            |   [27 .. 35]
    Lips            |   [48 .. 67]  take 48 to 61 as rest are the points between the both upper and lower lip
    Left Eyebrow    |   [0 .. 16]
    Right Eyebrow   |   [177 .. 21] 
                
"""
"""
Observation:
    The dlib predicts as in this order.. (currently tested only for image.jpg), expecting it will be same for every face image..
        1. Face borders..Starting from left-eye to chin then to right-eye)......17 land mark points ......range --> [0..16]
        2. Left Eyebrow................Left to right.............................5 landmark points .......range --> [17 .. 21]
        3. Right Eyebrow...............Left to right.............................5 landmark points .......range --> [22 .. 26]
        4. Nose........................top to bottom.............................8 landmark points .......range --> [27 .. 35]   27 to 30 (from upper cartilage to lower cartilage)   31, 32 --> left nostril, 33 mid part, 34, 35 --> right nostril   
        5. Left Eye..........Left to Right then Right to Left....................6 landmark points .......range --> [36 .. 41]   36 to 39 (Right to Left)upper eyelashes, (+39)40 to 41 (+36) (Right to Left) lower eye lashes  # The +39 and +36 are added because, they meet at point those points, without their inclusion detection becomes erroneous. 
        6. Right Eye.........Left to Right then Right to Left....................6 landmark points .......range --> [42 .. 47]   42 to 45 (Right to Left)upper eyelashes, (+45)46 to 47 (+42) (Right to Left) lower eye lashes  # +45 and +42 are added because they are meeting points
        7. Lips..............Left to Right then Right to Left.(Twice)............20 landmark points .......range --> [48 .. 67]
            |+ upper_lip.............................................................6 land mark points ......range --> [48 .. 54]  If would like to detect and change colors properly, at the end of the sequence of this points append the reverse the sequence of upper_lip_bottom_border
            |+ lower_lip.............................................................6 land mark points ......range --> [55 .. 60]  If would like to detect and change colors properly, at the end of the sequence of this points append the reverse order of the sequence of lower_lip_top_border
            |+ upper_lip_lower_lip_in_b/w............................................8 land mark points ......range --> [60 .. 67]
                |+ upper_lip_bottom_border................................................4(+1) land mark points ... range --> [60 .. 64]  Sequence goes like this 60, 61, 62, 63, 64 (From left to right)
                |+ lower_lip_top_border...................................................4(+1) land mark points.....range --> [64 .. 67]  Sequence goes like this 64, 65, 66, 67, 60 (From right to left)
       Total....................................................................68 land mark points
       
       The hard-coded sequence of the 
                upper lip is [48 .. 54, 64 .. 60, 48]
                lower lip is [54 .. 59, 48, 60, 67 .. 60, 54]
            
"""

"""
LOG for playable parameters..
    for the cv2.GaussianBlur():
        found (9, 9) for ksize as good results..
"""