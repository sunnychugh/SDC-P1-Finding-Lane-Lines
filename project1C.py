#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os


#reading in an image
image = mpimg.imread('c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\test_images\\whiteCarLaneSwitch.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)



def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #plt.imshow(img); plt.show()
    #plt.imshow(hsv); plt.show()
    white = cv2.inRange(hsv, (0, 0, 180), (255, 25, 255)) 
    #plt.imshow(white); plt.show()
    yellow = cv2.inRange(hsv, (20, 80, 80), (50, 255, 255))
    #plt.imshow(yellow); plt.show()
    gray = cv2.bitwise_or(yellow, white)
    #plt.imshow(gray); plt.show()
        
    return gray
    #return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    #print(masked_image)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
    leftLines=[]
    rightLines=[]
    
    #print(lines.shape)
    #print(img.shape)
    for line in lines:
        for x1,y1,x2,y2 in line:
            #print(x1, y1, x2, y2, ((y2-y1)/(x2-x1)))
            slope=((y2-y1)/(x2-x1))
            #print(slope)
            cordSlope=[x1,y1,x2,y2,((y2-y1)/(x2-x1))]
            #avoiding the cases where slope is inf or zero
            if (np.isinf(slope)):
                continue
            #making slopes range more limited to avoid jitters
            if (slope < -0.45 and slope > -0.85): 
                leftLines.append(cordSlope)                 
            elif (slope > 0.50 and slope < 0.85):
                rightLines.append(cordSlope)
                    

                    
    leftLinesNP = np.asarray(leftLines)
    rightLinesNP = np.asarray(rightLines)
    leftLine_avgX = ((np.mean(leftLinesNP,axis=0))[0] + (np.mean(leftLinesNP,axis=0))[2])/2
    rightLine_avgX = ((np.mean(rightLinesNP,axis=0))[0] + (np.mean(rightLinesNP,axis=0))[2])/2
    #print(leftLine_avgX, rightLine_avgX) 
    leftLine_avgY = ((np.mean(leftLinesNP,axis=0))[1] + (np.mean(leftLinesNP,axis=0))[3])/2
    rightLine_avgY = ((np.mean(rightLinesNP,axis=0))[1] + (np.mean(rightLinesNP,axis=0))[3])/2
    #print(leftLine_avgY, rightLine_avgY)
    leftLine_avgSlope = np.mean(leftLinesNP[:,4],axis=0)
    rightLine_avgSlope = np.mean(rightLinesNP[:,4],axis=0)
    #print(leftLine_avgSlope, rightLine_avgSlope)
    leftLine_intercept = leftLine_avgY - (leftLine_avgSlope*leftLine_avgX)
    rightLine_intercept = rightLine_avgY - (rightLine_avgSlope*rightLine_avgX)
    # setting y1,y2 for image size (540,960)
    if (img.shape[0] == 540 and img.shape[1] == 960):
        leftLine_y1 = img.shape[0]
        leftLine_y2 = 320
        rightLine_y1 = img.shape[0]
        rightLine_y2 = 320
    # setting y1,y2 for image size (720,1280)
    if (img.shape[0] == 720 and img.shape[1] == 1280):
        leftLine_y1 = img.shape[0]
        leftLine_y2 = 450
        rightLine_y1 = img.shape[0]
        rightLine_y2 = 450
        
    #print(leftLine_y1)
    leftLine_x1 = int((leftLine_y1 - leftLine_intercept)/leftLine_avgSlope)
    #print(leftLine_x1)
    leftLine_x2 = int((leftLine_y2 - leftLine_intercept)/leftLine_avgSlope)
    #print(leftLine_x2)
    #print(leftLine_x1, leftLine_y1, leftLine_x2, leftLine_y2)
    cv2.line(img, (leftLine_x1,leftLine_y1), (leftLine_x2, leftLine_y2), color, thickness)
    
    #print(rightLine_y1)
    rightLine_x1 = int((rightLine_y1 - rightLine_intercept)/rightLine_avgSlope)
    #print(rightLine_x1)
    rightLine_x2 = int((rightLine_y2 - rightLine_intercept)/rightLine_avgSlope)
    #print(rightLine_x2)
    #print(rightLine_x1, rightLine_y1, rightLine_x2, rightLine_y2)
    cv2.line(img, (rightLine_x1,rightLine_y1), (rightLine_x2, rightLine_y2), color, thickness)
    
    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print(lines.shape)
    #print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #print(line_img.shape)
    #plt.imshow(line_img);  plt.show()
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
    
    
def find_lane_lines(image):    
    grayImage = grayscale(image) 
    #print(grayImage.shape)  
    #plt.imshow(grayImage);  plt.show()
    blurGray = gaussian_blur(grayImage, 5)
    #plt.imshow(blurGray);  plt.show()
    edges = canny(blurGray, 50, 150)
    #plt.imshow(edges);  plt.show()
    #plt.figure()
    #plt.imshow(edges,cmap='Greys_r');  plt.show()
    imshape = image.shape

    
    # setting vertices for image size (540,960)
    if (image.shape[0] == 540 and image.shape[1] == 960):
        vertices = np.array([[(0,imshape[0]),(450, 320), (510, 320), (imshape[1],imshape[0])]], dtype=np.int32)
    # setting vertices for image size (720,1280)
    if (image.shape[0] == 720 and image.shape[1] == 1280):
        vertices = np.array([[(0,imshape[0]),(600, 450), (800, 450), (imshape[1],imshape[0])]], dtype=np.int32)


    masked_image = region_of_interest(edges, vertices)
    #print(masked_image.shape)
    #plt.imshow(masked_image);  plt.show()
    line_img = hough_lines(masked_image, 1, np.pi/180, 20, 20, 20)
    #plt.imshow(line_img);  plt.show()
    lines_edges_image = weighted_img(line_img, image)
    #plt.imshow(lines_edges_image);  plt.show()
    return lines_edges_image
 
lines_edges_image = find_lane_lines(image)
#plt.imshow(lines_edges_image);  plt.show()    
mpimg.imsave("C:\\Users\\murli-dell\\OneDrive\\spyder\\lines_edges.png", lines_edges_image)
#print(lines_edges_image.shape)

print(os.listdir())
print(os.listdir("c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\test_images"))
directoryFolder = os.listdir("c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\test_images")
print(directoryFolder)
path = "c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\test_images\\"
print(os.listdir(path))

for file in directoryFolder:
    #print(file)
    file = path + file
    #print(file)
    image = mpimg.imread(file)
    #plt.imshow(image);    plt.show();
    imageName = os.path.splitext(file)[0]
    #print(imageName)
    imageName=imageName + "Copy.jpg"
    #print(imageName)
    lines_edges_image = find_lane_lines(image)
    #plt.imshow(lines_edges_image);    plt.show();
    mpimg.imsave(imageName, lines_edges_image)
    
    
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    
    
    resultFrame = find_lane_lines(image)
    #plt.imshow(resultFrame);  plt.show()
      
    return resultFrame

    
white_output = 'c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\white.mp4'
clip1 = VideoFileClip("c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
    
#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(white_output))

yellow_output = 'c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\yellow.mp4'
clip2 = VideoFileClip('c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(yellow_output))

challenge_output = 'c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\extra.mp4'
clip2 = VideoFileClip('c:\\users\\murli-dell\\oneDrive\\spyder\\CarND-LaneLines-P1-master\\challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)

#HTML("""
#<video width="960" height="540" controls>
#  <source src="{0}">
#</video>
#""".format(challenge_output))