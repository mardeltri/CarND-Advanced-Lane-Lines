import matplotlib.pyplot as plt
# import cv2
from calibration import CalibrateCamera
from param import CHESSBOARD_SHAPE
from threshold import Thresholds
from transform import PerspectiveTransform
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d.proj3d import persp_transformation
from identify_lines import fit_polynomial, draw_fill, add_radius2img
from line import Line
from moviepy.editor import VideoFileClip

IMAGE = 0

class main_P2():
    def __init__(self, calibration_path):
        self._cam_calibration = CalibrateCamera(calibration_path, CHESSBOARD_SHAPE)
        self._left_line = Line()
        self._right_line = Line()
        
    def process(self,img):
        self._undistorted = self._cam_calibration.undistort(img)
        self._threshold = Thresholds(self._undistorted)
        self._thresholded = self._threshold.pipeline()
        self._perspective_transformer = PerspectiveTransform()
        warped = self._perspective_transformer.warp(self._thresholded)
        out_img = fit_polynomial(warped, self._left_line, self._right_line)
        result = draw_fill(warped,self._left_line,self._right_line,self._perspective_transformer, self._undistorted)
        result = add_radius2img(result, self._left_line, self._right_line)
        
    
        return result
    
if __name__ == '__main__':
    P2 = main_P2('..\\camera_cal\\')
    if IMAGE:
        image = mpimg.imread('..\\test_images\\test4.jpg')
        result = P2.process(image)
        # Visualize undistortion
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(result)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     plt.savefig('output_images/undistort_output_c'+str(idx+1)+'.png')
        plt.show()
    else:
        video = VideoFileClip("..\\project_video.mp4")
        output_video = video.fl_image(P2.process)
        output_video.write_videofile("..\\output_videos\\project_video_output.mp4", audio=False)