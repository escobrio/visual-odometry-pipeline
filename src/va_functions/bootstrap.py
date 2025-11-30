import numpy as np


def initialize_visual_odometry(frames: list, K: np.ndarray) -> dict:
    '''Initialize visual odometry using certain amount of frames, the frame indices are given in `frames`.
    Inputs:
        frames: list of images, to be used for initialization
        K: np.ndarray, camera intrinsic matrix
    Outputs:
        return_dictionary: dict, containing the following keys:
            'R_Wi': np.ndarray, rotation matrix from world to last frame
            'W_t_Wi': np.ndarray, translation vector from world to last frame
            'matched_keypoints': list of np.ndarray, matched keypoints between the first and last frame
            'W_landmarks_of_keypoints': list of np.ndarray, 3D landmarks corresponding to the matched keypoints
    '''
    
    R_Wi = np.eye(3)
    W_t_Wi = np.zeros((3, 1))
    matched_keypoints = []
    W_landmarks_of_keypoints = []
    
    #TODO
    
    
    return_dictionary = {
        'R_Wi': R_Wi,
        'W_t_Wi': W_t_Wi,
        'matched_keypoints': matched_keypoints,
        'W_landmarks_of_keypoints': W_landmarks_of_keypoints
    }
    
    return return_dictionary

def select_keypoint_correspondence(image1, image2) -> tuple:
    '''Select keypoint correspondences between two images.
    Inputs:
        image1: np.ndarray, first image
        image2: np.ndarray, second image
    Outputs:
        keypoints1: np.ndarray, keypoints in the first image
        keypoints2: np.ndarray, corresponding keypoints in the second image
    '''
    
    keypoints1 = np.array([])
    keypoints2 = np.array([])
    
    #TODO: Vermutlich mit KLT aus opencv, aber es muss noch genau geschaut werden was man da alles verwenden darf, denn OpenCV funktion ist glaube isch schon weiter als unsere implementation.
    
    
    return keypoints1, keypoints2

def calculate_final_relative_R_t( ... ) -> tuple: #TODO: add inputs
                                
    '''#TODO: Funktionbeschriebung
    blabla
    '''
    
    R_Wi = np.eye(3)
    W_t_Wi = np.zeros((3, 1))
    matched_keypoints = []
    W_landmarks_of_keypoints = []
    
    #TODO
    
    return R_Wi, W_t_Wi, matched_keypoints, W_landmarks_of_keypoints