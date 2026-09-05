import numpy as np
import cv2


def pose_WC_to_projection(T_WC, K):
    """Convert world->camera pose T_WC (4x4) to projection matrix P (3x4)."""
    T_CW = np.linalg.inv(T_WC) # camera-from-world
    R = T_CW[:3, :3]
    t = T_CW[:3, 3:4] # 3x1
    Rt = np.hstack([R, t]) # 3x4
    return K @ Rt # 3x4

def triangulate_new_landmarks(keypoints_prev, T_prev, keypoints_curr, T_curr, K):
    '''Triangulate new landmarks using the current and previous frame.
    Input:
        keypoints_prev: n x 2D keypoints in previous frame
        T_prev: 4x4 camera pose at previous frame

        keypoints_curr: n x 2D keypoints in current frame
        T_curr: 4x4 camera pose at current frame
        K: Camera intrinsic matrix
    Output:
        points_3d: n x 3D landmarks corresponding to the triangulated keypoints
        valid_mask: Boolean mask indicating which landmarks are valid (positive depth in both cameras)
    '''
    # TODO for now via for loop, later vectorize manual 
    # (np.triangulatePoints does not support batch processing for different projection matrices)

    N = keypoints_prev.shape[0]
    P_curr = pose_WC_to_projection(T_curr, K)

    points_3d_world = np.zeros((N, 3))
    valid_mask = np.zeros(N, dtype=bool)

    for i in range(N):
        P_prev = pose_WC_to_projection(T_prev[i], K)
        points_hom = cv2.triangulatePoints(P_prev, P_curr, keypoints_prev[i].reshape(2, 1), keypoints_curr[i].reshape(2, 1))
        point_3d_world = cv2.convertPointsFromHomogeneous(points_hom.T).reshape(-1, 3)[0]
        
        # Check: Points have to lie infront of the camera 
        T_CW_prev = np.linalg.inv(T_prev[i])
        point_in_prev_cam = (T_CW_prev[:3, :3] @ point_3d_world + T_CW_prev[:3, 3])
        
        T_CW_curr = np.linalg.inv(T_curr)
        point_in_curr_cam = (T_CW_curr[:3, :3] @ point_3d_world + T_CW_curr[:3, 3])
        
        if point_in_prev_cam[2] > 0 and point_in_curr_cam[2] > 0:
            points_3d_world[i] = point_3d_world
            valid_mask[i] = True
        else:
            points_3d_world[i] = [0, 0, 0]

    return points_3d_world, valid_mask