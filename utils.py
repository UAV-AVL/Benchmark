from Matching_Models.RoMa.demo.Roma_match import Roma_Init, Roma_match
from pyproj import Transformer
import shutil
import glob
import pickle
from matplotlib import pyplot as plt
from Retrieval_Models.multi_model_loader import get_Model
import math
import json
from Retrieval_Models.feature_extract import *
import os
import time
import cv2
from math import sqrt
import numpy as np
from PIL import Image


def img_name(sensing_path):
    file_name_with_ext = os.path.basename(sensing_path)
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    return file_name_without_ext


def load_data(save_path, *args):
    with open(save_path, 'rb') as file:
        data = pickle.load(file)
    results = tuple(data[arg] for arg in args if arg in data)

    return results

def save_data(filename, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = kwargs
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def get_jpg_files(folder_path):
    # Ensure the input path exists
    if not os.path.exists(folder_path):
        print("Path does not exist. Please check your input!")
        return []

    # If the path is a file rather than a folder, return an empty list
    if not os.path.isdir(folder_path):
        print("The input is a file path, not a folder path!")
        return []

    # Use the glob module to search for all JPG files, including uppercase and lowercase extensions
    jpg_files = glob.glob(os.path.join(folder_path, '*.JPG'))

    return jpg_files


def copy_image(src_path, dest_path, new_name):
    dest_file_path = os.path.join(dest_path, new_name)
    shutil.copy2(src_path, dest_file_path)


def view_center(region, config, truePos, initialX, initialY, ref_resolution, reverseMatRotation):
    '''This function estimates the location of the center of drone image (the center of view)'''
    pitch = -truePos['pitch'] / 180 * np.pi
    yaw = -truePos['yaw'] / 180 * np.pi
    utm_x, utm_y = deg2utm(region, config, truePos['lon'], truePos['lat'])
    delta_y = truePos['rel_alt'] / np.tan(pitch) * np.cos(yaw)
    delta_x = -truePos['rel_alt'] / np.tan(pitch) * np.sin(yaw)
    UTM_Y_c = utm_y + delta_y
    UTM_X_c = utm_x + delta_x
    x_center0 = int((UTM_X_c - initialX) / ref_resolution)
    y_center0 = int((initialY - UTM_Y_c) / ref_resolution)
    refCoordinate=np.array([x_center0, y_center0, 1])
    refCoordinate2 = refCoordinate @ reverseMatRotation.T
    x_center, y_center = refCoordinate2[0], refCoordinate2[1]
    return x_center, y_center

def Match2Pos_all(opt, region, config, uav_img0, finescale, K, ref_image, dsm_image, refLocY_list, refLocX_list, cut_H,
              cut_W, save_path, method_dict, matRotation):

    '''This function first matches drone image and the reference images 
    and then use the dsm data to solve pnp problem to get the drone position.
    opt and config: options and configurations for the experiment
    region(str): the region of current test data
    finescale(float): drone_resolution / ref_resolution
    K(ndarray): the camera intrinsic matrix using to solve pnp problem
    ref_image(ndarray): the 2D reference map used for AVL
    dsm_image(ndarray): the dsm map used for AVL
    refLocY_list(list) and refLocX_list(list): the position of image retrieval results
    cut_H,cut_W(int): the reference(gallery) image size
    save_path(str): path where we save the visualization result
    method_dict(dict): the information about image retrieval approaches and the model parameters
    matRotation(matrix): the rotation matrix of reference map since we use the prior yaw angle to align the
                        drone image and reference map to similar rotation'''

    if method_dict['matching_method'] == 'Roma':
        Roma_model = method_dict['matching_model']
    else:
        print('Method name is wrong!!!')

    if opt.pose_priori == 'yp':
        reverseMatRotation = cv2.invertAffineTransform(matRotation)
    else:
        reverseMatRotation = None
    resize_ratio = opt.resize_ratio

    Ref_type = opt.Ref_type
    initialX = config[f'{region}_{Ref_type}_REF_initialX']
    initialY = config[f'{region}_{Ref_type}_REF_initialY']
    ref_resolution = config[f'{region}_{Ref_type}_REF_resolution']
    dsm_resolution = config[f'{region}_{Ref_type}_DSM_resolution']
    dsm_coor = config[f'{region}_{Ref_type}_DSM_COORDINATE']
    ref_coor = config[f'{region}_{Ref_type}_REF_COORDINATE']
    dsm_ratio = dsm_resolution / ref_resolution
    dsm_coor = [dsm_coor[0] * dsm_ratio, dsm_coor[1] * dsm_ratio]
    dsm_offset = (int(dsm_coor[0] - ref_coor[0]), int(dsm_coor[1] - ref_coor[1]))


    if resize_ratio<1:
        uav_img = cv2.resize(uav_img0, None, fx=resize_ratio, fy=resize_ratio)
    else:
        uav_img = uav_img0
    BLH_list = []

    inliers_list = []
    time_list = []
    # Extract fine regions from reference image and DSM image
    np.random.seed(0)
    if not isinstance(refLocY_list, list):
        refLocX_list = [refLocX_list]
        refLocY_list = [refLocY_list]

    if opt.strategy == 'Top1':
        top_n = 1
    elif opt.strategy == 'Topn_opt':
        top_n = min(method_dict['retrieval_topn'], len(refLocY_list))
    else:
        top_n = len(refLocY_list)

    for index in range(top_n):
        print(index/len(refLocY_list))
        ransac_name = '/top{}_ransac.png'.format(index+1)
        refLocY = refLocY_list[index]
        refLocX = refLocX_list[index]
        fineRef = ref_image[refLocX:refLocX+cut_W, refLocY:refLocY+cut_H ]
        fineRef = cv2.resize(fineRef, None, fx=resize_ratio/finescale, fy=resize_ratio/finescale)

        match_time_start = time.time()

        if method_dict['matching_method'] == 'Roma':
            Sen_pts, Ref_pts = Roma_match(uav_img, fineRef, Roma_model, save_path, ransac_name,
                                               need_ransac=False,
                                               show_matches=False)
        else:
            print('Method name is wrong!!!')

        match_time_end = time.time()
        single_match_time = match_time_end-match_time_start

        if len(Ref_pts)>=5:
            refCoordinate = np.array(Ref_pts)/resize_ratio*finescale + np.array([refLocY, refLocX])
            dsm_x = refCoordinate[:, 1] + dsm_offset[1]
            dsm_y = refCoordinate[:, 0] + dsm_offset[0]
            dsm_x, dsm_y = (dsm_x+1)/dsm_ratio-1, (dsm_y+1)/dsm_ratio-1
            dsm_x1 = dsm_x.astype(int)
            for ii in range(len(dsm_x1)):
                if dsm_x1[ii] >= dsm_image.shape[0]:
                    dsm_x1[ii] = dsm_image.shape[0]-1
            dsm_y1 = dsm_y.astype(int)
            for jj in range(len(dsm_y1)):
                if dsm_y1[jj] >= dsm_image.shape[1]:
                    dsm_y1[jj] = dsm_image.shape[1] - 1
            DSM = dsm_image[dsm_x1, dsm_y1]

            if opt.pose_priori == 'yp':
                refCoordinate1 = np.hstack([refCoordinate, np.ones((refCoordinate.shape[0], 1))])
                refCoordinate2 = refCoordinate1 @ reverseMatRotation.T
                UTM_X = refCoordinate2[:, 0] * ref_resolution + initialX
                UTM_Y = initialY - refCoordinate2[:, 1] * ref_resolution
            else:
                UTM_X = refCoordinate[:, 0] * ref_resolution + initialX
                UTM_Y = initialY - refCoordinate[:, 1] * ref_resolution

            match_points = np.array(Sen_pts)/resize_ratio
            BLH, _, inliers, inliers_all = estimate_drone_pose(region, config, match_points, K, UTM_X, UTM_Y,  DSM)
            if config['SHOW_RETRIEVAL_RESULT']:
                # _, _, inliers, inliers_all = estimate_drone_pose(region, config, match_points, K, UTM_X, UTM_Y, DSM)
                process_and_save_matches(Sen_pts, Ref_pts, inliers_all, uav_img, fineRef, save_path+f'/{index+1}-{inliers}.png')
        else:
            BLH = {'B': None, 'L': None, 'H': None}
            inliers = 0

        PnP_time_end = time.time()
        single_PnP_time = PnP_time_end - match_time_end
        time_list.append([single_match_time, single_PnP_time])
        BLH_list.append(BLH)
        inliers_list.append(inliers)

    match_time = [t[0] for t in time_list]
    pnp_time = [t[1] for t in time_list]
    return BLH_list,  inliers_list, match_time, pnp_time


def process_and_save_matches(Sen_pts, Ref_pts, inliers, uav_image, ref_image, save_path):
    Sen_pts_temp = [Sen_pts[i[0]] for i in inliers]
    Ref_pts_temp = [Ref_pts[i[0]] for i in inliers]

    inlier_keypoints_left1 = [cv2.KeyPoint(point[0], point[1], 1) for point in Sen_pts_temp]
    inlier_keypoints_right1 = [cv2.KeyPoint(point[0], point[1], 1) for point in Ref_pts_temp]

    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(inliers))]
    image3 = cv2.drawMatches(uav_image, inlier_keypoints_left1, ref_image, inlier_keypoints_right1,
                             placeholder_matches,  outImg=None)
    cv2.imwrite(save_path, image3)

def pos2error(truePos, BLH_list, inliers_list):
    '''
    This function calculates the visual localization  error e=((x_true-x_pred)**2+(y_true-y_pred)**2)**0.5
    Note that we only care about the error of longitude and latitude, and do not consider the altitude error in this formula
    The coordinate system is BLH system, and the top n retrievals are re-ranked according to number of pnp ransac inliers
    truePos(dict): the ground truth of drone location
    BLH_list(list): the predicted location of top n images
    inliers_list(list): the inliers number of top n images
    '''
    inliers_array = np.array(inliers_list)
    max_index = np.argmax(inliers_array)
    location_error_list = []
    for BLH_temp in BLH_list:
        if BLH_temp['L'] and BLH_temp['B']:
            error_lat_temp = (BLH_temp['B'] - truePos['lat']) * 110000
            cos_latitude = math.cos(truePos['lat'])
            error_lon_temp = (BLH_temp['L'] - truePos['lon']) * 110000 * cos_latitude
            location_error_list.append(np.sqrt(error_lat_temp ** 2 + error_lon_temp ** 2 ))
        else:
            location_error_list.append(10000) # if there is no location data, the error is set as 10000
    loc_err_inliners = location_error_list[max_index]
    pred_loc = {'lon':BLH_list[max_index]['L'],
                'lat':BLH_list[max_index]['B']}
    return pred_loc, loc_err_inliners, location_error_list


def computeCameraMatrix(truePos):
    imW0 = truePos['width']
    imH0 = truePos['height']
    cameraSize = truePos['cam_size']
    focal = truePos['focal_len']
    # Calculate pixel sizes
    pixelSize_x = cameraSize / np.sqrt(imW0**2 + imH0**2)
    pixelSize_y = cameraSize / np.sqrt(imW0**2 + imH0**2)

    # Calculate focal lengths
    focalx = focal / pixelSize_x
    focaly = focal / pixelSize_y

    # Calculate image centers
    centerX = imW0 / 2
    centerY = imH0 / 2

    # Construct the camera intrinsic matrix K
    K = np.array([
        [focalx, 0, centerX],
        [0, focaly, centerY],
        [0, 0, 1]
    ])

    return K

def estimate_drone_pose(region, config, match_points, K, UTM_X, UTM_Y,  DSM):

    pose_3d = np.column_stack((UTM_X, UTM_Y, DSM))
    dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(pose_3d, match_points, K, dist_coeffs,
                                                     flags=cv2.SOLVEPNP_P3P)
    if inliers is None:
        inliers = [0]
    R1 = rotvector2rot(rvec)
    X0 = -R1.T @ tvec
    lon, lat = utm2deg(region, config, X0[0], X0[1])
    BLH = {'B': float(lat), 'L': float(lon), 'H': float(X0[2])}
    Rx_90 = np.array([[1, 0, 0],
                      [0, np.cos(-np.pi / 2), np.sin(-np.pi / 2)],
                      [0, -np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
    cam_angle = rot_to_euler(Rx_90 @ R1)

    return BLH, cam_angle, len(inliers), inliers

def dumpRotateImage(img, degree):
    radians = degree/180*np.pi
    height, width = img.shape[:2]
    heightNew = int(width * abs(np.sin(radians)) + height * abs(np.cos(radians)))
    widthNew = int(height * abs(np.sin(radians)) + width * abs(np.cos(radians)))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0,2] += (widthNew - width)//2
    matRotation[1,2] += (heightNew - height)//2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation, matRotation

def rotvector2rot(rotvector):
    Rm = cv2.Rodrigues(rotvector)[0]
    return Rm

def utm2deg(region, config, x,y):
    utm_system = config[f'{region}_UTM_SYSTEM']
    transformer = Transformer.from_crs(f"epsg:326{utm_system[:2]}", "epsg:4326")
    lat, lon = transformer.transform(x, y)
    return lon, lat

def deg2utm(region, config, lon,lat):
    '''Convert latitude and longitude to UTM coordinate system (in meters)
    for subsequent localization error calculation.'''
    utm_system = config[f'{region}_UTM_SYSTEM']
    transformer = Transformer.from_crs("epsg:4326", f"epsg:326{utm_system[:2]}")
    x, y = transformer.transform(lat, lon)
    return x, y

def rot_to_euler(dcm, lim=None):
    r11 = -dcm[1][0]
    r12 = dcm[1][1]
    r21 = dcm[1][2]
    r31 = -dcm[0][2]
    r32 = dcm[2][2]
    r11a = dcm[0][1]
    r12a =  dcm[0][0]
    # Find angles for rotations about X, Y, and Z axes
    r1 = np.arctan2(r11, r12)
    r21 = np.clip(r21, -1, 1)  # Clip r21 to ensure it's within valid range for asin
    r2 = np.arcsin(r21)
    r3 = np.arctan2(r31, r32)
    if lim == 'zeror3':
        for i in np.where(np.abs(r21) == 1.0)[0]:
            r1[i] = np.arctan2(r11a[i], r12a[i])
            r3[i] = 0
    camAngle = np.array([-r1 - np.pi, -r2, r3 + np.pi]) * 180 / np.pi;
    return camAngle



def resolution_size(data, opt):
    '''estimate the spatial resolution of drone image'''
    img_width = data['width']
    img_height = data['height']
    cam_size = data['cam_size']
    focal_len = data['focal_len']

    if opt.pose_priori == 'unknown':
        pitch = -30
    else:
        pitch = data['pitch']
    resolution = 2 * data['rel_alt'] / np.sin(-np.pi * pitch / 180) * cam_size/2/focal_len \
                 / sqrt(img_width ** 2 + img_height ** 2)

    min_size = min(img_width, img_height) # Using square image as query image
    size = np.array([min_size, min_size])
    return resolution, size


def crop_center(image_path, width, height):
    # Read the image
    img = cv2.imread(image_path)

    # Get the original image dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the center point of the cropping area
    center_x, center_y = original_width // 2, original_height // 2

    # Calculate the top-left and bottom-right coordinates of the cropping area
    left = int(center_x - width // 2)
    top = int(center_y - height // 2)
    right = int(center_x + width // 2)
    bottom = int(center_y + height // 2)

    # Ensure the cropping area does not exceed the image boundaries
    left = max(0, left)
    top = max(0, top)
    right = min(original_width, right)
    bottom = min(original_height, bottom)

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    return Image.fromarray(cropped_img)

def find_values(config, search_string):
    results = 'None'

    for key, value in config.items():
        if  search_string in key:
            results = value
    return results


def load_config_parameters_new(config, opt, region):

    Ref_type = opt.Ref_type
    # Read images
    ref_image = cv2.imread(config[f'{region}_{Ref_type}_REF_PATH'])
    ref_image = ref_image[:,:,0:3]
    ref_image = ref_image.astype(np.uint8)
    dsm_image = cv2.imread(config[f'{region}_{Ref_type}_DSM_PATH'], cv2.IMREAD_UNCHANGED).astype(np.float32)

    # Read parameters
    ref_resolution = config[f'{region}_{Ref_type}_REF_resolution']
    save_path = opt.save_dir
    print('Result saved in : {}'.format(save_path))
    return ref_image, dsm_image, save_path, ref_resolution

def query_data_from_file(file_name, **query):
    '''Query the pre-saved metadata based on the drone image name.'''
    data = read_data_from_file(file_name)
    results = []

    for entry in data:
        if all(entry.get(key) == value for key, value in query.items()):
            results.append(entry)

    return results

def read_data_from_file(file_name):
    """Read UAV data from a file"""
    if not os.path.exists(file_name):
        print("File does not exist!")
        return []
    with open(file_name, 'r') as f:
        data = json.load(f)

    return data

def matching_init(method_dict):
    ''' This function is used to initialize the image matching model.
    If you need to test your own image matching method,
    please add the xxx_Init() function here.
    method_dict(dict): contains the information of the methods applied in the experiment'''

    if method_dict['matching_method'] == 'Roma':
        method_dict['matching_model'] = Roma_Init()
    else:
        print('Method name is wrong!!!')
    return  method_dict

def retrieval_init(method_dict, config):
    '''This function is used to initialize the image retrieval model and some relevant parameters
    If you want to test your own image retrieval method, please add the Initialization Function to get_Model().
    method_dict(dict): contains the information of the methods applied in the experiment'''

    if method_dict['retrieval_method'] not in config['RETRIEVAL_METHODS']:
        print('** The input model name is invalid. Please ensure you enter the correct model name. **')
        exit(1)
    else:
        method_dict['retrieval_model'], method_dict['img_transform'] = get_Model(method_dict['retrieval_method'])
        method_dict['retrieval_model'].to(config['DEVICE'])
        # Overlap rate between adjacent reference image blocks, used to calculate the moving step size.
        method_dict['retrieval_cover'] = config['RETRIEVAL_COVER']
        method_dict['retrieval_topn'] = config['RETRIEVAL_TOPN']
        method_dict['retrieval_img_name'] = config['RETRIEVAL_IMG']
        method_dict['retrieval_feat_norm'] = config['RETRIEVAL_FEATURE_NORM']
    return method_dict

def retrieval_all(ref_image, UAV_path, uav_data, ref_resolution, matRotation, save_path, opt, region, config, method_dict):
    '''The key function for image retrieval, which mainly includes parameter initialization,
    gallery image generation, image feature extraction, similarity measurement,
    retrieval metric calculation, and result visualization.
    ref_image(ndarray): the 2D reference map used for AVL
    UAV_path(str): the path of the drone image
    uav_data(dict): the flight metadata including the ground truth of drone location
                    as well as other imaging parameters
    ref_resolution(float): the spatial resolution of the 2D reference map
    matRotation(matrix): the rotation matrix of reference map since we use the prior yaw angle to align the
                        drone image and reference map to similar rotation
    save_path(str): path where we save the visualization result
    opt and config: options and configurations for the experiment
    region(str): the region of current test data
    method_dict(dict): the information about image retrieval approaches and the model parameters'''

    # Parameter Initialization
    retrieval_size = 384
    Ref_type = opt.Ref_type
    initialX = config[f'{region}_{Ref_type}_REF_initialX']
    initialY = config[f'{region}_{Ref_type}_REF_initialY']

    # estimate the geo-location of the center of drone image
    center_x, center_y = view_center(region, config, uav_data, initialX, initialY, ref_resolution, matRotation)

    # calculate the query image size as well as the sampling interval
    cover = method_dict['retrieval_cover']
    drone_resolution, drone_size = resolution_size(uav_data, opt)
    finescale = drone_resolution / ref_resolution
    view_size = drone_size*drone_resolution
    block_size = [math.ceil(view_size[0] / ref_resolution) + (math.ceil(view_size[0] / ref_resolution) % 2 != 0),
                  math.ceil(view_size[1] / ref_resolution) + (math.ceil(view_size[1] / ref_resolution) % 2 != 0)]
    step_size = [int(block_size[0] * (100 - cover) / 100), int(block_size[1] * (100 - cover)/100)]

    # Filters out gallery images with too many invalid areas (black)
    MID = compute_block_mid_wo_black(ref_image, block_size, step_size)
    mids = MID.reshape(-1, 2)

    # crop the square image from the rectangular drone image to serve as the query image
    UAV_image = crop_center(UAV_path, drone_size[0], drone_size[1])

    # extracting the feature of drone image and reference tiles
    retrieval_model = method_dict['retrieval_model']
    img_transform = method_dict['img_transform']
    retrieval_t0 = time.time()
    UAV_image = UAV_image.resize((retrieval_size, retrieval_size))
    UAV_img = img_transform(UAV_image)
    gf, qf = extract_features(retrieval_size, config, method_dict, MID, block_size, ref_image, UAV_img, img_transform, retrieval_model)

    retrieval_t1 = time.time()

    # ranking the gallery images(reference tiles) according to the similarity
    score = gf @ qf.unsqueeze(-1)
    score = score.squeeze().cpu().numpy()
    order = np.argsort(score)  # from small to large
    order = order[::-1]

    # the time used to extract feature from a single image
    retrieval_time_cost = (retrieval_t1 - retrieval_t0)/(len(MID)+1)

    # calculate the retrieval error which is then used for the PDM@K metric
    best_start_x = []
    best_start_y = []
    PDE_list = []
    for i in range(len(order)):
        mid_x, mid_y = mids[order[i], :]
        best_start_x.append(max(0, int(mid_x - block_size[0] / 2)))
        best_start_y.append(max(0, int(mid_y - block_size[1] / 2)))
        d_i = ((center_x - mid_y) ** 2 + (center_y - mid_x) ** 2) ** 0.5
        p_i = d_i / block_size[0]
        PDE_list.append(p_i)

    # Draw the image retrieval result (Top N)
    show_n = config['RETRIEVAL_IMG_NUM']
    if config['SHOW_RETRIEVAL_RESULT']:
        retrieval_img_name = method_dict['retrieval_img_name']
        fig = plt.figure(figsize=(16, 4))
        ax0 = plt.subplot(1, show_n + 1, 1)
        ax0.axis('off')
        UAV_image_1 = np.array(UAV_image)
        plt.imshow(UAV_image_1[..., ::-1])
        plt.title('UAV')
        for i in range(min(len(mids), show_n)):
            ax = plt.subplot(1, show_n + 1, i + 2)
            ax.axis('off')
            mid_x, mid_y = mids[order[i], :]
            img = ref_image[int(mid_x - block_size[0] / 2):int(mid_x + block_size[0] / 2),
                  int(mid_y - block_size[1] / 2):int(mid_y + block_size[1] / 2)]
            if i == 0:
                middle_position = (ax0.get_position().xmax + ax.get_position().xmin) / 2
                fig.add_artist(
                    plt.Line2D([middle_position, middle_position], [0.25, 0.75], color='#CD2626',
                               linestyle='dashed'))
            plt.imshow(img[..., ::-1])
            ax.set_title(f'{i + 1} PDE:{PDE_list[i]:.3f}', color='#1C86EE', fontsize=20)

        plt.ioff()
        plt.savefig(save_path + retrieval_img_name)
        plt.close(fig)

    return order, best_start_x, best_start_y, PDE_list, block_size[1], block_size[0], finescale, retrieval_time_cost


def compute_block_mid_wo_black(image, block_size, step_size):
    '''
    Filters out gallery images with too many invalid areas (black) by finding blocks of the image that meet certain criteria.

    Parameters:
    image (np.ndarray): Input image array.
    block_size (tuple): Size of the image block (height, width).
    step_size (tuple): Step size for sliding window (height, width).

    Returns:
    np.ndarray: Array of center coordinates of blocks that meet the criteria.
    '''

    x_size, y_size, _ = image.shape
    # resize the origin image to improve efficiency
    small_img = cv2.resize(image, (int(y_size / 10), int(x_size / 10)), interpolation=cv2.INTER_NEAREST)

    num_blocks_x = len(range(0, x_size, step_size[0]))
    num_blocks_y = len(range(0, y_size, step_size[1]))
    mids = []
    total_pixels = block_size[0] * block_size[1] / 100
    for i in range(num_blocks_x):
        for j in range(num_blocks_y):
            # Calculate the starting coordinates of the current block
            start_x = i * step_size[0]
            start_y = j * step_size[1]
            start_x = min(start_x, x_size - block_size[0] - 1)
            start_y = min(start_y, y_size - block_size[1] - 1)
            # Calculate the ending coordinates of the current block
            end_x = start_x + block_size[0]
            end_y = start_y + block_size[1]
            block = small_img[int(start_x / 10):int(end_x / 10), int(start_y / 10):int(end_y / 10)]
            # Calculate the center coordinates of the block
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            count_above_threshold = np.sum(block[:, :, 0] > 0)
            # Check if the number of pixels above the threshold is at least 2/5 of the total pixels
            if count_above_threshold >= total_pixels / 5 * 2:
                mids.append([mid_x, mid_y])
    return np.array(mids)
