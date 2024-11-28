import argparse
import yaml
from utils import *
import warnings
warnings.filterwarnings("ignore")

def get_parse():
    parser = argparse.ArgumentParser(description='UAV-Visual-Localization')
    parser.add_argument('--yaml', default='config.yaml', type=str, help='yaml dir')
    parser.add_argument('--save_dir', default=r'./Result/Experiment1/', type=str, help='where we save the result')
    parser.add_argument('--device', default='cuda', type=str, help='Inference device')
    parser.add_argument('--pose_priori', default='yp', type=str,
                        help='priori about the pose, yp: yaw and pitch, p: pitch, ''unknown: no pitch and yaw')
    parser.add_argument('--strategy', default='Topn_opt', type=str, help='Inliners;Top1;Topn_opt,wo_retrieval')
    parser.add_argument('--PnP_method', default='P3P', type=str, help='see option.yaml')
    parser.add_argument('--Ref_type', default='HIGH', type=str, help='HIGH(LOW): use aerial(satellite) map')
    parser.add_argument('--resize_ratio', default=0.2, type=float, help='save inference time and memory, '
                                                                        'especially when reference image is too large')

    opt = parser.parse_args()
    print(opt)
    return opt


if __name__ == '__main__':

    # Parameter Initialization
    opt = get_parse()
    with open(opt.yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    All_Region = config['REGIONS']
    All_Retrieval = config['RETRIEVAL_METHODS']
    All_Matching = config['MATCHING_METHODS']
    for region in All_Region:
        yaml_file = f'./Regions_params/{region}.yaml'
        with open(yaml_file, 'r') as f:
            region_config = yaml.safe_load(f)
        region_config.update(config)

        UAV_path = find_values(region_config, 'UAV_PATH')
        json_file = f'./Data/metadata/{region}.json'
        places = find_values(region_config, 'PLACES')
        UAV_img_list0 = []
        # Read all the UAV images, use the 'TEST_INTERVAL' to adjust the number of test samples
        for place in places:
            UAV_img_list0 = UAV_img_list0 + get_jpg_files(UAV_path + place)
        UAV_img_list = UAV_img_list0[0::region_config['TEST_INTERVAL']]
        method_dict = {}
        for retrieval_index in range(len(All_Retrieval)):
            method_dict['retrieval_method'] = All_Retrieval[retrieval_index]
            method_dict = retrieval_init(method_dict, region_config)
            for method_index in range(len(All_Matching)):
                method_dict['matching_method'] = All_Matching[method_index]
                method_dict = matching_init(method_dict)
                # Read the 2D reference map as well as the corresponding DSM data
                ref_map0, dsm_map0, save_path0, ref_resolution = load_config_parameters_new(region_config, opt, region)

                for index, uav_path in enumerate(tqdm(UAV_img_list, desc=f"{region}", unit="image")):
                    place = os.path.basename(os.path.dirname(uav_path))
                    print('Region: {} Place: {} Pic: {} Ratio: {:.1f}%'.format(region, place, os.path.basename(uav_path),
                                                                    index / len(UAV_img_list) * 100))

                    # Generate the path to save the important results
                    VG_pkl_path = '{}/{}/pkl_{}/resize_{}/{}-{}-{}-{}/VG_data_{}.pkl'. \
                        format(opt.save_dir, region, place, opt.resize_ratio, opt.Ref_type, All_Retrieval[retrieval_index],
                               All_Matching[method_index], opt.pose_priori,  img_name(uav_path))
                    if os.path.exists(VG_pkl_path):
                        continue

                    save_path = '{}/{}/{}'.format(save_path0, place, index+1)
                    truePos = query_data_from_file(json_file, name=f'{os.path.dirname(uav_path)}/{os.path.basename(uav_path)}')[0]
                    K = computeCameraMatrix(truePos)
                    uav_image = cv2.imread(uav_path)

                    # Rotate the reference map to align with the UAV image by utilizing the prior yaw angle.
                    ref_map, matRotation = dumpRotateImage(ref_map0, truePos['yaw'])
                    dsm_map, _ = dumpRotateImage(dsm_map0, truePos['yaw'])
                    VG_time0 = time.time()

                    # Visual Localization Step1: Image-level Retrieval
                    IR_order, refLocX, refLocY, PDE_list, cut_H, cut_W, fineScale, retrieval_time = retrieval_all(ref_map, uav_path, truePos,  ref_resolution,  matRotation, save_path, opt, region, region_config, method_dict)

                    # Visual Localization Step2&3: Pixel-level Matching & PnP Problem Solving
                    BLH_list, inliners_list, match_time, pnp_time = Match2Pos_all(opt, region, region_config, uav_image, fineScale, K, ref_map, dsm_map, refLocY, refLocX, cut_H, cut_W,
                                                                                save_path, method_dict, matRotation)
                    # Calculate the visual localization error
                    pred_loc, pred_error, location_error_list = pos2error(truePos, BLH_list, inliners_list)
                    print('pred_error: {}'.format(pred_error) )

                    # Save data to pkl file for subsequent analysis
                    VG_time_cost  = time.time() - VG_time0
                    save_data(VG_pkl_path, opt = opt, region_config = region_config, img_path=uav_path, truePos=truePos, refLocX= refLocX, refLocY= refLocY,IR_order = IR_order, PDE=PDE_list, inliners =inliners_list, BLH_list = BLH_list,
                              location_error_list= location_error_list,  pred_loc=pred_loc, pred_error = pred_error, retrieval_time = retrieval_time, match_time = match_time, pnp_time = pnp_time, total_time = VG_time_cost)