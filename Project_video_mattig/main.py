import time
import json
from stabilization import stabilize
from background_substraction import background_substraction
from matting import matting
from tracking import tracking


ID1 = '207829144'
ID2 = '315129551'
input_video_name = 'Input/INPUT.mp4'
background = 'Input/background.jpg'
stabilized_video_name = f"Outputs/stabilized_{ID1}_{ID2}.avi"
extracted_video_name = f"Outputs/extracted_{ID1}_{ID2}.avi"
binary_mask_video_name = f"Outputs/binary_{ID1}_{ID2}.avi"
matted_video_name = f"Outputs/matted_{ID1}_{ID2}.avi"
alpha_video_name = f"Outputs/alpha_{ID1}_{ID2}.avi"
output_video_name = f"Outputs/OUTPUT_{ID1}_{ID2}.avi"
timing_json_name = "Outputs/timing.json"
tracking_json_name = "Outputs/tracking.json"


#############################################################
# Main
#############################################################
def main():
    start_time = time.time()
    timing_d = {
        "time_to_stabilize": 0,
        "time_to_binary": 0,
        "time_to_alpha": 0,
        "time_to_matted": 0,
        "time_to_output": 0,
    }
    
    # Stabillization
    stabilize(input_video_name, stabilized_video_name)
    timing_d["time_to_stabilize"] = time.time() - start_time

    # BG Substruction
    background_substraction(stabilized_video_name, binary_mask_video_name, extracted_video_name)
    timing_d["time_to_binary"] = time.time() - start_time

    # Alpha and Matting
    matting(stabilized_video_name, binary_mask_video_name, background, alpha_video_name, matted_video_name)
    timing_d["time_to_alpha"] = time.time() - start_time
    timing_d["time_to_matted"] = time.time() - start_time

    # Tracking
    tracking_d = tracking(matted_video_name, output_video_name, alpha_video_name)
    timing_d["time_to_output"] = time.time() - start_time

    with open(timing_json_name, 'w') as f:
        json.dump(timing_d, f, indent=4)

    with open(tracking_json_name, 'w') as f:
        json.dump(tracking_d, f, indent=4)


if __name__ == '__main__':
    main()