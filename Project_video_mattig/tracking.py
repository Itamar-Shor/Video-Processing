import cv2
from tqdm import tqdm
from collections import OrderedDict
import utils


def tracking(input_video_path, output_video_path, alpha_video_path):
    """
    We already did all the hard work of extracting the object (=foreground) from the video when creating the alpha.
    [1]: find boundaries of the foreground in each alpha frame (max/min coordinates of white pixels).
    """
    capture_input = cv2.VideoCapture(input_video_path)
    capture_alpha = cv2.VideoCapture(alpha_video_path)
    parameters = utils.get_video_parameters(capture_input)
    output = cv2.VideoWriter(output_video_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             parameters['fps'],
                             (parameters['width'], parameters['height']),
                             isColor=True)
    tracking_dict = OrderedDict()
    total_frames = int(capture_input.get(cv2.CAP_PROP_FRAME_COUNT))
    for fno in tqdm(range(total_frames), desc="Tracking"):
        ret1, frame = capture_input.read()
        ret2, alpha = capture_alpha.read()
        if ret1 or ret2:
            # Generate variables
            gray = cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
            # threshold
            thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
            x, y, w, h = cv2.boundingRect(thresh)
            tracking_dict[f"{fno}"] = [y + h // 2, x + w // 2, h // 2, w // 2]  # write [center_row,center_col,half_height,half_width]
            # Draw bounding rectangle - this can also be done using min and max x,y values of 255 in the alpha frame but is more elegant this way
            rectangle_img = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
            output.write(rectangle_img)

        else:
            break

    capture_input.release()
    capture_alpha.release()
    output.release()
    cv2.destroyAllWindows()

    return tracking_dict
