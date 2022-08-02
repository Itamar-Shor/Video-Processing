import GeodisTK
import cv2
import numpy as np
from tqdm import tqdm
import utils


def matting(stabilize_video_path, binary_video_path, new_background, alpha_video_path, matted_video_path):
    """
    for each frame:
        [1] use morphological transformations to create the trimap from the binary.
        [2] calc the distance maps of the current frame (bg and fg).
        [3] estimate the undecided region (narrow band) from the distance maps.
        [4] estimate probabilities of bg/fg in the narrow band using KDE.
        [5] calc weights from the distance maps and probs as we saw in class (D^-r * prob)
        [6] calc alpha using the weights.
        [7] using the alpha, create matted frame with new background.
    """
    # Read input video
    new_background = cv2.imread(new_background)
    capture_stab = cv2.VideoCapture(stabilize_video_path)
    capture_binary = cv2.VideoCapture(binary_video_path)
    parameters = utils.get_video_parameters(capture_stab)
    alpha_writer = cv2.VideoWriter(alpha_video_path,
                                   cv2.VideoWriter_fourcc(*'XVID'),
                                   parameters['fps'],
                                   (parameters['width'], parameters['height']),
                                   isColor=False)

    matted_writer = cv2.VideoWriter(matted_video_path,
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    parameters['fps'],
                                    (parameters['width'], parameters['height']),
                                    isColor=True)
    # Get frame count
    total_frames = int(capture_stab.get(cv2.CAP_PROP_FRAME_COUNT))
    wf, hf = parameters['width'], parameters['height']

    new_background = cv2.resize(new_background, (wf, hf))

    for fno in tqdm(range(total_frames), desc="Matting"):
        ret, frame = capture_stab.read()
        ret2, binary = capture_binary.read()
        if ret and ret2:
            mask = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
            mask = (mask > 200).astype(np.uint8)  # S for geodist
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # work only on small window of object (extracted from binary)
            NB_width = 15
            x, y, w, h = cv2.boundingRect(mask)
            xL, xR, yT, yB = max(0, x - NB_width), min(wf, x + w + NB_width), max(0, y - NB_width), min(hf, y + h + NB_width)
            mask_window = mask[yT:yB, xL:xR]
            frame_window = frame[yT:yB, xL:xR]
            frame_gray_window = frame_gray[yT:yB, xL:xR]
            new_background_window = new_background[yT:yB, xL:xR]

            # build fg, bg masks and narrow band - NB
            fg_mask = cv2.erode(mask_window, np.ones((5, 5), np.uint8), iterations=1)
            bg_mask = 1 - cv2.dilate(mask_window, np.ones((5, 5), np.uint8), iterations=1)

            fg_distance_map = GeodisTK.geodesic2d_raster_scan(frame_gray_window, fg_mask, 1.0, 2)
            bg_distance_map = GeodisTK.geodesic2d_raster_scan(frame_gray_window, bg_mask, 1.0, 2)

            epsilon_NB = 0.99
            fg_distance_map = fg_distance_map / (fg_distance_map + bg_distance_map)
            bg_distance_map = 1 - fg_distance_map
            NB_mask = (np.abs(fg_distance_map - bg_distance_map) < epsilon_NB).astype(np.uint8)
            NB_indices = np.where(NB_mask == 1)

            fg_distance_mask = (fg_distance_map < bg_distance_map - epsilon_NB).astype(np.uint8)
            bg_distance_mask = (bg_distance_map >= fg_distance_map - epsilon_NB).astype(np.uint8)

            n_of_samples = 150
            omega_f = np.random.permutation(frame_window[fg_distance_mask == 1])[:n_of_samples]
            omega_b = np.random.permutation(frame_window[bg_distance_mask == 1])[:n_of_samples]

            kde_estimator = utils.KDE()
            kde_estimator.calc_distributions_with_kde(omega_b, omega_f, 1)
            NB_bg_prob, NB_fg_prob = kde_estimator.evaluate_pdfs(frame_window[NB_indices][:, np.newaxis, :])

            # calc alpha
            w_f = np.power(fg_distance_map[NB_indices], -2) * np.squeeze(NB_fg_prob)
            w_b = np.power(bg_distance_map[NB_indices], -2) * np.squeeze(NB_bg_prob)
            alpha_narrow_band = w_f / (w_f + w_b)
            alpha_window = np.copy(fg_mask).astype(np.float64)
            alpha_window[NB_indices] = alpha_narrow_band

            # matting
            matted_window = alpha_window[:, :, np.newaxis] * frame_window + (1 - alpha_window[:, :, np.newaxis]) * new_background_window

            matted_frame = np.copy(new_background)
            matted_frame[yT:yB, xL:xR] = matted_window
            matted_writer.write(matted_frame)

            alpha_frame = np.zeros_like(mask)
            alpha_frame[yT:yB, xL:xR] = (alpha_window * 255)
            alpha_writer.write(alpha_frame)
        else:
            break

    capture_stab.release()
    capture_binary.release()
    alpha_writer.release()
    matted_writer.release()
    cv2.destroyAllWindows()
