import cv2
import numpy as np
from tqdm import tqdm
import utils


def estimate_initial_segmentation(frames):
    """
    use openCV's createBackgroundSubtractorKNN to estimate the background and foreground.
    return: fg_mask (1=foreground, 0=background) per frame.
    """
    nof_frames = len(frames)
    #middle = nof_frames // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # createBackgroundSubtractorKNN has a learning curve so we use the latter half of frames from each direction
    backSub_forward = cv2.createBackgroundSubtractorKNN()
    #backSub_backward = cv2.createBackgroundSubtractorKNN()

    foreground_estimations_forward = []
    #foreground_estimations_backward = []

    for i in tqdm(range(nof_frames),desc="Background Subtraction - Estimate Initial Segmentation part 1"):
        backSub_forward.apply(frames[i])

    for i in tqdm(range(nof_frames),desc="Background Subtraction - Estimate Initial Segmentation part 2"):
        backSub_forward.apply(frames[i])

    for i in tqdm(range(nof_frames),desc="Background Subtraction - Estimate Initial Segmentation part 3"):
        mask = backSub_forward.apply(frames[i])
        mask = (mask > 150).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        foreground_estimations_forward.append(utils.crop_largest_contour(mask)[0])

        # mask = backSub_forward.apply(frames[nof_frames - i - 1])
        # mask = (mask > 150).astype(np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # foreground_estimations_backward.append(utils.crop_largest_contour(mask)[0])

    return foreground_estimations_forward #foreground_estimations_backward[::-1] + foreground_estimations_forward

def sample_colors_and_create_KDE(frames, fg_mask_per_frame, n_samples, bH, tH, bw):

    fg_colors = np.random.permutation(frames[0][bH:tH,:][fg_mask_per_frame[0][bH:tH,:] == 255])[:n_samples]
    bg_colors = np.random.permutation(frames[0][bH:tH,:][fg_mask_per_frame[0][bH:tH,:] == 0])[:n_samples]

    for fno in tqdm(range(1, len(frames)), desc="Background Subtraction - Color Extraction"):
        # sampling colors to make a kde estimation of fg and bg
        fg_colors = np.concatenate(
            (fg_colors, np.random.permutation(frames[fno][bH:tH,:][fg_mask_per_frame[fno][bH:tH,:] == 255])[:n_samples]))

        bg_colors = np.concatenate(
            (bg_colors, np.random.permutation(frames[fno][bH:tH,:][fg_mask_per_frame[fno][bH:tH,:] == 0])[:n_samples]))

    kde_estimator = utils.KDE()
    kde_estimator.calc_distributions_with_kde(bg=bg_colors, fg=fg_colors, bw=bw)
    return kde_estimator


def background_substraction(input_video_path, binary_video_path, extracted_video_name):
    """
    [1] use createBackgroundSubtractorKNN to get Sigma_F and Sigma_B (initial scrabbles).
    [2] calc f(c|F) and f(c|B) using KDE.
    [3] calc f(F|c) = f(c|F)/(f(c|F)+f(c|B)) and f(B|c) in the same fashion.
    [4] estimate the binary from the probabilities.
    """
    capture = cv2.VideoCapture(input_video_path)
    parameters = utils.get_video_parameters(capture)
    output_binary = cv2.VideoWriter(binary_video_path,
                                    cv2.VideoWriter_fourcc(*'XVID'),
                                    parameters['fps'],
                                    (parameters['width'], parameters['height']),
                                    isColor=False)
    output_extracted = cv2.VideoWriter(extracted_video_name,
                                       cv2.VideoWriter_fourcc(*'XVID'),
                                       parameters['fps'],
                                       (parameters['width'], parameters['height']),
                                       isColor=True)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    for i in range(total_frames):
        ret, frame = capture.read()
        if ret:
            frames.append(frame)
        else:
            break

    h, w = frames[0].shape[:2]

    # generate bg/fg scribbles
    fg_mask_per_frame = estimate_initial_segmentation(frames)

    n_samples = 15

    h1 = int(h*0.3)
    h2 = int(h*0.66)
    face_pdf = sample_colors_and_create_KDE(frames, fg_mask_per_frame, n_samples, 0, h1, 0.1)
    body_pdf = sample_colors_and_create_KDE(frames, fg_mask_per_frame, n_samples, h1, h2, 1)
    shoes_pdf = sample_colors_and_create_KDE(frames, fg_mask_per_frame, n_samples, h2, h, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    epsilon_x = 50
    epsilon_w = 150
    for fno in tqdm(range(total_frames), desc="Background Subtraction - KDE evaluation"):

        # we can rule out area far from our scribble (based on prior knowledge of problem) to make less calculations
        x, y, rw, rh = cv2.boundingRect(fg_mask_per_frame[fno])
        xL, xR, yT, yB = max(0, x - epsilon_x), min(w, x + rw + epsilon_x + epsilon_w), 0, h

        # get probability maps based on kde evaluation of fg and bg colors
        face_bg_dist, face_fg_dist = face_pdf.evaluate_pdfs(frames[fno][yT:yB, xL:xR][:h1, :])
        body_bg_dist, body_fg_dist = body_pdf.evaluate_pdfs(frames[fno][yT:yB, xL:xR][h1:h2, :])
        shoes_bg_dist, shoes_fg_dist = shoes_pdf.evaluate_pdfs(frames[fno][yT:yB, xL:xR][h2:, :])

        # derive binary window mask
        extracted = np.zeros_like(frames[fno])
        tri_map = np.zeros((h, w), dtype=np.uint8)

        tri_map_win = np.zeros((yB - yT, xR - xL), dtype=np.uint8)
        tri_map_win[0:h1,:][face_fg_dist < face_bg_dist] = 0
        tri_map_win[0:h1,:][face_fg_dist > face_bg_dist] = 255
        tri_map_win = cv2.erode(tri_map_win, np.ones((9, 9), np.uint8), iterations=1)
        tri_map_win, c = utils.crop_largest_contour(tri_map_win)
        tri_map_win = cv2.dilate(tri_map_win, np.ones((9, 9), np.uint8), iterations=1)
        tri_map_win[h1:h2,:][body_fg_dist < body_bg_dist] = 0
        tri_map_win[h1:h2,:][body_fg_dist > body_bg_dist] = 255
        tri_map_win[h2:h,:][shoes_fg_dist < shoes_bg_dist] = 0
        tri_map_win[h2:h,:][shoes_fg_dist > shoes_bg_dist] = 255

        tri_map_win = cv2.morphologyEx(tri_map_win, cv2.MORPH_OPEN, kernel)

        tri_map_win, c = utils.crop_largest_contour(tri_map_win)

        # draw binary window back to binary frame
        tri_map[yT:yB, xL:xR][tri_map_win == 255] = 255

        output_binary.write(tri_map)
        extracted[yT:yB, xL:xR][tri_map_win == 255] = frames[fno][yT:yB, xL:xR][tri_map_win == 255]
        output_extracted.write(extracted)

    capture.release()
    output_binary.release()
    output_extracted.release()
    cv2.destroyAllWindows()
