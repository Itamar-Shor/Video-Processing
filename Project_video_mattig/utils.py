import cv2
import numpy as np
from scipy import stats


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """ Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.

    Returns:
        parameters: dict. Video parameters extracted from the video.
    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def crop_largest_contour(mask):
    """ 
    Extract largest contour from binary image (return cropped image and contour)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    foreground_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(foreground_mask, [contours[-1]], -1, 255, -1)
    return foreground_mask, contours[-1]


class KDE:
    """
    This class handles all the KDE calculations more efficiently.
    """
    def __init__(self):
        self.fg_seen_colors = dict()
        self.bg_seen_colors = dict()
        self.color_dist_given_B = None
        self.color_dist_given_F = None

    def calc_distributions_with_kde(self, bg, fg, bw):
        """
        calc f(c|F) and f(c|B) using KDE.
        return: (f(c|B), f(c|F))
        """
        self.color_dist_given_F = stats.gaussian_kde(fg.T, bw_method=bw)
        self.color_dist_given_B = stats.gaussian_kde(bg.T, bw_method=bw)

    def evaluate_pdfs(self, frame):
        """
        Evaluate the KDE function on every pixel in a given frame
        """
        h, w = frame.shape[:2]
        unique_colors, inv = np.unique(frame.reshape(-1, frame.shape[2]), axis=0, return_inverse=True)

        colors = set([tuple(color) for color in unique_colors if tuple(color) not in self.fg_seen_colors])
        colors_eval = np.array([np.array(c) for c in colors])

        Z_f = self.color_dist_given_F.evaluate(colors_eval.T)
        Z_b = self.color_dist_given_B.evaluate(colors_eval.T)

        zip_iterator = zip(colors, Z_f)
        self.fg_seen_colors.update(dict(zip_iterator))
        zip_iterator = zip(colors, Z_b)
        self.bg_seen_colors.update(dict(zip_iterator))

        # unique_colors_eval, inv = np.unique(frame.reshape(-1, frame.shape[2]), axis=0, return_inverse = True)
        Zf = np.array([self.fg_seen_colors[tuple(c)] for c in unique_colors])[inv].reshape((h, w))
        Zb = np.array([self.bg_seen_colors[tuple(c)] for c in unique_colors])[inv].reshape((h, w))

        mehane = (Zb + Zf)
        bg_dist = np.divide(Zb, mehane, out=np.zeros_like(Zb), where=mehane != 0).reshape(h, w)
        fg_dist = np.divide(Zf, mehane, out=np.zeros_like(Zf), where=mehane != 0).reshape(h, w)

        return bg_dist, fg_dist
