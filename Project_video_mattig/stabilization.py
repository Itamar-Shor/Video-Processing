import cv2
import numpy as np
from tqdm import tqdm
import utils


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')
    # we want to pad the curve in such a way the the averaging convolution will give the edges the original value
    for i in range(radius):
        curve_pad[i] += 2 * (curve_pad[radius] - curve_pad[i])
        curve_pad[len(curve_pad) - i - 1] += 2 * (curve_pad[len(curve_pad) - radius - 1] - curve_pad[len(curve_pad) - i - 1])
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=5)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def stabilize(input_video_path: str, output_video_path: str) -> None:
    """
    The idea was taken from - https://learnopencv.com/video-stabilization-using-point-feature-matching-in-opencv,
    but instead of estimating interest points in the following frame with LK optical flow, we find
    interest points in both frames and try to find a match between them.
    The general idea is similiar to HW2 LK_faster_video_stabilization with openCV functions:
      [1]: find interest point to track in each frame.
      [2]: estimate the motion between those points in following frames:
          [2.1]: match features by calculating distance between thier orb descriptors.
          [2.2]: estimate homograpy between the matching features.
      [3]: smooth the motion to avoid sharp movements (this is an extra step we didn't do in HW2).
      [4]: wrap each frame by the motion model to the following frame.
      [5]: write the wrapped frame to the output video.
    """
    # initiate structs
    capture = cv2.VideoCapture(input_video_path)
    parameters = utils.get_video_parameters(capture)
    output = cv2.VideoWriter(output_video_path,
                             cv2.VideoWriter_fourcc(*'XVID'),
                             parameters['fps'],
                             (parameters['width'], parameters['height']),
                             isColor=True)
    ret, I1 = capture.read()
    I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    transforms = np.zeros((total_frames - 1, 9), np.float32)
    # for feature matching
    orb = cv2.ORB_create(nfeatures=1000, edgeThreshold=1)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for fno in tqdm(range(total_frames - 1), desc="Stablization - Extracting Trasnformations"):
        ret, I2 = capture.read()
        if ret:
            I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
            # extract interest points from both images and get descriptors for them
            prev_pts = cv2.goodFeaturesToTrack(I1_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            keypoints_prev = [cv2.KeyPoint(x, y, 1) for p in prev_pts for x, y in p]
            keypoints_prev, descriptors_prev = orb.compute(I1_gray, keypoints_prev)

            next_pts = cv2.goodFeaturesToTrack(I2_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
            keypoints_next = [cv2.KeyPoint(x, y, 1) for p in next_pts for x, y in p]
            keypoints_next, descriptors_next = orb.compute(I2_gray, keypoints_next)

            # match interest points - take only the top 15% matches
            matches = list(bf.match(descriptors_prev, descriptors_next))
            matches.sort(key=lambda x: x.distance, reverse=False)
            matches = matches[:int(0.15 * len(matches))]

            # getting actual list of matching points
            prevPts = np.zeros((len(matches), 2), dtype=np.float32)
            nextPts = np.zeros((len(matches), 2), dtype=np.float32)
            for i, match in enumerate(matches):
                prevPts[i, :] = keypoints_prev[match.queryIdx].pt
                nextPts[i, :] = keypoints_next[match.trainIdx].pt

            # Find transformation matrix using matching points (approximation)
            m, _ = cv2.findHomography(prevPts, nextPts, cv2.RANSAC)
            # Store transformation
            transforms[fno] = m.flatten()
            # Move to next frame
            I1_gray = I2_gray

        else:
            break

    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)
    # smooth the moition (this added on top of the algo from HW2)
    smoothed_trajectory = smooth(trajectory)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory
    # Calculate newer transformation array
    transforms_smooth = transforms + difference

    # Reset stream to first frame
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for fno in tqdm(range(total_frames - 1), desc="Stablization - Applying Smoothed Transformations"):
        ret, frame = capture.read()
        if ret:
            # Extract transformations from the new transformation array
            m = transforms_smooth[fno].reshape((3, 3))
            # Apply wrapping to the given frame (by the motion estimation)
            frame_stabilized = cv2.warpPerspective(frame, m, (parameters['width'], parameters['height']))
            # Fix border artifacts (scaling the image)
            frame_stabilized = fixBorder(frame_stabilized)
            output.write(frame_stabilized)
        else:
            break

    _, frame = capture.read()
    output.write(fixBorder(frame))

    capture.release()
    output.release()
    cv2.destroyAllWindows()
