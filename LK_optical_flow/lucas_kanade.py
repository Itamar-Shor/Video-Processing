import cv2
import numpy as np
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata


# FILL IN YOUR ID
ID1 = 207829144
ID2 = 315129551


PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

WINDOW_SIZE = 5


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.

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


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    for level in range(num_levels):
        next_level_conv = signal.convolve2d(pyramid[level], PYRAMID_FILTER, boundary='symm', mode='same')
        pyramid.append(next_level_conv[::2,::2])

    return pyramid


def find_LS_solution(Ix: np.ndarray, 
                     Iy: np.ndarray, 
                     It: np.ndarray, 
                     window_size: int, 
                     row: int, 
                     col: int) -> np.ndarray:
      """
      find Least-Squares solution
      """
      boundry = window_size // 2
      bottom = max(row-boundry, 0)
      top = min(row+boundry+1, Ix.shape[0])
      left = max(col-boundry, 0)
      right = min(col+boundry+1, Ix.shape[1])
    
      Ax = Ix[bottom:top, left:right].T.reshape(-1, 1) # flatten
      Ay = Iy[bottom:top, left:right].T.reshape(-1, 1)
      b = It[bottom:top, left:right].T.reshape(-1, 1)
      A = np.column_stack((Ax, Ay))
    
      try:
            inv_res = np.linalg.inv(np.matmul(A.T, A))
            return - np.matmul(np.matmul(inv_res, A.T), b)
      except np.linalg.LinAlgError as err:
            return np.zeros((2, 1))


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixelâ€™s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """
    h, w = I1.shape
    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = np.subtract(I2, I1, dtype=np.float64)
    boundry = window_size // 2
    du = np.zeros((h,w))
    dv = np.zeros((h,w))

    for i in range(boundry, h - boundry):
        for j in range(boundry, w - boundry):
            solution = find_LS_solution(Ix, Iy, It, window_size, row=i, col=j)
            du[i,j] = solution[0][0]
            dv[i,j] = solution[1][0]
        
    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """
    image_warp = image.copy()

    u_factor_scale = image.shape[1] / u.shape[1] # use num of cols
    v_factor_scale = image.shape[0] / v.shape[0] # use num of rows
    u = cv2.resize(u, (image.shape[1],image.shape[0])) * u_factor_scale
    v = cv2.resize(v, (image.shape[1],image.shape[0])) * v_factor_scale

    x,y = np.meshgrid([i for i in range(image.shape[1])], [i for i in range(image.shape[0])])
    
    image_warp = griddata((x.flatten(),y.flatten()), image.flatten(), 
                            ((x+u).flatten(), (y+v).flatten()), method='linear').reshape(image.shape)
    # replacing nan values
    image_warp[np.isnan(image_warp)] = image[np.isnan(image_warp)]

    return image_warp


def locas_kanade_optical_flow_template(I1: np.ndarray,
                                       I2: np.ndarray,
                                       window_size: int,
                                       max_iter: int,
                                       num_levels: int,
                                       step_func):# -> tuple[np.ndarray, np.ndarray]:)
    """
    template for lucas kanade optical flow. need to supply step function.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyarmid_I2 = build_pyramid(I2, num_levels)

    # start from u and v in the size of smallest image
    u = np.zeros(pyarmid_I2[-1].shape)
    v = np.zeros(pyarmid_I2[-1].shape)

    # following the fill algorithm shown in the lab (slide 14)
    for level in range(num_levels, -1, -1): # For every level in the image pyramid (from smallest)
        I2_warped = warp_image(pyarmid_I2[level], u, v) # Warp I2 from that level according to the current u and v
        for k in range(max_iter): # for num_iterations (max_iter)
            du, dv = step_func(pyramid_I1[level], I2_warped, window_size)
            u += du
            v += dv
            I2_warped = warp_image(pyarmid_I2[level], u, v) # new I2_warp

        if level > 0:
            # use resize (times 2?)
            u = 2 * cv2.resize(u, (u.shape[1]*2, u.shape[0]*2))
            v = 2 * cv2.resize(v, (v.shape[1]*2, v.shape[0]*2))
    return u, v


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    return locas_kanade_optical_flow_template(I1, I2, window_size, max_iter, 
                                                num_levels, step_func=lucas_kanade_step)


def lucas_kanade_video_stabilization_template(input_video_path: str,
                                              output_video_path: str,
                                              window_size: int,
                                              max_iter: int,
                                              num_levels: int,
                                              LK_OF,
                                              start_rows = 0,
                                              start_cols = 0,
                                              end_rows = 0,
                                              end_cols = 0) -> None:
    # 1
    capture = cv2.VideoCapture(input_video_path)
    # 2
    parameters = get_video_parameters(capture)
    output = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 
                             parameters['fps'], (parameters['width'], parameters['height']), isColor=False)
    # 3
    ret, I1 = capture.read()
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    output.write(np.uint8(
        cv2.resize(I1[start_rows:I1.shape[0] - end_rows, start_cols:I1.shape[1] - end_cols]
                    ,(parameters['width'], parameters['height']))
        )) # image.depth() == CV_8U
    # 4
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1 + 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1 + 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1 + 1)),
                  h_factor * (2 ** (num_levels - 1 + 1)))
    IMAGE_SIZE_np = (h_factor * (2 ** (num_levels - 1 + 1)),
                  w_factor * (2 ** (num_levels - 1 + 1)))
    if I1.shape != IMAGE_SIZE_np:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    # 5
    u, v = np.zeros(IMAGE_SIZE_np), np.zeros(IMAGE_SIZE_np)
    # 6
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for fno in tqdm(range(1, total_frames)):
        ret, I2 = capture.read()
        if ret:
            I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
            boundry = window_size // 2
            du, dv = LK_OF(I1, I2, window_size, max_iter, num_levels)
            # ignore edge pixels
            u += np.average(du[boundry:du.shape[0] - boundry + 1, boundry:du.shape[1] - boundry + 1])
            v += np.average(dv[boundry:dv.shape[0] - boundry + 1, boundry:dv.shape[1] - boundry + 1])

            output.write(np.uint8(
                cv2.resize(warp_image(I2, u, v)[start_rows:I2.shape[0] - end_rows, start_cols:I2.shape[1] - end_cols]
                            ,(parameters['width'], parameters['height']))
                )) # image.depth() == CV_8U

            I1 = I2
        else:
            break

    capture.release()
    output.release()
    cv2.destroyAllWindows()


def lucas_kanade_video_stabilization(input_video_path: str,
                                     output_video_path: str,
                                     window_size: int,
                                     max_iter: int,
                                     num_levels: int) -> None:
    """Use LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.

    Recipe:
        (1) Open a VideoCapture object of the input video and read its
        parameters.
        (2) Create an output video VideoCapture object with the same
        parameters as in (1) in the path given here as input.
        (3) Convert the first frame to grayscale and write it as-is to the
        output video.
        (4) Resize the first frame as in the Full-Lucas-Kanade function to
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (5) Create a u and a v which are og the size of the image.
        (6) Loop over the frames in the input video (use tqdm to monitor your
        progress) and:
          (6.1) Resize them to the shape in (4).
          (6.2) Feed them to the lucas_kanade_optical_flow with the previous
          frame.
          (6.3) Use the u and v maps obtained from (6.2) and compute their
          mean values over the region that the computation is valid (exclude
          half window borders from every side of the image).
          (6.4) Update u and v to their mean values inside the valid
          computation region.
          (6.5) Add the u and v shift from the previous frame diff such that
          frame in the t is normalized all the way back to the first frame.
          (6.6) Save the updated u and v for the next frame (so you can
          perform step 6.5 for the next frame.
          (6.7) Finally, warp the current frame with the u and v you have at
          hand.
          (6.8) We highly recommend you to save each frame to a directory for
          your own debug purposes. Erase that code when submitting the exercise.
       (7) Do not forget to gracefully close all VideoCapture and to destroy
       all windows.
    """
    return lucas_kanade_video_stabilization_template(input_video_path, output_video_path, 
                                                     window_size, max_iter, num_levels, 
                                                     LK_OF=lucas_kanade_optical_flow)


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    h,w = I1.shape
    SIZE_TH = 84*68

    # image is small enough
    if h*w <= SIZE_TH:
        return lucas_kanade_step(I1, I2, window_size)

    Ix = signal.convolve2d(I2, X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, Y_DERIVATIVE_FILTER, mode='same')
    It = np.subtract(I2, I1, dtype=np.float64)
    du = np.zeros((h,w))
    dv = np.zeros((h,w))
    response = cv2.cornerHarris(np.uint8(I2), blockSize=5, ksize=3, k=0.04)
    CORNERS_TH = 0.03 * np.max(response) 

    for i in range(h):
        for j in range(w):
            if response[i,j] >= CORNERS_TH:
                solution = find_LS_solution(Ix, Iy, It, window_size, row=i, col=j)
                du[i,j] = solution[0][0]
                dv[i,j] = solution[1][0]

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    return locas_kanade_optical_flow_template(I1, I2, window_size, max_iter, 
                                                num_levels, step_func=faster_lucas_kanade_step)


def lucas_kanade_faster_video_stabilization(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        None.
    """
    return lucas_kanade_video_stabilization_template(input_video_path, output_video_path, 
                                                     window_size, max_iter, num_levels, 
                                                     LK_OF=faster_lucas_kanade_optical_flow)


def lucas_kanade_faster_video_stabilization_fix_effects(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    return lucas_kanade_video_stabilization_template(input_video_path, output_video_path, 
                                                     window_size, max_iter, num_levels, 
                                                     LK_OF=faster_lucas_kanade_optical_flow,
                                                     start_rows=start_rows, start_cols=start_cols, 
                                                     end_rows=end_rows, end_cols=end_cols)


