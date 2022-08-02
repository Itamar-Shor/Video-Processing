import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "207829144"
ID2 = "315129551"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


X_CENTER = 0
Y_CENTER = 1
WIDTH = 2
HEIGHT = 3
X_VEL = 4
Y_VEl = 5

noise_mean = [0,0,0,0,0,0]

noise_cov = [[25,0,0,0,0,0],
             [0,4,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,1,0],
             [0,0,0,0,0,0.5]]


def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    s_prior[:2,:] += s_prior[4:,:]
    state_drifted = s_prior + np.random.multivariate_normal(noise_mean, noise_cov, s_prior.shape[1]).T # can put N instead of s_prior.shape[1]
    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    hist = np.zeros((16,  16,  16))

    x_center = state[X_CENTER]
    width = state[WIDTH]
    y_center = state[Y_CENTER]
    height = state[HEIGHT]

    window = image[max(0,y_center-height):min(y_center+height+1,image.shape[0]), 
                   max(0,x_center-width):min(x_center+width+1,image.shape[1])] // 16

    # care out of bounds for i,j
    for i in range(window.shape[0]):
        for j in range(window.shape[1]):
          hist[window[i,j,0], window[i,j,1], window[i,j,2]] += 1
    hist = np.reshape(hist, 16 * 16 * 16)

    # normalize
    hist = hist / np.sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = np.zeros(previous_state.shape)

    for i in range(previous_state.shape[1]):
      r = np.random.rand()
      j = np.searchsorted(cdf, r, side='left')
      S_next[:,i] = previous_state[:,j]
    return S_next


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = np.exp(20*np.sum(np.sqrt(p*q)))
    return distance


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg, _, _) = np.sum(state*W, axis=1)
    (x_avg, y_avg, w_avg, h_avg) = (x_avg-w_avg, y_avg-h_avg, w_avg*2, h_avg*2)

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max, _, _) = state[:,np.argmax(W)]
    (x_max, y_max, w_max, h_max) = (x_max-w_max, y_max-h_max, w_max*2, h_max*2)

    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state


def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    W = []
    C = []
    sum = 0
    for col in range(S.shape[1]):
      p = compute_normalized_histogram(image, S[:,col])
      W.append(bhattacharyya_distance(p,q))
      sum += W[-1]
      C.append(sum)
    W = W / np.sum(W)
    C = C / sum

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        W = []
        C = []
        sum = 0
        for col in range(S.shape[1]):
          p = compute_normalized_histogram(current_image, S[:,col])
          W.append(bhattacharyya_distance(p,q))
          sum += W[-1]
          C.append(sum)
        W = W / np.sum(W)
        C = C / sum

        # CREATE DETECTOR PLOTS
        images_processed += 1
        
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)


    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
