import copy
import numpy as np

# Defining the functions for the framework
def gaussian_smoothing(kernel_size, sigma=1):
  kernel_size = int(kernel_size) // 2
  x, y = np.mgrid[-kernel_size:kernel_size+1, -kernel_size:kernel_size+1]
  normal = 1 / (2.0 * np.pi * sigma**2)
  g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
  return g

def connected_component_labeling(prediction, gaussian_filter, threshold=128):
    mask = np.uint8(prediction*255)
    mask = cv2.filter2D(mask,-1,gaussian_filter)
    _, binary_image = cv2.threshold(np.uint8(mask), threshold, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=4)
    # Stats is --> https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
    
    #print(f"Number of labels: {num_labels}")
    #print("Stats: ")
    #print(stats)
    #print("Centroids: ")
    #print(centroids)
    return num_labels, labels, stats, centroids

def calculate_shape_index(stats):
  perimeter = 2 * (stats[2] + stats[3])
  return perimeter / (4 * math.sqrt(stats[-1]))

def remove_noise(image, num_labels, labels, stats, threshold=1.25, isprint=False):
    output = copy.deepcopy(image)
    # Map component labels to hue value
    for label in range(1, num_labels):
        mask = labels == label
        index = calculate_shape_index(stats[label].tolist())
        if isprint:
            print(label, 'and', index)
        if index < threshold:
          output[mask] = 0 # removing the object
    return output

# Applying gaussian blur + ccl + lm
#test_pred = test_pred.reshape(test_pred.shape[0], test_pred.shape[1], test_pred.shape[2], 1)
def apply_lm(sigma=3, ccl_threshold=128, shape_index_threshold=1.3):
    gaussian_filter = gaussian_smoothing(filter_size, sigma)
    filter_size = 6*sigma+1 # Rule of thumb: size is 6 times standard deviation
    
    output = []
    
    for i in range(test_pred.shape[0]):
        num_labels, labels, stats, centroids = connected_component_labeling(test_pred[i], gaussian_filter, threshold)
        lm_output = remove_noise(test_pred[i], num_labels, labels, stats, shape_index_threshold)
        output.append(lm_output)
    output = np.array(output)
    return output