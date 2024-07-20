import numpy as np





def linear_interpolation(img, points):
    px, py = points[..., 0], points[..., 1]
    w, h = img.shape[:2]
    outside = np.logical_or(np.logical_or(px < 0, px >= w), np.logical_or(py < 0, py >= h))
    
    x0 = np.minimum(np.maximum(0, px.astype(np.int32)), w-1)
    y0 = np.minimum(np.maximum(0, py.astype(np.int32)), h-1)
    x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
    a, b = px - x0, py - y0
    if len(img.shape) == 3: # 3 dim image; [w, h, c]
        a = a[..., np.newaxis]
        b = b[..., np.newaxis]
        outside = outside[..., np.newaxis]
    result = img[x0, y0] * (1-a) * (1-b) + img[x1, y0] * a * (1-b) + img[x0, y1] * (1-a) * b + img[x1, y1] * a * b
    return np.where(outside, 0.0, result)


def angle_sample_points(angle_samples):
    return np.linspace(0, 2*pi, angle_samples, endpoint=False, dtype=np.float32)


# return a [distances, angles, 2] array of points describing sampling locations on a circle
def get_circle_sample_points(angle_samples, sample_radius, dist_samples = None):        
    angles = angle_sample_points(angle_samples)
    directions = np.stack((np.cos(angles), np.sin(angles)), 1)  #shape [angles, 2]
    if dist_samples is None:
        return np.array([[sample_radius]]) * directions

    distances = np.linspace(sample_radius, 0, dist_samples, endpoint=False, dtype=np.float32)
    return directions[np.newaxis] * distances[:, np.newaxis, np.newaxis]
