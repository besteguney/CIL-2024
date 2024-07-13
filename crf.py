import numpy as np
from scipy.ndimage import gaussian_filter

# Defining the class for the last step of DCED framework --> CRF
class CRF():
    def __init__(self, kernel_1_weight=10, kernel_2_weight=5, alpha=60, beta=10, gamma=1, efficient=False, spatial_downsampling=15, range_downsampling=15, iterations=3):
        self.kernel_1_weight = kernel_1_weight
        self.kernel_2_weight = kernel_2_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.efficient = efficient
        self.spatial_downsampling = spatial_downsampling
        self.range_downsampling = range_downsampling
        self.iterations = iterations

    def appearance_kernel(self, x_1, y_1, p_1, x_2, y_2, p_2):
        """Compute appearance kernel.
    
        Args:
            x_1: X coordinate of first pixel.
            y_1: Y coordinate of first pixel.
            p_1: Color vector of first pixel.
            x_2: X coordinate of second pixel.
            y_2: Y coordinate of second pixel.
            p_2: Color vector of second pixel.
            theta_alpha: Standard deviation for the position.
            theta_beta: Standard deviation for the color.
    
        Returns:
            The output of the appearence kernel.
        """
        result = np.exp(
        -((x_1 - x_2) ** 2.0 + (y_1 - y_2) ** 2.0) / (2 * self.alpha ** 2.0)
        - np.sum((p_1 - p_2) ** 2.0) / (2.0 * self.beta ** 2.0)
        )
        #print(f'Result of apperance kernel is {result}')
        return result


    def smoothness_kernel(self, x_1, y_1, p_1, x_2, y_2, p_2):
        """Compute smoothness kernel.
    
        Args:
            x_1: X coordinate of first pixel.
            y_1: Y coordinate of first pixel.
            p_1: Color vector of first pixel.
            x_2: X coordinate of second pixel.
            y_2: Y coordinate of second pixel.
            p_2: Color vector of second pixel.
            theta_gamma: Standard deviation for the position.
    
        Returns:
            The output of the smoothness kernel.
        """
        del p_1, p_2
        result = np.exp(
            -((x_1 - x_2) ** 2.0 + (y_1 - y_2) ** 2.0) / (2.0 * self.gamma ** 2.0)
        )
        #print(f'Result of smoothness kernel {result}')
        return result

    def normalize(self, potentials):
        """Normalize potentials such that output is a valid pixelwise distribution.
    
        Args:
            potentials: Array of potentials. Shape (H,W,N).
    
        Returns:
            Probability array with same shape as potentials.
            Probabilities sum up to 1 at every slice (i,j,:).
        """
        # Sum the potentials along the last axis (the class axis)
        sum_potentials = np.sum(potentials, axis=-1, keepdims=True)
    
        # Avoid division by zero
        sum_potentials[sum_potentials == 0] = 1
    
        # Normalize by dividing each potential by the sum of potentials at that pixel
        normalized_potentials = potentials / sum_potentials
    
        return normalized_potentials

    def message_passing(self, image, current_probabilities) :
        """Perform "message passing" as the first step of the inference loop.
    
        Args:
            image:
                Array of size ROWS x COLUMNS x CHANNELS, representing the image used to
                compute the kernel.
            current_probabilities:
                Array of size ROWS x COLUMNS x CLASSES, representing the current
                probabilities.
            kernel_functions: The kernel functions defining the edge potential.
    
        Returns:
            Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the intermediate
            result of message passing for each kernel.
        """
        # naive version
        rows = image.shape[0]
        cols = image.shape[1]
        classes = current_probabilities.shape[2] # road or not
        result = np.zeros(
            (
                current_probabilities.shape[0],
                current_probabilities.shape[1],
                classes, #1 class --> road or not
                2, # 2 kernels
            ),
            dtype=float,
        )
        
    
        # TODO implement naive message passing (using loops)
        for i in range(rows):
            for j in range(cols):
                probability_1 = 0
                probability_2 = 0
                color_vector_1 = image[i, j, :]
                for k in range(rows):
                    for l in range(cols):
                        if (i == k) and (j == l):
                            pass
                        else:
                            color_vector_2 = image[k, l, :]
                            probability_1 = probability_1 + result[k, l, 0, 0] * self.appearance_kernel(i, j, color_vector_1, k, l, color_vector_2)
                            probability_2 = probability_2 + result[k, l, 0, 1] * self.smoothness_kernel(i, j, color_vector_1, k, l, color_vector_2)
                result[i, j, 0, 0] = probability_1
                result[i, j, 0, 1] = probability_2
                #print(f'----------- {i}, {j}, {probability_1}, {probability_2}')
        return result

    def compatibility_transform(self,q_tilde):
        """Perform compatability transform as part of the inference loop.
    
        Args:
            q_tilde:
                Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the
                intermediate result of message passing for each kernel.
            weights: Weights of each kernel.
    
        Returns:
            Array of size ROWS x COLUMNS x CLASSES, representing the result after combining
            the kernels and applying the label compatability function (here: Potts model).
        """
    
        # TODO: implement compatability transform (try with matrix operations only)
        weights = [self.kernel_1_weight, self.kernel_2_weight]
        q_tilde[..., 0] *= weights[0] 
        q_tilde[..., 1] *= weights[1]
        result = np.sum(q_tilde, axis=-1)
        return result

    def get_unary_potential(self, image):
        return -np.log(image)

    def local_update(self, q_hat, unary_potential):
        """Perform local update as part of the interefence loop.
    
        Args:
            q_hat:
                Array of size ROWS x COLUMNS x CLASSES, representing the intermediate result
                after combining the kernels and applying the label compatability function.
            unary_potential:
                Array of size ROWS x COLUMNS x CLASSES, representing the prior energy for
                each pixel and class from a different source.
        Returns:
            Array of size ROWS x COLUMNS x CLASSES, representing the probabilities for each
            pixel and class.
        """
        result = np.exp(-unary_potential - q_hat)
        #print(f'Local update result is {result}')
        return np.exp(-unary_potential - q_hat)

    def efficient_message_passing(self, image, current_probabilities):
        """Perform efficient "message passing" by downsampling and convolution in 5D.
    
        This assumes two kernels: an appearance kernel based on theta_alpha and theta_beta,
        and a smoothness kernel based on theta_gamma.
    
        Args:
            image:
                Array of size ROWS x COLUMNS x CHANNELS, representing the image used to
                compute the kernel.
            current_probabilities:
                Array of size ROWS x COLUMNS x CLASSES, representing the current
                probabilities.
            spatial_downsampling:
                Factor to downsample the spatial dimensions for the 5D representation.
            range_downsampling:
                Factor to downsample the range dimensions for the 5D representation.
            theta_alpha: Spatial standard deviation for the appearance kernel.
            theta_beta: Color standard deviation for the appearance kernel.
            theta_gamma: Spatial standard deviation for the smoothness kernel.
    
        Returns:
            Array of size ROWS x COLUMNS x CLASSES x KERNELS, representing the intermediate
            result of message passing for each kernel.
        """
        #t_0 = time.time()
    
        rows = image.shape[0]
        cols = image.shape[1]
        classes = current_probabilities.shape[2]
        color_range = 255
    
        ds_rows = int(np.ceil(rows / self.spatial_downsampling))
        ds_cols = int(np.ceil(cols / self.spatial_downsampling))
        ds_range = int(np.ceil(color_range / self.range_downsampling))
    
        #print(f"Downsampled to: {ds_rows}x{ds_cols}x{ds_range}")
    
        result = np.zeros(
            (
                current_probabilities.shape[0],
                current_probabilities.shape[1],
                current_probabilities.shape[2],
                2,
            ),
            dtype=float,
        )
    
        # Precompute indices
        indices_list = []
        for row in np.arange(rows):
            for col in np.arange(cols):
                indices_list.append(
                    (row, col, image[row, col, 0], image[row, col, 1], image[row, col, 2])
                )
        indices_list = np.array(indices_list, dtype=float)
        indices_list[:, 0:2] = indices_list[:, 0:2] / float(self.spatial_downsampling)
        indices_list[:, 2:] = indices_list[:, 2:] / float(self.range_downsampling)
        indices_list = np.round(indices_list).astype(int)

        for class_id in np.arange(classes):
            # Allocate 5D feature space
            feature_space = np.zeros((ds_rows+1, ds_cols+1, ds_range+1, ds_range+1, ds_range+1))
            # Downsample with box filter and go to 5D space at same time
            for row in np.arange(rows):
                for col in np.arange(cols):
                    idx = indices_list[row * cols + col]
                    feature_space[idx[0], idx[1], idx[2], idx[3], idx[4]] += current_probabilities[row, col, 0]
    
            for kernel_id in np.arange(2):
                if kernel_id == 0:  # Appearance kernel
                    # Apply appearance kernel as a Gaussian filter
                    filtered_feature_space = gaussian_filter(feature_space, sigma=[self.alpha , self.alpha, self.beta, self.beta, self.beta])
        
                if kernel_id == 1:  # Smoothness kernel
                    # Apply smoothness kernel as a Gaussian filter
                    filtered_feature_space = gaussian_filter(feature_space, sigma=[self.gamma, self.gamma, 0, 0, 0])
        
                # Upsample with simple lookup (no interpolation for simplicity)
                for row in np.arange(rows):
                    for col in np.arange(cols):
                        idx = indices_list[row * cols + col]
                        result[row, col, 0, kernel_id] = filtered_feature_space[idx[0], idx[1], idx[2], idx[3], idx[4]]
        
        #t_1 = time.time()
        #print(f"Efficient message passing took {t_1-t_0}s")
    
        return result
    
    def inference(self, image, initial_probabilities):
        """Perform inference in fully connected CRF with Gaussian edge potentials.
    
        Args:
            image:
                Array of size ROWS x COLUMNS x CHANNELS, representing the image used the
                features.
            initial_probabilities:
                Initial pixelwise probabilities for each class. Used to initialize unary
                potential.
            params:
                Parameter class for fully connected CRFs (see CrfParameters documentation).
        Return:
            Array of size ROWS x COLS x CLASSES
        """
        # initialize
        current_probabilities = initial_probabilities
    
        unary_potential = -np.log(current_probabilities)
    
        for _ in np.arange(self.iterations):
            if self.efficient:
                q_tilde = self.efficient_message_passing(image,current_probabilities)
            else:
                q_tilde = self.message_passing(image, current_probabilities)
            q_hat = self.compatibility_transform(q_tilde)
            unnormalized_probabilities = self.local_update(q_hat, unary_potential)
            #print(unnormalized_probabilities)
            current_probabilities = self.normalize(unnormalized_probabilities)
            #print(current_probabilities)
            print('Iteration completed')
    
        return current_probabilities