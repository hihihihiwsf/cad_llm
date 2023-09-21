import math
import cv2
from geometry.opencv_colors import CV2_COLORS

import numpy as np
from numpy.linalg import cholesky

def get_distances(x: np.ndarray, length_scale: float, squared: bool=False) -> np.ndarray: 
    x2d = np.atleast_2d(x) / length_scale
    squared_distances = (x2d-x2d.T)**2
    distances = np.sqrt(squared_distances)
    return distances if squared is False else (distances, squared_distances)

def jitter(arr: np.ndarray, nugget: float=1e-6) -> np.ndarray: 
    m, _ = arr.shape 
    return arr + (np.eye(m) * nugget)

def matern(x: np.ndarray, length_scale: float, amplitude: float, **kwargs) -> np.ndarray:   
    nu = kwargs.get("nu", 3)
    if nu == 3: 
        distances = get_distances(x, length_scale)
        K = (1 + np.sqrt(nu) * distances) * np.exp(-np.sqrt(nu) * distances)
    elif nu == 5: 
        distances, squared_distances = get_distances(x, length_scale, squared=True)
        K = (1 + np.sqrt(nu) * distances + nu * squared_distances/3) * np.exp(-np.sqrt(nu) * distances)
    else: 
        raise NotImplementedError
    return amplitude**2 * jitter(K)


class Curve:
    def __init__(self, points):
        self.points = points
        self.good = False
        self.invalid_reason = None
        
        self.resolution = 500
        self.max_x = -100
        self.min_x = 100
        self.max_y = -100
        self.min_y = 100
    
    def update_x(self, x: np.ndarray):
        if x > self.max_x:
            self.max_x = x
        if x < self.min_x:
            self.min_x = x
            
    def update_y(self, y: np.ndarray):
        if y > self.max_y:
            self.max_y = y
        if y < self.min_y:
            self.min_y = y

    def _get_chol(self, ls: float=0.05, amp: float=0.002):
        self.scale = 10 * np.sqrt((self.max_x-self.min_x)**2 + (self.max_y-self.min_y)**2)
        self.x = np.linspace(0, 1, self.resolution)
        K = matern(self.x, ls, amp)
        self.cK = cholesky(K)

    def draw_points(self, ax):
        """
        Draw markers for the points
        """
        assert self.good, "The curve is not in the good state"
        for x, y in self.points:
            ax.plot(x, y, 'b.')

    def draw_points_np(self, np_image, cell_size, radius=2):
        """
        Draw markers for the points using opencv
        """
        for point in self.get_shifted_points(cell_size=cell_size):
            cv2.circle(np_image, point, radius=radius, color=CV2_COLORS["black"], thickness=-1)

    def draw_points_pil(self, img_draw, color="black", transform=None):
        r = 3
        points = self.points
        if transform:
            points = [(transform(x), transform(y)) for x, y in points]

        for x, y in points:
            img_draw.ellipse(xy=(x-r, y-r, x+r, y+r), fill=color, outline=None, width=1)


    def get_shifted_points(self, cell_size):
        """
        Shift points to center of cell in quantized grid
        """
        return cell_size * self.points + (cell_size // 2, cell_size // 2)

    def shift_point(self, point, cell_size):
        return cell_size * point + (cell_size // 2, cell_size // 2)