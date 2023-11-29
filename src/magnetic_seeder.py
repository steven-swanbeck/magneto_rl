#!/usr/bin/env python3
# %%
import numpy as np
import cv2
import random
from scipy import signal

# %%
class MagneticSeeder (object):
    def __init__ (self, width=5, height=5):
        self.wall_width = width # m
        self.wall_height = height # m
        self.pixel_density = 0.01 # (m)^2/pixel (ie pixel side length)
        self.im_width = int(self.wall_width / self.pixel_density) # pixels
        self.im_height = int(self.wall_height / self.pixel_density)  # pixels
        self.map = None
        
    def generate_map (self, num_seeds=5):
        blank_image = np.zeros((2 * self.im_height, 2 * self.im_width))
        
        seed_locations = []
        for ii in range(num_seeds):
            seed_locations.append(np.array([random.randint(0 + 250, 2 * self.im_height - 250), random.randint(0 + 250, 2 * self.im_width - 250)]))
        
        seeded_image = self.seed_gaussian_decay(blank_image, seed_locations)
        map = self.clip_map(seeded_image)
        
        self.map = self.convert_mask_to_image(map)
        
        converted_seeds = []
        for ii in range(len(seed_locations)):
            converted_seeds.append(self.image_to_cartesian_coordinates(seed_locations[ii]))
        
        return self.map, converted_seeds, map.astype(np.uint8)
        
    def seed_gaussian_decay (self, image, seeds):
        for seed in seeds:
            N = 249   # kernel size
            k1d = signal.gaussian(N, std=30).reshape(N, 1)
            kernel = np.outer(k1d, k1d)

            image[seed[0]-(N//2):seed[0]+(N//2)+1, seed[1]-(N//2):seed[1]+(N//2)+1] += 255 * kernel
        
        for ii in range(image.shape[0]):
            for jj in range(image.shape[1]):
                if image[ii, jj] > 255:
                    image[ii, jj] = 255
                    
        # - safe zone for robot
        N = 501   # kernel size
        k1d = signal.gaussian(N, std=30).reshape(N, 1)
        kernel = np.outer(k1d, k1d)
        image[250 * 2 - (N//2): 250 * 2 + (N//2) + 1, 250 * 2 - (N//2): 250 * 2 + (N//2) + 1] -= 255 * kernel
        
        for ii in range(image.shape[0]):
            for jj in range(image.shape[1]):
                if image[ii, jj] < 0:
                    image[ii, jj] = 0

        return image
    
    def clip_map (self, map):
        clip = np.zeros((self.im_height, self.im_width))
        clip[:, :] = map[int((map.shape[0] - self.im_height) / 2): int((map.shape[0] - self.im_height) / 2 + self.im_height), int((map.shape[1] - self.im_width) / 2): int((map.shape[1] - self.im_width) / 2 + self.im_width)]
        return clip
    
    def convert_mask_to_image (self, image):
        self.single_channel_map = image
        blank_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        blank_image[:,:,0] = image
        blank_image[:,:,1] = image
        blank_image[:,:,2] = image
        return blank_image
    
    def save_map_as_image (self):
        raise NotImplementedError
    
    def transform_image_into_pygame (self, image):
        trans = np.rot90(np.flip(image, axis=0), k=3)
        trans[:, :, 1] = 0
        trans[:, :, 2] = 0
        return trans
    
    def cartesian_to_image_coordinates (self, coords):
        output = np.array([
            int(coords[0] * (self.im_height / (2 * self.wall_width)) + self.im_height / 2),
            int(coords[1] * (self.im_width / (2 * self.wall_height)) + self.im_width / 2),
        ])
        for ii in range(2):
            if output[ii] > 499:
                output[ii] = 499
            elif output[ii] < 0:
                output[ii] = 0
        return output
    
    def image_to_cartesian_coordinates (self, coords):
        output = np.array([
            (2 * self.wall_width / self.im_height) * (coords[0] - self.im_height / 2),
            (2 * self.wall_height / self.im_width) * (coords[1] - self.im_width / 2),
        ])
        return output
    
    def cartesian_to_pygame_coordinates (self, coords):
        output = np.array([
            int(coords[1] * (self.im_width / (2 * self.wall_width)) + self.im_width / 2),
            int(coords[0] * (self.im_height / (2 * self.wall_height)) + self.im_height / 2),
        ])
        for ii in range(2):
            if output[ii] > 499:
                output[ii] = 499
        return output
    
    def lookup_magnetism_modifier (self, coords):
        if self.map is None:
            return 1.
        pixel_coords = self.cartesian_to_image_coordinates(coords)
        # return float(255 - self.map[pixel_coords[0], pixel_coords[1]][0]) / 255
        modifier = float(255 - self.map[pixel_coords[0], pixel_coords[1]][0]) / 255
        if modifier > 0.7:
            modifier = 1.
        return modifier
