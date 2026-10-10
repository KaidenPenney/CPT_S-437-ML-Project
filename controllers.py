import pygame


class KeyboardControls:
    def get_steering(self):
        keys = pygame.key.get_pressed()
        steering = 0.0
        if keys[pygame.K_LEFT]:
            steering = -1.0
        elif keys[pygame.K_RIGHT]:
            steering = 1.0
        return steering
