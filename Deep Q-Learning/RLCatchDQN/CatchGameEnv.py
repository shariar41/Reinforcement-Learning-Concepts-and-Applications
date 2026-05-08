import pygame
import random
import collections
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class CatchGameEnv(object):
    def __init__ (self):
        pygame.init()
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self.GAME_COLOR = (255,255,0)
        self.COLOR_BLACK = (0,0,0)
        self.CATCHER_WIDTH = 98
        self.CATCHER_HEIGHT = 30
        self.GAME_WIDTH = 500
        self.GAME_HEIGHT = 500
        self.BALL_SPEED = 30


    def returnXpos(self): # ball will start at a random x pos from top
        number = random.randint(0,4) # 5 possible positions for ball, catcher
        if number == 0: # function returns xposition of ball, and zone
            return 47, 0 # for catcher so that we can determine if ball
        elif number == 1: # hits the catcher or not, zone=0,100,200,300 or 400
            return 147,100
        elif number == 2:
            return 247,200
        elif number == 3:
            return 347,300
        else:
            return 447,400

    def returnFrames(self):
        return np.array(list(self.frames))
    def reset(self):
        self.CATCHER_X = 200
        self.CATCHER_Y = 470
        self.BALL_X, self.ZONE = self.returnXpos()
        self.BALL_Y = 20
        self.GAME_OVER = False
        self.REWARD = 0
        self.frames = collections.deque(maxlen=4)
        self.screen = pygame.display.set_mode((self.GAME_WIDTH,self.GAME_HEIGHT))
        pygame.display.set_caption('RL Catch Game CPEG 591')
        self.clock = pygame.time.Clock()
        self.catcher_rect = pygame.draw.rect(self.screen,self.GAME_COLOR,pygame.Rect(self.CATCHER_X,self.CATCHER_Y,self.CATCHER_WIDTH,self.CATCHER_HEIGHT))

    def mainGame(self,action):
        pygame.event.pump()
        if action == 1: # action = 1 move left
            self.CATCHER_X -= 100
            if self.CATCHER_X < 0:
                self.CATCHER_X = 0
        elif action == 2: # action = 2 move right
            self.CATCHER_X += 100
            if self.CATCHER_X > 499:
                self.CATCHER_X = 400
        else: # action = 0 do nothing
            pass
        self.BALL_Y += self.BALL_SPEED
        # Create catcher rect (IMPORTANT: update every frame)
        self.catcher_rect = pygame.Rect(
            self.CATCHER_X,
            self.CATCHER_Y,
            self.CATCHER_WIDTH,
            self.CATCHER_HEIGHT
        )

        # Reward logic
        if self.BALL_Y >= self.CATCHER_Y:
            if self.catcher_rect.left == self.ZONE:
                self.REWARD = 1
            else:
                self.REWARD = -1
            self.GAME_OVER = True

        self.screen.fill(self.COLOR_BLACK)
        self.catcher_rect = pygame.draw.rect(self.screen,self.GAME_COLOR,pygame.Rect(self.CATCHER_X,self.CATCHER_Y,self.CATCHER_WIDTH,self.CATCHER_HEIGHT))
        self.circle = pygame.draw.circle(self.screen,self.GAME_COLOR,(self.BALL_X,self.BALL_Y),15,0)
        pygame.display.flip()
        self.frames.append(pygame.surfarray.array2d(self.screen))
        self.clock.tick(30)
        return self.GAME_OVER, self.returnFrames(), self.REWARD

if __name__ == "__main__":
    num_wins = 0
    frames = np.array([])
    for i in range(100):
        game = CatchGameEnv()
        game.reset()
        game_over = False
        while not game_over:
            action = random.randint(0,2)
            game_over, frames, reward = game.mainGame(action)
            if reward == 1:
                num_wins +=1
                print('*****win count=',num_wins, ' games played=',i)
            # for frame in frames: # for visualization of images in the game
            # img = Image.fromarray(frame)
            # x_t = img.resize(size=(100, 100))
            # x_t = np.array(x_t).T # need transpose
            # x_t = x_t.astype('float')/np.max(x_t)
            # plt.imshow(x_t)
            # plt.show()
            # a = 5