

import pygame
import random
import sys

class CGame(object):
    def __init__ (self):
        pygame.init()
        self.GAME_COLOR = (255,255,0)
        self.COLOR_BLACK = (0,0,0)
        self.CATCHER_WIDTH = 98
        self.CATCHER_HEIGHT = 30
        self.GAME_WIDTH = 500
        self.GAME_HEIGHT = 500
        self.BALL_SPEED = 10
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
    def reset(self):
        self.CATCHER_X = 200 # CATCHER is at the bottom
        self.CATCHER_Y = 470
        self.BALL_X, self.ZONE = self.returnXpos() # zone = 0,100,200,300,or 400
        self.BALL_Y = 20
        self.GAME_OVER = False
        self.REWARD = 0
        
        self.screen = pygame.display.set_mode((self.GAME_WIDTH,self.GAME_HEIGHT))
        pygame.display.set_caption('RL Catch Game CPEG591')
        self.clock = pygame.time.Clock()
        self.catcher_rect = pygame.draw.rect(self.screen,self.GAME_COLOR,pygame.Rect(self.CATCHER_X,self.CATCHER_Y,self.CATCHER_WIDTH,self.CATCHER_HEIGHT))
    
    def mainGame(self):
        pygame.event.pump()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.CATCHER_X  -= 100
                    if self.CATCHER_X < 0:
                        self.CATCHER_X = 0
                elif event.key == pygame.K_RIGHT:
                    self.CATCHER_X += 100
                    if self.CATCHER_X > 499:
                        self.CATCHER_X = 400

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
        
        print("Reward : %d"%(self.REWARD))
        self.screen.fill(self.COLOR_BLACK)
        self.catcher_rect = pygame.draw.rect(self.screen,self.GAME_COLOR,pygame.Rect(self.CATCHER_X,self.CATCHER_Y,self.CATCHER_WIDTH,self.CATCHER_HEIGHT))
        self.circle = pygame.draw.circle(self.screen,self.GAME_COLOR,(self.BALL_X,self.BALL_Y),15,0)
        pygame.display.flip()
        self.clock.tick(10) # 10-60 , higher number goes faster
        return self.GAME_OVER

if __name__ == "__main__":
    game = CGame()
    game.reset()

    game_over = False
    while not game_over:
        game_over = game.mainGame()