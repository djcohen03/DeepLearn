import os
import sys
import random
import pygame
import numpy as np
from pygame.locals import *

class Game(object):

    def __init__(self):
        self.board = Board()
        self.play()

    def play(self, automated=False):
        self.score = 0
        self.board.randomfill()
        while True:
            try:
                self.board.randomfill()
                self.board.logboard()
                if automated:
                    raise Exception("Automation not yet enabled")
                    # result = False
                    # while result == False:
                    #     result = self.board.move('left')
                else:
                    result = False
                    while result is False:
                        result = self.board.move(self.getmove())
                    self.score += int(result)
                if not self.board.moveexists():
                    raise GameOver
            except GameOver:
                break

        self.gameover()

    def getmove(self):
        direction = raw_input("%s Please Enter a Move: " % self.score)
        moves = {
            '\x1b[A': 'up',
            '\x1b[B': 'down',
            '\x1b[C': 'right',
            '\x1b[D': 'left',
            'exit': 'end',
            'q': 'end',
            'quit': 'end'
        }
        value = moves.get(direction)
        if value:
            if value == 'end':
                raise GameOver
            else:
                return value
        else:
            print "Invalid Response"
            return self.getmove()

    def gameover(self):
        print "Game Over! Final Score: %s" % self.score
        if raw_input("Play Again? y/N: ").lower() == 'y':
            self.board = Board()
            self.play()
        else:
            print "Thanks for playing!"


class Board(object):
    def __init__(self, render=False):
        self.slots = np.matrix([[0] * 4] * 4)
        self.render = render
        if render:
            self.display = Display()

    def _move_up(self, slots):
        # Keep track of which cells have already joined:
        joined = np.matrix([[False] * 4] * 4)
        foundmove = False
        score = 0

        for x in range(4):
            for y in range(4):
                item = slots[x, y]
                if item == 0:
                    continue

                maxdeltax = 0
                for delta in range(1, x + 1):
                    neighbor = slots[x - delta, y]
                    if neighbor == 0:
                        # There is a vacancy
                        maxdeltax = delta
                    else:
                        # There is a blocker
                        break

                newx, newy = (x - maxdeltax, y)

                if x != newx:
                    # Shift our item over to the max delta
                    slots[x, y] = 0
                    slots[newx, y] = item
                    foundmove = True

                if newx > 0:
                    # Check if we can merge into neighbor
                    neighbor = slots[newx - 1, y]
                    if neighbor == item:
                        if not joined[newx - 1, y]:
                            # Cell values are equal, and this cell hasn't been joined yet
                            newvalue = item * 2
                            score += newvalue
                            slots[newx, newy] = 0
                            slots[newx - 1, newy] = newvalue
                            joined[newx - 1, y] = True
                            foundmove = True

        return slots, foundmove, score

    def moveexists(self):
        for k in range(4):
            _, foundmove, _ = self._move_up(np.rot90(self.slots, k=k).copy())
            if foundmove:
                return True
        return False

    def move(self, direction):
        ''' Move board corresponding to user's input direction
        '''
        # Maybe rotate/flip slots matrix
        k = {
            'up': 0,
            'right': 1,
            'down': 2,
            'left': -1,
        }.get(direction)
        slots = np.rot90(self.slots, k=k).copy()

        # Try to shift all pieces up:
        slots, foundmove, score = self._move_up(slots)

        # Check if we found a move or not:
        if not foundmove:
            return False

        self.slots = np.rot90(slots, k=-k)
        return score

    def logboard(self):
        if self.render:
            self.display.render(self.slots)
        else:
            # os.system('clear')
            print self.slots
            print "-------------------"

    def randomfill(self):
        empties = self.empties()
        index = random.randint(0, len(empties) - 1)
        (x,y) = empties[index]
        value = self.two_or_four()
        self.slots[x, y] = value
        return (x, y), value

    def empties(self):
        coords = np.argwhere(self.slots == 0)
        if not len(coords):
            raise GameOver
        return coords

    @classmethod
    def two_or_four(cls):
        return np.random.choice([2, 4], p=[0.9, 0.1])



class Display():

    red = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)

    def __init__(self, width=400, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height), 0, 32)

    def render(self, matrix):
        print matrix
        # draw grid:
        self.display.fill(self.white)
        for i in range(1, 4):
            x = int(self.width / 4.0 * i)
            y = int(self.height / 4.0 * i)
            pygame.draw.line(self.display, self.black, (x, 0), (x, self.height), 1)
            pygame.draw.line(self.display, self.black, (0, y), (self.width, y), 1)

        # Draw tiles:
        for x in range(4):
            for y in range(4):
                value = matrix[y, x]
                if value:
                    pygame.draw.rect(self.display, self.red, (x * 100, y * 100, 100, 100))
                    # number =
        # pygame.display.set_caption('Score: %s' % score)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()




class GameOver(Exception):
    pass


if __name__ == '__main__':
    Game()
