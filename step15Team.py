# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import math


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """
    oldQ = {
        "Defender": {
            "Stop": 0,
            "North": 0,
            "East": 0,
            "West": 0,
            "South": 0
        },
        "Offender": {
            "Stop": 0,
            "North": 0,
            "East": 0,
            "West": 0,
            "South": 0
        }
    }

    weightsList = {
            "Offender": {
                "Stop": {
                    'successorFood': 50,
                    'successorScore': 1000,
                    'capsuleScore': 6,
                    'distanceToFood': -4,
                    'distanceToGhost': 7,
                    'isPacman': 4,
                    'steps': -3
                },
                "North": {'successorScore': 997.9721165869875, 'distanceToFood': -6.02788341301295, 'distanceToGhost': 5.957398902084358, 'steps': -5.027883413012955, 'successorFood': 47.97211658698685, 'isPacman': 1.9721165869870452, 'capsuleScore': 3.9721165869870485},
                "South": {'successorScore': 998.5687654724113, 'distanceToFood': -5.431234527588932, 'distanceToGhost': 6.499481923263105, 'steps': -4.43123452758894, 'successorFood': 48.56876547241111, 'isPacman': 2.5687654724110636, 'capsuleScore': 4.568765472411071},
                "West": {'successorScore': 999.5273767691459, 'distanceToFood': -4.4726232308542615, 'distanceToGhost': 7.047417599670629, 'steps': -3.4726232308542615, 'successorFood': 49.527376769145725, 'isPacman': 3.52737676914574, 'capsuleScore': 5.527376769145741},
                "East": {'successorScore': 998.8749140472075, 'distanceToFood': -5.125085952792434, 'distanceToGhost': 6.147660735703763, 'steps': -4.12508595279243, 'successorFood': 48.87491404720757, 'isPacman': 2.874914047207577, 'capsuleScore': 4.87491404720757}
            },
            "Defender": {
                "East": {'numInvaders': -997, 'onDefense': 102.5, 'invaderDistance': -10, 'stop': -97.5, 'reverse': -1},
                "West": {'numInvaders': -997, 'onDefense': 102.5, 'invaderDistance': -9, 'stop': -97.5, 'reverse': -1},
                "North": {'numInvaders': -997, 'onDefense': 105, 'invaderDistance': -8, 'stop': -97.5, 'reverse': -1},
                "South": {'numInvaders': -997, 'onDefense': 102.5, 'invaderDistance': -9, 'stop': -99, 'reverse': -1.5},
                "Stop": {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}
            }
        }
    prevAction = None
    repeatActionCount = 0
    randomPath = []
    actionsTakenOffensive = 0
    actionsTakenOffensivePrev = 0
    myPrevScore = 0
    oppositeDir = {"North": "South", "South": "North", "East": "West", "West": "East", "Stop": "Stop"}
    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        if actions != ["Stop"]:
            actions.remove("Stop")

        # TODO: get random next step for each action
        values = []
        oppositeDir = {"North": "South", "South": "North", "East": "West", "West": "East", "Stop": ""}


        iter = 0
        while iter < 3:
            action_id = 0
            for action in actions:
                # dict = {}
                successor1 = self.getSuccessor(gameState, action)
                next_action1 = random.choice([x for x in successor1.getLegalActions(self.index) if x != oppositeDir[action]])
                successor2 = self.getSuccessor(successor1, next_action1)
                next_action2 = random.choice([x for x in successor2.getLegalActions(self.index) if x != oppositeDir[next_action1]])
                successor3 = self.getSuccessor(successor2, next_action2)
                next_action3 = random.choice([x for x in successor3.getLegalActions(self.index) if x != oppositeDir[next_action2]])
                successor4 = self.getSuccessor(successor3, next_action3)
                next_action4 = random.choice([x for x in successor4.getLegalActions(self.index) if x != oppositeDir[next_action3] ])
                successor5 = self.getSuccessor(successor4, next_action4)
                next_action5 = random.choice([x for x in successor5.getLegalActions(self.index) if x != oppositeDir[next_action4]])
                successor6 = self.getSuccessor(successor5, next_action5)
                next_action6 = random.choice([x for x in successor6.getLegalActions(self.index) if x != oppositeDir[next_action5]])
                successor7 = self.getSuccessor(successor6, next_action6)
                next_action7 = random.choice([x for x in successor7.getLegalActions(self.index) if x != oppositeDir[next_action6]])
                successor8 = self.getSuccessor(successor7, next_action7)
                next_action8 = random.choice([x for x in successor8.getLegalActions(self.index) if x != oppositeDir[next_action7]])
                successor9 = self.getSuccessor(successor8, next_action8)
                next_action9 = random.choice([x for x in successor9.getLegalActions(self.index) if x != oppositeDir[next_action8]])
                successor10 = self.getSuccessor(successor9, next_action9)
                next_action10 = random.choice([x for x in successor10.getLegalActions(self.index) if x != oppositeDir[next_action9]])
                successor11 = self.getSuccessor(successor10, next_action10)
                next_action11 = random.choice([x for x in successor11.getLegalActions(self.index) if x != oppositeDir[next_action10]])
                successor12 = self.getSuccessor(successor11, next_action11)
                next_action12 = random.choice(
                    [x for x in successor12.getLegalActions(self.index) if x != oppositeDir[next_action11]])
                successor13 = self.getSuccessor(successor12, next_action12)
                next_action13 = random.choice(
                    [x for x in successor13.getLegalActions(self.index) if x != oppositeDir[next_action12]])
                successor14 = self.getSuccessor(successor13, next_action13)
                next_action14 = random.choice(
                    [x for x in successor14.getLegalActions(self.index) if x != oppositeDir[next_action13]])
                successor15 = self.getSuccessor(successor14, next_action14)
                next_action15 = random.choice(
                    [x for x in successor15.getLegalActions(self.index) if x != oppositeDir[next_action14]])
                successor16 = self.getSuccessor(successor15, next_action15)
                next_action16 = random.choice(
                    [x for x in successor16.getLegalActions(self.index) if x != oppositeDir[next_action15]])
                successor17 = self.getSuccessor(successor16, next_action16)
                next_action17 = random.choice(
                    [x for x in successor17.getLegalActions(self.index) if x != oppositeDir[next_action16]])
                successor18 = self.getSuccessor(successor17, next_action17)
                next_action18 = random.choice(
                    [x for x in successor18.getLegalActions(self.index) if x != oppositeDir[next_action17]])
                successor19 = self.getSuccessor(successor18, next_action18)
                next_action19 = random.choice(
                    [x for x in successor19.getLegalActions(self.index) if x != oppositeDir[next_action18]])
                successor20 = self.getSuccessor(successor19, next_action19)
                next_action20 = random.choice(
                    [x for x in successor20.getLegalActions(self.index) if x != oppositeDir[next_action19]])

                sum = (
                        self.evaluate(gameState, action)
                        + self.evaluate(successor1, next_action1) * 0.9
                        + self.evaluate(successor2, next_action2) * 0.8
                        + self.evaluate(successor3, next_action3) * 0.7
                        + self.evaluate(successor4, next_action4) * 0.6
                        + self.evaluate(successor5, next_action5) * 0.5
                        + self.evaluate(successor6, next_action6) * 0.4
                        + self.evaluate(successor7, next_action7) * 0.3
                        + self.evaluate(successor8, next_action8) * 0.2
                        + self.evaluate(successor9, next_action9) * 0.1
                        + self.evaluate(successor10, next_action10) * 0.09
                        + self.evaluate(successor11, next_action11) * 0.08
                        + self.evaluate(successor12, next_action12) * 0.07
                        + self.evaluate(successor13, next_action13) * 0.06
                        + self.evaluate(successor14, next_action14) * 0.05
                        + self.evaluate(successor15, next_action15) * 0.04
                        + self.evaluate(successor16, next_action16) * 0.03
                        + self.evaluate(successor17, next_action17) * 0.02
                        + self.evaluate(successor18, next_action18) * 0.01
                        + self.evaluate(successor19, next_action19) * 0.009
                        + self.evaluate(successor20, next_action20) * 0.008
                )

                if iter == 0:
                    values.append(sum / 5)
                else:
                    values[action_id] += sum / 5
                action_id += 1
            iter += 1
        self.actionsTakenOffensivePrev += 1
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        foodLeft = len(self.getFood(gameState).asList())
        bestAction = random.choice(bestActions)
        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction
        # update Q table when start a new state
        if self.getPreviousObservation() is not None:
            self.updateWeights(gameState, bestAction, maxValue)
        # record to old dict before end state
        self.oldQ["Offender" if gameState.getAgentState(self.index).isPacman else "Defender"][bestAction] = self.evaluate(gameState, bestAction)
        # record action to prev action
        self.repeatActionCount = self.repeatActionCount + 1 if oppositeDir[bestAction] == self.prevAction else 0
        self.prevAction = bestAction
        self.actionsTakenOffensive += 1
        return bestAction


    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)

        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}



class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()

        if successor.getAgentState(self.index).isPacman:

            # get ghost and find min dist
            # don't choose stop (they will def die if they stop)
            distance = [100]
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
            for a in enemies:
                if a.isPacman or a.scaredTimer > 10:
                    distance.append(100)
                elif not a.isPacman and a.getPosition() is not None and a.scaredTimer < 10:
                    distance.append(self.getMazeDistance(myPos, a.getPosition()))
            features['distanceToGhost'] = min(distance)


        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        features['successorFood'] = -len(foodList)  # self.getScore(successor)

        # what about score? - if close to border than get score!!! CAN'T PUT IT TO BORDER!!
        if self.getScore(successor) > self.getScore(gameState):
            features['successorScore'] = self.getScore(successor) - self.getScore(gameState)
            # self.myPrevScore += features['successorScore']
        else:
            features['successorScore'] = 0
        # print "features['successorScore']", features['successorScore']

        # features['foodLeft'] = len(self.getFood(successor).asList())

        # what about capsules?
        succ_pos = successor.getAgentPosition(self.index)
        cap_score = 1 if succ_pos in self.getCapsules(gameState) else 0
        features['capsuleScore'] = cap_score

        # it is important to be a pacman, or else it just keep running away from ghost
        features['isPacman'] = 1 if successor.getAgentState(self.index).isPacman else 0

        # features['steps'] = self.actionsTakenOffensive

        return features

    def updateWeights(self, gameState, action, newQ):
        alpha = 0.000001
        if self.getPreviousObservation is not None:
            for featureWeight in self.weightsList["Offender"][action]:
                self.weightsList["Offender"][action][featureWeight] = \
                    self.weightsList["Offender"][action][featureWeight] + \
                    alpha * ((newQ - self.oldQ["Offender"][action])) \
                    if self.oldQ != 0 \
                    else self.weightsList["Offender"][action][featureWeight]
                            # + alpha * (- 0.1 * len(self.getFood(gameState).asList()) + \
                            # 0.1 * len(self.getFood(self.getPreviousObservation()).asList()))


    def getWeights(self, gameState, action):
        # print action, self.weightsList["Offender"][action]
        if gameState.isOnRedTeam(self.index):
            return self.weightsList["Offender"][action]
        else:
            return self.weightsList["Offender"][self.oppositeDir[action]]

    def deepcopy(self, num):
        res = [num]
        return res[0]

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        if Directions.STOP in actions:
            actions.remove(Directions.STOP)

        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)