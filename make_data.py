from typing import Optional
from itertools import product
import math
import random
import numpy as np


def d1(
        order: list, #list of question order EX: [2,3,4,1,5] for 5 questions
        goodness_scores=None, #list of specific goodness scores from real data (rather than random)
        rng=None, #Seed for the random list of public probabilities 
):
    """ Takes a sequence of given or randomely generated likeability scores, and updates subsequent scores 
    dependent on the current score such that it is either raised or lowered by the average difference 
    between the two scores - thus making outcomes "more even". This rule, known as "even-handedness", typically 
    comes into play with questions containing similar objects; such as "Do you have a good opinion of Bill Clinton?" and "Do you
    have a good opinion of Al Gore?" The probability of asnwering "yes" to one of these questions depends on which one is asked first
    and the likeability scores of the individuals. This is described more in detail in Ref. https://www.jstor.org/stable/3078697. 

    :param order: list of question order EX: [2,3,4,1,5] for 5 questions
    :param goodness_scores: Input scores on the "goodness" of a person. If none, scores are random.
    :param rng: Numpy random number generator.
    :return: The probability distribution over bitstrings where "000" corresponds to "no, no, no".
    """ 
    n_questions = len(order)
    if goodness_scores == None:
        goodness_scores = []
        for i in range(0, n_questions):
            n = rng.random()
            goodness_scores.append(n)

    scorelist = [goodness_scores[i] for i in order]

    for count, (i, j) in enumerate(zip(scorelist[:-1], scorelist[1:])):
        mean_diff = (i - j) / 2
        if mean_diff < 0:
            scorelist[count + 1] = j + mean_diff
        if mean_diff > 0:
            scorelist[count + 1] = j + mean_diff
        else:
            pass

    bin_str = [''.join(p) for p in product('10', repeat=n_questions)]
    bin_str.sort(key=lambda s: s.count('1'))
    prob_dist = {}

    for string in bin_str:
        total_prob = 1
        for (s, score) in zip(string, scorelist):
            if s == "0":
                prob = 1 - score
            if s == "1":
                prob = score
            total_prob = total_prob * prob
        prob_dist[string] = total_prob

    return np.asarray(list(prob_dist.values()))

def d2(
    order_input: list, #list of question order EX: [2,3,4,1,5] for 5 questions
    rescale_coefficient: float, #portion amount to raise or lower the probabilitiy 
    seed = None, #Seed for the random list of public probabilities 
) -> dict: 
    """ Takes a sequence of general to specific ranking questions with a corresponding random list of probabilities for public answers. If the order is changed
    such that a more specific question preceeds a more general question, the answer to the more general question will obtain a 45% increase in the probability that the 
    answer is yes. This dataset is based off of real data regarding school bullying, where we see 45% increases of students answering yes to being bullied
    if they are asked about a specific type of bullying first. This is found in the study: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5965535/#:~:text=A%20randomized%20experiment%20(n%20%3D%205%2C951,several%20widely%20used%20bullying%20surveys.

        :param order_input: list of question orders starting at an index of zero. 
        :param seed: random seed if the goodness_scores are randomely generated. 
        :param rescale_coefficient: portion amount to raise or lower the probabilitiy 
        :return:  The probability distribution over bitstrings where "000" corresponds to "no, no, no".
    """
    n_questions = len(order_input)
    question_specifications = [i for i in np.arange(0, 1, 1 / n_questions)] #General is Low, Specific is High
    group_answers = []
    random.seed(seed)
    for i in range(0,n_questions):
        n = random.random() #0 is No, 1 is Yes
        group_answers.append(n)


    question_specifications = [question_specifications[i] for i in order_input]
    for count, (i,j) in enumerate(zip(question_specifications[:-1], question_specifications[1:])): 
        if i > j: 
            group_answers[count + 1] +=  (group_answers[count+1]) * rescale_coefficient
        else: 
            pass
    bin_str = [''.join(p) for p in product('10', repeat= n_questions)]
    bin_str.sort(key=lambda s: s.count('1'))
    prob_dist = {}
    
    group_answers = group_answers / np.sum(group_answers)
    for string in bin_str: 
        total_prob = 1 
        for (s, score) in zip(string, group_answers): 
            if s == "0": 
                prob = 1 - score
            if s == "1": 
                prob = score 
            total_prob = total_prob * prob
        prob_dist[string] = total_prob

    return np.asarray(list(prob_dist.values()))

def get_data_noncommute_score(order_distributions):    
    for i, j in zip(order_distributions[:-1], order_distributions[1:]):
        diff = i - j 
    return np.sum(diff * diff) / len(order_distributions)


