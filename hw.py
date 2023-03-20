"""
Question 2.1:
Loop through the first ten files 
accumulating a running total of 
the number of times the first
character in the filename 
is equal to 'e' or 's' or
'j'. Then apply the formula
to calculate prior prob.
"""
import os

NUM_FILES = 30
NUM_LABELS = 3
NUM_CHARS = 27
ALPHA = 0.5

files = [file for file in os.listdir('data') if len(file) == 6]

for lang in ['English', 'Spanish', 'Japanese']:

    # Only consider those files with the first 
    # character the same as the first character 
    # of the current lang being considered (ex: 'e')
    # and with filename of length 6 (ex: 'e5.txt')
    file_count = len( [file for file in files if \
                       file[0] == lang[0].lower()] )

    # Apply formula for prior prob
    prior_p = (file_count + ALPHA) / (NUM_FILES + NUM_LABELS * ALPHA)
    prior_e = prior_p if lang == 'English' else None
    prior_s = prior_p if lang == 'Spanish' else None
    prior_j = prior_p if lang == 'Japanese' else None
    # print(f'Prior probability for {lang}: {prior_p}')
        

"""
Question 2.2:
Estimate class conditional probabilities 
for English.
"""
import numpy as np

valid_chars = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
char_counts = np.zeros(len(valid_chars))
english_files = [file for file in files if file[0] == 'e']
spanish_files = [file for file in files if file[0] == 's']
japanese_files = [file for file in files if file[0] == 'j']
assert len(valid_chars) == 27 and len(char_counts) == 27 and len(english_files) == 10

def cond_probs(lang, lang_files):
    for lang_file in lang_files:
        with open(os.path.join('data', lang_file)) as f:
            file_txt = f.read()
            # Add the character counts in the file
            for c_idx, char in enumerate(valid_chars):
                file_char_count = file_txt.count(char)
                char_counts[c_idx] = char_counts[c_idx] + file_char_count

    # Compute cond_probs
    numer = char_counts + ALPHA # array
    total_chars = char_counts.sum() 
    denom = total_chars + ALPHA * NUM_CHARS # float
    cond_probs = numer / denom
    # print(f'Conditional probabilities for {lang}:\n', cond_probs)
    return cond_probs

theta_eng = cond_probs('English', english_files)

"""
Question 2.3:
Estimate class conditional probabilities 
for Spanish and Japanese.
"""
theta_span = cond_probs('Spanish', spanish_files)
theta_jap = cond_probs('Japanese', japanese_files)


def test(filename='e10.txt'):
    """
    Question 2.4:
    Treat e10.txt as a test document x. 
    Represent x as a bag-of-words count 
    vector (Hint: the vocabulary has
    size 27). Print the bag-of-words vector x.
    """
    e10_path = './data/' + filename
    e10_bag = np.zeros(len(valid_chars), dtype=np.int32)
    with open(e10_path) as f:
        e10_text = f.read()
        for c_idx, char in enumerate(valid_chars):
            e10_bag[c_idx] = e10_text.count(char)
    # Normalize e10_bag for values in [0, 1]
    e10_bag = e10_bag / e10_bag.sum()


    """
    Question 2.5:
    compute ^p(x | y) for y = e, j, s 

    Formula from my piazza post:
    \hat p(x | y) = {e}^{x_1 \ln( \theta_{1, y}) + x_2 \ln( \theta_{2, y}) + . . . }
    """
    import math

    # English
    exp_sum_eng = 0 
    for c_idx in range(len(valid_chars)):
        exp_sum_eng += e10_bag[c_idx] * math.log(theta_eng[c_idx]) # x_1  * ln(\theta_{1, y})
    # p_x_giv_eng = math.e**exp_sum_eng
    # print(f'p(x | y=e) = e^{exp_sum_eng}')

    # Spanish
    exp_sum_sp = 0 
    for c_idx in range(len(valid_chars)):
        exp_sum_sp += e10_bag[c_idx] * math.log(theta_span[c_idx]) # x_1  * ln(\theta_{1, y})
    # p_x_giv_span = math.e**exp_sum_sp
    # print(f'p(x | y=s) = e^{exp_sum_sp}')

    # Japanese
    exp_sum_jap = 0 
    for c_idx in range(len(valid_chars)):
        exp_sum_jap += e10_bag[c_idx] * math.log(theta_jap[c_idx]) # x_1  * ln(\theta_{1, y})
    # p_x_giv_jap = math.e**exp_sum_jap
    # print(f'p(x | y=j) = e^{exp_sum_jap}')

    # Since every probability raises e to the power of
    # a very large  negative number, you can find the 
    # max probability with the largest exponent (smallest 
    # negative num)
    if exp_sum_eng == max([exp_sum_eng, exp_sum_sp, exp_sum_jap]):
        return 'e'
    if exp_sum_sp == max([exp_sum_eng, exp_sum_sp, exp_sum_jap]):
        return 's'
    if exp_sum_jap == max([exp_sum_eng, exp_sum_sp, exp_sum_jap]):
        return 'j'
    
    raise Exception("Failed to find max probability")

"""
Question 2.6:
Use Bayes rule and your estimated 
prior and likelihood, compute the posterior
^p(y | x) for y in {e,s,j}

P(A | B) = ( P(B | A) * P(A) ) / P(B)
P(y | x) = ( P(x | y) * P(y) ) / P(x)

This part was completed on the Latex file instead.
"""


"""
Question 2.7: 
Test the classifier on files 10-19 
for all 3 languages. 
"""
correct_e, correct_s, correct_j = (0,0,0)
test_files = [file for file in os.listdir('data') if len(file) == 7]
for f in test_files:
    lang = test(filename=f)
    if lang == f[0]:
        if f[0] == 'e': correct_e += 1
        if f[0] == 's': correct_s += 1
        if f[0] == 'j': correct_j += 1
print(f'correct_e: {correct_e}, correct_s: {correct_s}, correct_j: {correct_j}')