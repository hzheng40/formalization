# evaluation metrics
from nltk.translate.bleu_score import sentence_bleu

# formality
# fluency
# meaning perservation

# bleu
def getBleuScore(reference, candidate):
    # reference: a list of lists of str
    # candidate: a list of str
    return sentence_bleu(reference, candidate)