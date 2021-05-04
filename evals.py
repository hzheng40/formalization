from nltk.parse import CoreNLPParser, CoreNLPDependencyParser
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from collections import defaultdict
from textblob import TextBlob
from nltk.tree import *
import pickle
import gensim
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.metrics import make_scorer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion

NER_TAGSET = ['LOCATION', 'PERSON', 'ORGANIZATION', 'MONEY', 'MISC', 'NUMBER', 'PERCENT', \
              'DATE', 'TIME', 'DURATION', 'SET', 'ORDINAL']
POS_TAGSET = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', \
              'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',\
              'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH',\
              'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',\
              'WDT', 'WP', 'WP$', 'WRB']
FP_PRO_LIST = ['i', 'we', 'me', 'us', 'my', 'mine', 'our', 'ours']
TP_PRO_LIST = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'her', 'hers', 'its', 'their', 'theirs']

"""
Formality: implementation of Pavlick and Tetreault 2016
modified based on https://github.com/raosudha89/style_transfer
"""

class StanfordAnnotations:
    # helper class
    def __init__(self, token, lemma, pos, ner, head, depRel):
        self.token = token
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        self.head = head
        self.depRel = depRel

    def __repr__(self):
        return self.token + ' ' + self.lemma + ' ' + self.pos + ' ' + self.ner + ' ' + str(self.head) + ' ' + self.depRel + '\n'

class FeatureExtractor:
    def __init__(self, w2v_path, corpus_dict_path, port=9000):
        # corenlp client
        self.parser = CoreNLPParser(url='http://localhost:' + str(port))
        self.dep_parser = CoreNLPDependencyParser(url='http://localhost:' + str(port))
        # w2v
        self.word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('data/saved_models/GoogleNews-vectors-negative300.bin', binary=True)
        print('w2v model loaded')
        # training corpus for one hot features
        corpus_dict = pickle.load(open(corpus_dict_path, 'rb'))

        self.dep_tuple_vectorizer = DictVectorizer(sparse=False)
        self.dep_tuple_vectorizer = self.dep_tuple_vectorizer.fit(corpus_dict['dep_tuple'])

        self.unigram_vectorizer = DictVectorizer(sparse=False)
        self.unigram_vectorizer = self.unigram_vectorizer.fit(corpus_dict['unigram'])

        self.bigram_vectorizer = DictVectorizer(sparse=False)
        self.bigram_vectorizer = self.bigram_vectorizer.fit(corpus_dict['bigram'])

        self.trigram_vectorizer = DictVectorizer(sparse=False)
        self.trigram_vectorizer = self.trigram_vectorizer.fit(corpus_dict['trigram'])

        self.lexical_vectorizer = DictVectorizer(sparse=False)
        self.lexical_vectorizer = self.lexical_vectorizer.fit(corpus_dict['lexical'])


    def _get_case_features(self, sent_annotations, sentence):
        num_all_caps = 0
        for word_annotations in sent_annotations:
            if word_annotations.token.isupper():
                num_all_caps += 1
        if sentence.islower():
            is_sent_lower = 1
        else:
            is_sent_lower = 0
        if sent_annotations[0].token.isupper():
            is_first_word_caps = 1
        else:
            is_first_word_caps = 0
        return [num_all_caps, is_sent_lower, is_first_word_caps]

    def _get_dependency_tuples(self, sent_annotations):
        # (gov, typ, dep)  (gov, typ)  (typ, dep)  (gov, dep)
        dependency_tuple_dict = defaultdict(int)
        for word_annotations in sent_annotations:
            gov = sent_annotations[int(word_annotations.head)-1].pos
            typ = word_annotations.depRel
            dep = word_annotations.pos
            gov_typ_dep = '_'.join([gov, typ, dep])
            dependency_tuple_dict[gov_typ_dep] = 1
            gov_typ = '_'.join([gov, typ])
            dependency_tuple_dict[gov_typ] = 1
            typ_dep = '_'.join([typ, dep])
            dependency_tuple_dict[typ_dep] = 1
            gov_dep = '_'.join([gov, dep])
            dependency_tuple_dict[gov_dep] = 1
        return dependency_tuple_dict

    def _get_entity_features(self, sent_annotations):
        ner_tags = [0]*len(NER_TAGSET)
        person_mentions_total_len = 0
        for word_annotations in sent_annotations:
            if word_annotations.ner == 'O':
                continue
            if word_annotations.ner not in NER_TAGSET:
                continue
            else:
                index = NER_TAGSET.index(word_annotations.ner)
                ner_tags[index] = 1
            if word_annotations.ner == 'PERSON':
                person_mentions_total_len += len(word_annotations.token)
        person_mentions_avg_len = person_mentions_total_len*1.0/len(sent_annotations)
        return ner_tags + [person_mentions_avg_len]

    def _get_lexical_features(self, words):
        num_contractions = 0
        total_word_len = 0
        for word in words:
            if '\'' in word:
                num_contractions += 1
            total_word_len += len(word)
        avg_num_contractions = num_contractions*1.0/len(words)
        avg_word_len = total_word_len*1.0/len(words)
        #TODO: avg word-log frequency acc to Google Ngram
        #TODO: avg formality score using Pavlick & Nenkova (2015)
        return [avg_num_contractions, avg_word_len]

    def _get_ngrams(self, sent_annotations):
        # tokens = [w.token for w in sent_annotations]
        tokens = [w.lemma for w in sent_annotations]
        sentence = ' '.join(tokens)
        # .decode('utf-8', 'ignore')
        blob = TextBlob(sentence)
        unigrams = tokens
        bigrams = blob.ngrams(n=2)
        trigrams = blob.ngrams(n=3)
        unigram_dict = defaultdict(int)
        bigram_dict = defaultdict(int)
        trigram_dict = defaultdict(int)
        for unigram in unigrams:
            unigram_dict[unigram] = 1
        for bigram in bigrams:
            bigram_dict['_'.join(bigram)] = 1
        for trigram in trigrams:
            trigram_dict['_'.join(trigram)] = 1
        return unigram_dict, bigram_dict, trigram_dict

    def _get_parse_features(self, stanford_parse_tree, sent_annotations):
        sent_len = len(sent_annotations)
        avg_depth = stanford_parse_tree.height()*1.0/sent_len
        lexical_production_dict = defaultdict(int)
        for production in stanford_parse_tree.productions():
            if production.is_lexical():
                continue
            lexical_production_dict[production] += 1
        avg_depth_feature = [avg_depth]
        return avg_depth_feature, lexical_production_dict

    def _get_POS_features(self, sent_annotations):
        pos_tag_ct = [0]*len(POS_TAGSET)
        for word_annotations in sent_annotations:
            try:
                pos_tag_ct[POS_TAGSET.index(word_annotations.pos)] += 1
            except:
                # print word_annotations.pos
                continue
        for i in range(len(pos_tag_ct)):
            pos_tag_ct[i] = pos_tag_ct[i]*1.0/len(sent_annotations)
        return pos_tag_ct

    def _get_punctuation_features(self, sentence):
        num_question_marks = sentence.count('?')
        num_ellipses = sentence.count('...')
        num_exclamations = sentence.count('!')
        return [num_question_marks, num_ellipses, num_exclamations]

    def _get_readability_features(self, sentence, words):
        num_words = len(words)
        num_chars = len(sentence) - sentence.count(' ')
        return [num_words, num_chars]

    def _get_subjectivity_features(self, sent_annotations, sentence):
        subjectivity_features = []
        fp_pros = 0
        tp_pros = 0
        for word_annotations in sent_annotations:
            if word_annotations.lemma in FP_PRO_LIST:
                fp_pros += 1
            if word_annotations.lemma in TP_PRO_LIST:
                tp_pros += 1
        subjectivity_features.append(fp_pros*1.0/len(sent_annotations))
        subjectivity_features.append(tp_pros*1.0/len(sent_annotations))
        polarity, subjectivity = TextBlob(sentence).sentiment
        subjectivity_features.append(float(np.sign(polarity)))
        subjectivity_features.append(subjectivity)
        return subjectivity_features

    def _get_word2vec_features(self, sent_annotations):
        word_vectors = []
        for word_annotations in sent_annotations:
            try:
                word_vector = self.word2vec_model[word_annotations.lemma]
                word_vectors.append(word_vector)
            except:
                # print word_annotations.token
                continue
        if len(word_vectors) == 0:
            avg_word_vectors = np.zeros(300)
        else:
            avg_word_vectors = np.transpose(np.mean(word_vectors, axis=0))
        return avg_word_vectors

    def _remove_less_frequent(self, dict, reference_dict, freq_cutoff):
        new_dict = defaultdict(int)
        for item,count in dict.iteritems():
            if reference_dict[item] > freq_cutoff:
                new_dict[item] = count
        return new_dict

    def extract_features(self, sentence, sent_annotations, parse_tree):
        words = sentence.split()
        feature_set = []
        #case features
        case_features = self._get_case_features(sent_annotations, sentence)
        feature_set += case_features

        # dependency features
        dependency_tuple_dict = self._get_dependency_tuples(sent_annotations)

        # entity features
        entity_features = self._get_entity_features(sent_annotations)
        feature_set += entity_features

        # lexical features
        lexical_features = self._get_lexical_features(words)
        feature_set += lexical_features

        # ngram features
        unigram_dict, bigram_dict, trigram_dict = self._get_ngrams(sent_annotations)

        # parse features
        avg_depth_feature, lexical_production_dict = self._get_parse_features(parse_tree, sent_annotations)
        feature_set += avg_depth_feature

        # POS features
        pos_features = self._get_POS_features(sent_annotations)
        feature_set += pos_features

        # punctuation features
        punctuation_features = self._get_punctuation_features(sentence)
        feature_set += punctuation_features

        # readability features
        readability_features = self._get_readability_features(sentence, words)
        feature_set += readability_features

        # subjectivity features
        # subjectivity_features = self._get_subjectivity_features(sent_annotations, sentence)
        # feature_set += subjectivity_features

        # word2vec features
        word2vec_features = self._get_word2vec_features(sent_annotations)
        feature_set = np.concatenate((feature_set, word2vec_features), axis=0)

        # get one hot features
        dependency_tuple_feature = self.dep_tuple_vectorizer.transform(dependency_tuple_dict)
        unigram_feature = self.unigram_vectorizer.transform(unigram_dict)
        bigram_feature = self.bigram_vectorizer.transform(bigram_dict)
        trigram_feature = self.trigram_vectorizer.transform(trigram_dict)
        lexical_production_feature = self.lexical_vectorizer.transform(lexical_production_dict)

        feature_vectors = np.array([feature_set])
        feature_vectors = np.concatenate((feature_vectors, dependency_tuple_feature, unigram_feature, bigram_feature, trigram_feature, lexical_production_feature), axis=1)

        return feature_vectors

    def _transform_raw(self, sentence):
        sent_annotations = []

        for dependency in sentence['basicDependencies']:
            dep_idx = dependency['dependent']
            token = sentence['tokens'][dep_idx - 1]

            annotation = StanfordAnnotations(token['word'], token['lemma'], token['pos'], token['ner'], dependency['governor'], dependency['dep'])
            sent_annotations.append(annotation)

        return sent_annotations

    def extract_parse(self, s):
        """
        Easy, built in parser from nltk
        """
        tree_list = self.parser.raw_parse(s, outputFormat='penn')
        tree = next(tree_list)
        return tree

    def extract_annotations(self, s):
        """
        Needs some arm wrestling
        """

        props = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,dcoref'}
        raw_json = self.dep_parser.api_call(s, properties=props)
        sentence = raw_json['sentences'][0]
        return self._transform_raw(sentence)


# fluency
# meaning perservation
# bleu
def getBleuScore(reference, candidate):
    # reference: a list of lists of str
    # candidate: a list of str
    return sentence_bleu(reference, candidate)



if __name__ == '__main__':
    extractor = FeatureExtractor('data/saved_models/GoogleNews-vectors-negative300.bin', 'data/saved_models/corpus_dict.pkl')
    ridge = pickle.load(open('data/saved_models/pt16.pkl', 'rb'))

    # formal example
    s1 = "Pelosi's office steps into fight between Republican leaders Cheney and McCarthy"
    # informal example
    s2 = "ummm his bizzy goin out with me lol"


    feat1 = extractor.extract_annotations(s1)
    parse_tree_1 = extractor.extract_parse(s1)
    feature_vec_1 = extractor.extract_features(s1, feat1, parse_tree_1)
    score1 = ridge.predict(feature_vec_1)

    feat2 = extractor.extract_annotations(s2)
    parse_tree_2 = extractor.extract_parse(s2)
    feature_vec_2 = extractor.extract_features(s2, feat2, parse_tree_2)
    score2 = ridge.predict(feature_vec_2)

    print('Sentence 1:')
    print(s1)
    print('Formality score: ' + str(score1))
    print('---------------------------------')
    print('Sentence 2:')
    print(s2)
    print('Formality score: ' + str(score2))