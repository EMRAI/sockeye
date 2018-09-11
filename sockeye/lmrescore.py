from typing import List, Tuple, Dict
import mxnet as mx
#from pynlpl.lm.lm import ARPALanguageModel
import kenlm
import sys

#class LmRescorer:
#
#    def __init__(model, vocab, lmscale: float):
#        self.model = model
#        self.vocab = vocab
#        self.lmscale = lmscale
#
#
#    #def get_lm_scores(self, history: mx.nd.NDArray, scores: mx.nd.NDArray) -> None:
#    def get_lm_scores(self, history: mx.nd.NDArray, shape: Tuple[int, int]) -> mx.nd.NDArray:
#        scores = mx.nd.zeros(shape)
#        n_hyps = history.shape[0]
#        vocab_size = scores.shape[1]
#        # TODO: save states and make faster?
#        for wordind in range(vocab_size):
#            for hypind in range(n_hyps):
#                prob = _cond_prob(wordind, history[hypind])
#                scores[hypind][wordind] = prob * self.lmscale
#        return scores
#
#                
#    '''
#    Override this to get probability using whatever model is loaded
#    '''
#    def _cond_prob(self, word: int, history: List[int]) -> float:
#        return 0.
#
#class KenlmRescorer(LmRescorer):
#
#    def _cond_prob(self, word: int, history: List[int]):
#        word = self.vocab[word]
#        pass # TODO
#        state = kenlm.State()
#        state2 = kenlm.State()
#        model.BeginSentenceWrite(state)
#        switch = True
#        p = 0.
#        for contextword in history:
#            if switch:
#                p += self.model.score(state, contextword, state2)
#                switch = False
#            else:
#                p += self.model.score(state2, contextword, state)
#                switch = True
#        return p
#

class NgramRescorer:

    def __init__(self, modelpath: str, vocab: Dict[str, int], lmscale: float):
        #print("Loading ARPA language model from", modelpath, file=sys.stderr)
#        self.plm = ARPALanguageModel(modelpath)
        #print("Done loading ARPA", file=sys.stderr)
        self.lm = kenlm.Model(modelpath)
        self.vocab = sorted(vocab.keys(), key=lambda x: vocab[x])
        self.vocab[0] = '<s>'     # not '<pad>'
        self.lmscale = lmscale

    def add_conditional_scores(self, history: mx.nd.NDArray, scores: mx.nd.NDArray) -> None:
        #print('BEFORE', scores, file=sys.stderr)
        n_hyps = scores.shape[0]
        for hypind in range(n_hyps):
            thishist = list(history[hypind])
            thishist = tuple(self.vocab[int(x.asscalar())] for x in thishist[-self.lm.order:])
            scores[hypind] -= self._cond_probs(thishist) * self.lmscale
        #print('AFTER', scores, file=sys.stderr)

#    def _cond_prob_pynlpl(self, history: Tuple[str], word: str) -> float:
#        score = self.plm.scoreword(word, history)
#        return score

    def _cond_probs(self, history) -> mx.nd.NDArray:
        startstate = kenlm.State()
        self.lm.NullContextWrite(startstate)
        for word in history:
            endstate = kenlm.State()
            self.lm.BaseScore(startstate, word, endstate)
            startstate = endstate
        # base-10 log score
        # ONLY works on cpu
        return mx.nd.array([self.lm.BaseScore(startstate, word, kenlm.State()) for word in self.vocab])

#    def _cond_prob_test(self, content: str) -> float:
#        return [_ for _ in self.lm.full_scores(content)][-1][0]

#    def _cond_prob(self, history: Tuple[str], word: str) -> float:
#        startstate = kenlm.State()
#        self.lm.NullContextWrite(startstate)
#        for histword in history:
#            endstate = kenlm.State()
#            self.lm.BaseScore(startstate, histword, endstate)
#            startstate = endstate
#        endstate = kenlm.State()
#        score = self.lm.BaseScore(startstate, word, endstate)
#        # base-10 log score
#        return score
