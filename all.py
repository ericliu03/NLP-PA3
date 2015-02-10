import hmm
import corpus
class IceCreamCones(corpus.Document):

    def features(self):
        """How many ice cream cones were consumed on each day?"""
        return self.data  # counts

states = ('Healthy', 'Fever')
end_state = 'E'
 
observations = ('normal', 'cold', 'dizzy')
 
start_probability = {'Healthy': 0.6, 'Fever': 0.4}
 
transition_probability = {
   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
   }
 
emission_probability = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
   }

def fwd_bkw(x, states, a_0, a, e, end_st):
    L = len(x)

    fwd = []
    f_prev = {}
    # forward part of the algorithm
    for i, x_i in enumerate(x):
        f_curr = {}
        for st in states:
            if i == 0:
                # base case for the forward part
                prev_f_sum = a_0[st]
            else:
                prev_f_sum = sum(f_prev[k]*a[k][st] for k in states)

            f_curr[st] = e[st][x_i] * prev_f_sum

        fwd.append(f_curr)
        f_prev = f_curr

    p_fwd = sum(f_curr[k]*a[k][end_st] for k in states)

    bkw = []
    b_prev = {}
    # backward part of the algorithm
    for i, x_i_plus in enumerate(reversed(x[1:]+(None,))):
        b_curr = {}
        for st in states:
            if i == 0:
                # base case for backward part
                b_curr[st] = a[st][end_st]
            else:
                b_curr[st] = sum(a[st][l]*e[l][x_i_plus]*b_prev[l] for l in states)

        bkw.insert(0,b_curr)
        b_prev = b_curr

    p_bkw = sum(a_0[l] * e[l][x[0]] * b_curr[l] for l in states)

    # merging the two parts
    posterior = []
    for i in range(L):
        posterior.append({st: fwd[i][st]*bkw[i][st]/p_fwd for st in states})

    assert p_fwd == p_bkw
    return p_bkw,fwd, bkw, posterior

def example():
    return fwd_bkw(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability,
                   end_state)

hmmm = hmm.UnsupervisedHMM()
hmmm.train_ice_cream([],
                                 initial_probabilities=[.6, .4],        # P(Hot), P(Cold)
                                 transition_probabilities=[[0.69, 0.3, 0.01],    # P(Hot|Hot), P(Cold|Hot)
                                                           [0.4, 0.59, 0.01]],   # P(Hot|Cold), P(Cold|Cold)
                                 emission_probabilities=[[0.5, 0.4, 0.1],  # P(1, 2, 3|Hot)
                                                         [0.1, 0.3, 0.6]],  # P(1, 2, 3|Cold)
                                 states=('Healthy', 'Fever'),
                                 vocabulary=('normal', 'cold', 'dizzy'))
print hmmm.likelihood(IceCreamCones(['normal', 'cold', 'dizzy']))
print "sdfsdfsdf"
a = hmmm.compute_alpha(IceCreamCones(['normal', 'cold', 'dizzy']))
b = hmmm.compute_beta(IceCreamCones(['normal', 'cold', 'dizzy']))
print hmmm.compute_gamma(a, b, IceCreamCones(['normal', 'cold', 'dizzy']))
#print hmmm.compute_xi(a, b, IceCreamCones(['normal', 'cold', 'dizzy']))
