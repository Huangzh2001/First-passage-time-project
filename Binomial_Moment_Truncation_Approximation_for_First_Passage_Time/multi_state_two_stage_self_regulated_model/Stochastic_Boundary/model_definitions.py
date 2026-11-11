"""批跑使用的四种基因表达网络。"""

import numpy as np

from ssa import TwoStageModel


def network(variant):
    states=3+variant
    state_axis=np.arange(states,dtype=float)
    transcription=.25+(4.2+.5*variant)*(state_axis/(states-1))**(1+.3*variant)
    translation=.65+.85*state_axis/(states-1)
    decay_m=.55+.45*state_axis/(states-1)
    decay_n=.23+.30*state_axis/(states-1)

    def feedback(m, n):
        feedback_m=m*m/(4+m*m)
        feedback_n=n**(2+variant)/(5.**(2+variant)+n**(2+variant))
        if variant==0: # 蛋白激活、mRNA抑制
            activation=(1+2*feedback_n)/(1+feedback_m)
            production=(1+1.2*feedback_n)/(1+.8*feedback_m)
        elif variant==1: # 蛋白负反馈、mRNA激活
            activation=(1+1.8*feedback_m)/(1+2.5*feedback_n)
            production=(1+.7*feedback_m)/(1+2*feedback_n)
        elif variant==2: # 两种计数协同激活
            activation=.3+3*feedback_m*feedback_n
            production=.4+2.4*feedback_m*feedback_n
        else: # 竞争反馈与高Hill指数
            activation=(.4+2*feedback_n)/(1+2*feedback_m)
            production=(1+1.8*feedback_m)/(1+1.5*feedback_n)
        return activation, production, feedback_m, feedback_n

    targets = np.eye(states)

    def switching(i, m, n):
        activation, _, feedback_m, feedback_n = feedback(m, n)
        switches=(targets[(i+1)%states]*(.35+.1*i)[...,None]*activation[...,None]
                  +targets[(i-1)%states]*(.25+.08*i)[...,None]*(1+feedback_n)[...,None]
                  +targets[(i+2)%states]*.09*(1+feedback_m)[...,None])
        return switches

    def transcription_rate(i, m, n):
        _, production, _, _ = feedback(m, n)
        return transcription[i] * production

    names = ("protein_activation", "protein_repression", "cooperative", "competitive")
    return TwoStageModel(names[variant], (0, 0, 0), switching, transcription_rate,
                         translation, decay_m, decay_n)
