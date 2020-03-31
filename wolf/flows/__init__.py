__author__ = 'max'

from wolf.flows.flow import Flow
from wolf.flows.normalization import ActNorm1dFlow, ActNorm2dFlow
from wolf.flows.activation import LeakyReLUFlow, ELUFlow, PowshrinkFlow, IdentityFlow, SigmoidFlow
from wolf.flows.permutation import Conv1x1Flow, InvertibleLinearFlow, InvertibleMultiHeadFlow
from wolf.flows.multiscale_architecture import MultiScaleExternal, MultiScaleInternal
from wolf.flows.couplings import *
from wolf.flows.glow import Glow
from wolf.flows.macow import MaCow
