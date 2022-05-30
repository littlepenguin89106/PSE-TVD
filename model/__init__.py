from .insitu.arch import InsituG,InsituD
from .selfmod.arch import SelfModG,SelfModD

def get_models(args):
    kwargs = {
        'dsp':args.dsp,
        'dspe':args.dspe,
        'dvp':args.dvp,
        'dvpe':args.dvpe,
        'ch':args.ch,
    }
    if args.arch.upper() == 'INSITU':
        g_model = InsituG(**kwargs)
        d_model = InsituD(**kwargs)
    else:
        g_model = SelfModG(**kwargs,hidden_nums=args.hidden_nums)
        d_model = SelfModD(**kwargs)

    return g_model,d_model
