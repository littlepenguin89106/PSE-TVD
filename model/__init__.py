from .insitu.arch import InsituG,InsituD
from .selfmod.arch import SelfModG,SelfModD

def get_models(args):
    kwargs = {
        'dsp':args.dsp,
        'dspe':args.dspe,
        'dvp':args.dvp,
        'dvpe':args.dvpe,
    }
    if args.arch.upper() == 'INSITU':
        g_model = InsituG(**kwargs,ch=args.ch)
        d_model = InsituD(**kwargs,ch=args.d_ch)
    else:
        g_model = SelfModG(**kwargs,ch=args.ch,hidden_nums=args.hidden_nums)
        d_model = SelfModD(**kwargs,ch=args.d_ch)

    return g_model,d_model
