import torch
from data.data import normalize,denormalize
from model.insitu import Generator
from option import basic_parser
from pathlib import Path

def add_gen_args(parser):
    parser.add_argument('--filename')
    parser.add_argument('--storage')
    parser.add_argument('--hidden_nums',type=int)
    parser.add_argument('--sp',type=int,default=50)
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--end',type=int,default=48)
    return parser

def gen_n_fields(args):
    checkpoint = torch.load(args.filename)
    g_model = Generator(args.dsp,args.dspe,args.ch,args.hidden_nums)
    g_model.load_state_dict(checkpoint["g_model_state_dict"])
    g_model.to("cuda:0")
    g_model.eval()
    b_mp = []
    for t in range(args.start,args.end+1):
        n_sp = normalize(args.sp,0,99)
        n_t = normalize(t,1,48)
        b_mp.append([n_sp,n_t])
    b_mp = torch.tensor(b_mp).to("cuda:0")
    fields = g_model(b_mp)

    for t in range(args.start,args.end):
        f = fields[t].cpu().detach().numpy()
        f = denormalize(f,-5471.85791,3225.42578)
        f.tofile(Path(args.storage,f'pf{args.sp}_{t}.bin'))


if __name__ == "__main__":
    parser = basic_parser()
    parser = add_gen_args(parser)
    gen_n_fields(parser.parse_args())
