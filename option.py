import configargparse as ca_parse

def add_train_args(parser):
    parser.add_argument("--storage", required=True, type=str,
                        help="root for model storing")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--data-parallel", action="store_true", default=False,
                        help="enable data parallelism")
    parser.add_argument("--num-workers", type=int, default=6,
                        help="number of workers")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed")

    parser.add_argument("--dataset", required=True, type=str,
                        help="the dataset used to train model")
    parser.add_argument("--data", required=True, type=str,
                        help="root of the dataset")

    parser.add_argument("--lr", type=float, default=5e-5,
                        help="learning rate")
    parser.add_argument("--d-lr", type=float, default=2e-4,
                        help="learning rate of the discriminator")
    parser.add_argument("--beta1", type=float, default=0.,
                        help="beta1 of Adam")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="beta2 of Adam")
    
    parser.add_argument("--adv-loss-weight", type=float, default=0.01,
                        help="weight of adv_loss")
    parser.add_argument("--perc-loss", type=str, default='relu1_2',
                        help="the layer perc_loss use")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="batch size for training")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="start epoch number")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of epochs to train")

    parser.add_argument("--log-every", type=int, default= 20,
                        help="log training status every given number of batches")
    parser.add_argument("--check-every", type=int, default= 50,
                        help="save checkpoint every given number of epochs")
    parser.add_argument("--metric", type=bool, default=False,
                        help="generate images and compute metric after training")
    return parser

def basic_parser():
    parser = ca_parse.ArgumentParser(description="InSituNet",config_file_parser_class=ca_parse.YAMLConfigFileParser)

    parser.add_argument('--conf',is_config_file=True,
                        help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="path to the latest checkpoint")

    parser.add_argument("--arch", required=True, type=str,
                        help="the model arch")
    parser.add_argument("--dsp", type=int, default=2,
                        help="dimensions of the simulation parameters")
    parser.add_argument("--dvp", type=int, default=2,
                        help="dimensions of the simulation parameters")
    parser.add_argument("--dspe", type=int, default=512,
                        help="dimensions of the simulation parameters' encode")
    parser.add_argument("--dvpe", type=int, default=512,
                    help="dimensions of the simulation parameters' encode")
    parser.add_argument("--ch", type=int, default=32,
                        help="channel multiplier")
    parser.add_argument("--d-ch", type=int, default=32,
                        help="channel multiplier")
    parser.add_argument("--hidden-nums", type=int, default=None,
                    help="number of hidden node in self mod")

    return parser