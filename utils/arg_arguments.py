import models

__all__ = ['add_trainer_arg_parser', 'add_visdom_arg_parser']

def add_trainer_arg_parser(parser):
     
     parser.add_argument('--arch', '-a', type=str, metavar='ARCH', default='vgg19_bn_cifar',
                         choices=models.ALL_MODEL_NAMES,
                         help='model architecture: ' +
                         ' | '.join(name for name in models.ALL_MODEL_NAMES) +
                         ' (default: vgg19_bn_cifar)')
     parser.add_argument('--dataset', type=str, default='cifar10',
                         help='training dataset (default: cifar10)')
     parser.add_argument('--workers', type=int, default=20, metavar='N',
                         help='number of data loading workers (default: 20)')
     parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                         help='input batch size for training (default: 100)')
     parser.add_argument('--epochs', type=int, default=32, metavar='N',
                         help='number of epochs to train (default: 32)')
     parser.add_argument('--lr', dest='lr', type=float, default=1e-1, 
                         metavar='LR', help='initial learning rate (default: 1e-1)')
     parser.add_argument('--weight_decay', '-wd', dest='weight_decay', type=float,
                         default=1e-4, metavar='W', help='weight decay (default: 1e-4)')
     parser.add_argument('--gpu', type=str, default='0',
                         help='training GPU index(default:"0",which means use GPU0')
     parser.add_argument('--deterministic', '--det', action='store_true',
                         help='Ensure deterministic execution for re-producible results.')
     parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                         help='SGD momentum (default: 0.9)')
     parser.add_argument('--valuate', action='store_true',
                         help='valuate each training epoch')
     parser.add_argument('--resume', dest='resume_path', type=str, default='',
                         metavar='PATH', help='path to latest train checkpoint (default: '')')
     parser.add_argument('--refine', action='store_true',
                         help='refine from pruned model, use construction to build the model')
     parser.add_argument('--usr_suffix', type=str, default='',
                         help='usr_suffix(default:"", means no usr suffix)')
     parser.add_argument('--log_path', type=str, default='logs/log.txt',
                         help='default: logs/log.txt')
     parser.add_argument('--test_only', '--test_only', action='store_true',
                         help='Execute a test then return.(default: False)')


def add_visdom_arg_parser(parser):
     
     parser.add_argument('--visdom', dest='visdom', action='store_true',
                         help='visualize the training process using visdom')
     parser.add_argument('--vis_env', type=str, default='', metavar='ENV',
                         help='visdom environment (default: "", which means env is automatically set to args.dataset + "_" + args.arch)')
     parser.add_argument('--vis_legend', type=str, default='', metavar='LEGEND',
                         help='refine from pruned model (default: "", which means env is automatically set to args.arch)')
     parser.add_argument('--vis_interval', type=int, default=50, metavar='N',
                         help='visdom plot interval batchs (default: 50)')