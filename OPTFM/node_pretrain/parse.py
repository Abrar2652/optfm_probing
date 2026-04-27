from gnns import *
from ours import SGFormer_MIP as SGFormer_MIP_baseline
from ours_nocross import SGFormer_MIP as SGFormer_MIP_gcn
from ours_crossattention import SGFormer_MIP as SGFormer_MIP_hierarchical
from ours_crossattention_improve import SGFormer_MIP as SGFormer_MIP_hierarchical_improve


def parse_method_mip(args, c, var_d, con_d, device):
    
    if args.model == 'baseline':
        model = SGFormer_MIP_baseline(var_d, con_d, args.hidden_channels, c, graph_weight=args.graph_weight, aggregate=args.aggregate,
                trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout, trans_num_heads=args.trans_num_heads, trans_use_bn=args.trans_use_bn, trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight, trans_use_act=args.trans_use_act
                ).to(device)
    elif args.model == 'gcn':
        model = SGFormer_MIP_gcn(var_d, con_d, args.hidden_channels, c, graph_weight=args.graph_weight, aggregate=args.aggregate,
                trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout, trans_num_heads=args.trans_num_heads, trans_use_bn=args.trans_use_bn, trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight, trans_use_act=args.trans_use_act
                ).to(device)
    else:
        # model = SGFormer_MIP_hierarchical(var_d, con_d, args.hidden_channels, c, graph_weight=args.graph_weight, aggregate=args.aggregate,
        #         trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout, trans_num_heads=args.trans_num_heads, trans_use_bn=args.trans_use_bn, trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight, trans_use_act=args.trans_use_act
        #         ).to(device)
        
        model = SGFormer_MIP_hierarchical_improve(var_d, con_d, args.hidden_channels, c, graph_weight=args.graph_weight, aggregate=args.aggregate,
                trans_num_layers=args.trans_num_layers, trans_dropout=args.trans_dropout, trans_num_heads=args.trans_num_heads, trans_use_bn=args.trans_use_bn, trans_use_residual=args.trans_use_residual, trans_use_weight=args.trans_use_weight, trans_use_act=args.trans_use_act
                ).to(device)
    return model


def parser_add_main_args(parser):
    
    # Method
    parser.add_argument('--model', type=str, choices=['baseline', 'gcn', 'hierarchical'], default='hierarchical', help='Model descriptions')
    
    # dataset and evaluation
    parser.add_argument('--data_dir', type=str, default='Graphs/')  # Training data dir
    parser.add_argument('--valid_dir', type=str, default='Graphs_valid/')  # Validation data dir
    parser.add_argument('--test_dir', type=str, default='Graphs_test/')  # Testing data dir
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 1)')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--metric', type=str, default='f1', choices=['acc', 'rocauc', 'f1', 'recall'],
                        help='evaluation metric')
    
    parser.add_argument('--all_data', action='store_true', help='use all the data for training and testing')

    # gnn branch
    parser.add_argument('--hidden_channels', type=int, default=16) # 16
    parser.add_argument('--use_graph', action='store_false', help='use input graph')
    parser.add_argument('--aggregate', type=str, default='add', help='aggregate type, add or cat.')
    parser.add_argument('--graph_weight', type=float, default=0.5, help='graph weight.')

    parser.add_argument('--gnn_weight_decay', type=float, default=0.)

    # all-pair attention (Transformer) branch
    parser.add_argument('--trans_num_heads', type=int, default=1, help='number of heads for attention')
    parser.add_argument('--trans_use_weight', action='store_false', help='use weight for trans convolution')
    parser.add_argument('--trans_use_bn', action='store_false', help='use layernorm for trans')
    parser.add_argument('--trans_use_residual', action='store_false', help='use residual link for each trans layer')
    parser.add_argument('--trans_use_act', action='store_true', help='use activation for each trans layer')
    parser.add_argument('--trans_num_layers', type=int, default=1, help='number of layers for all-pair attention.')
    parser.add_argument('--trans_dropout', type=float, default=0., help='trans dropout.')
    parser.add_argument('--trans_weight_decay', type=float, default=0.)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--masked_edge_ratio', type=float, default=0.1, help='The ratio for masked edges in graph reconstruction')
    parser.add_argument('--batch_size', type=int, default=1, help='training graphs size per batch')
    parser.add_argument('--patience', type=int, default=200, help='early stopping patience.')

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--save_result', action='store_true',
                        help='save result')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--use_pretrained', action='store_true', help='whether to use pretrained model')
    parser.add_argument('--save_att', action='store_true', help='whether to save attention (for visualization)')
    parser.add_argument('--model_dir', type=str, default='../../model/')

    # other gnn parameters (for baselines)
    parser.add_argument('--hops', type=int, default=2,
                        help='number of hops for SGC')
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='attention heads for gat')
    parser.add_argument('--out_heads', type=int, default=1,
                        help='out heads for gat')


