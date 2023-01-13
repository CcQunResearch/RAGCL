import argparse


def pargs():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser()

    # parser.add_argument('--dataset', type=str, default='Weibo')
    # parser.add_argument('--unsup_dataset', type=str, default='UWeiboV1')
    # parser.add_argument('--tokenize_mode', type=str, default='jieba')
    #
    # parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    # parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    # parser.add_argument('--runs', type=int, default=10)
    # parser.add_argument('--ft_runs', type=int, default=10)
    #
    # parser.add_argument('--cuda', type=str2bool, default=True)
    # parser.add_argument('--gpu', type=int, default=0)
    #
    # # 622 or 802
    # parser.add_argument('--split', type=str, default='802')
    # parser.add_argument('--batch_size', type=int, default=128)
    # parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    # parser.add_argument('--n_layers_feat', type=int, default=1)
    # parser.add_argument('--n_layers_conv', type=int, default=3)
    # parser.add_argument('--n_layers_fc', type=int, default=2)
    # parser.add_argument('--hidden', type=int, default=128)
    # parser.add_argument('--global_pool', type=str, default="sum")
    # parser.add_argument('--skip_connection', type=str2bool, default=True)
    # parser.add_argument('--res_branch', type=str, default="BNConvReLU")
    # parser.add_argument('--dropout', type=float, default=0.3)
    # parser.add_argument('--edge_norm', type=str2bool, default=True)
    #
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--ft_lr', type=float, default=0.001)
    # parser.add_argument('--epochs', type=int, default=0)
    # parser.add_argument('--ft_epochs', type=int, default=100)
    # parser.add_argument('--weight_decay', type=float, default=0)
    # parser.add_argument('--lamda', dest='lamda', type=float, default=0.001)
    #
    # # Node centrality metric can be chosen from "Degree", "PageRank", "Eigenvector", "Betweenness".
    # parser.add_argument('--centrality', type=str, default="Betweenness")
    # # Augmentation can be chosen from "DropEdge,mean,0.3,0.7", "NodeDrop,0.3,0.7", "AttrMask,0.3,0.7",
    # # or augmentation combination "DropEdge,mean,0.3,0.7||NodeDrop,0.3,0.7", "DropEdge,mean,0.3,0.7||AttrMask,0.3,0.7".
    # # Str like "DropEdge,mean,0.3,0.7" means "AugName,[aggr,]p,threshold".
    # parser.add_argument('--aug1', type=str, default="DropEdge,mean,0.2,0.7")
    # parser.add_argument('--aug2', type=str, default="NodeDrop,0.2,0.7")
    #
    # parser.add_argument('--use_unlabel', type=str2bool, default=False)
    # parser.add_argument('--use_unsup_loss', type=str2bool, default=False)
    #
    # parser.add_argument('--k', type=int, default=10000)

    #################################################

    parser.add_argument('--dataset', type=str, default='Weibo')
    parser.add_argument('--unsup_dataset', type=str, default='UWeiboV1')
    parser.add_argument('--tokenize_mode', type=str, default='naive')

    parser.add_argument('--vector_size', type=int, help='word embedding size', default=200)
    parser.add_argument('--unsup_train_size', type=int, help='word embedding unlabel data train size', default=20000)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--ft_runs', type=int, default=10)

    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)

    # 622 or 802
    parser.add_argument('--split', type=str, default='802')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unsup_bs_ratio', type=int, default=1)
    parser.add_argument('--undirected', type=str2bool, default=True)

    # ResGCN or BiGCN
    parser.add_argument('--model', type=str, default='ResGCN')
    parser.add_argument('--n_layers_feat', type=int, default=1)
    parser.add_argument('--n_layers_conv', type=int, default=3)
    parser.add_argument('--n_layers_fc', type=int, default=2)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--global_pool', type=str, default="sum")
    parser.add_argument('--skip_connection', type=str2bool, default=True)
    parser.add_argument('--res_branch', type=str, default="BNConvReLU")
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--edge_norm', type=str2bool, default=True)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ft_lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ft_epochs', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lamda', dest='lamda', type=float, default=0.001)

    # Node centrality metric can be chosen from "Degree", "PageRank", "Eigenvector", "Betweenness".
    parser.add_argument('--centrality', type=str, default="PageRank")
    # Augmentation can be chosen from "DropEdge,mean,0.3,0.7", "NodeDrop,0.3,0.7", "AttrMask,0.3,0.7",
    # or augmentation combination "DropEdge,mean,0.3,0.7||NodeDrop,0.3,0.7", "DropEdge,mean,0.3,0.7||AttrMask,0.3,0.7".
    # Str like "DropEdge,mean,0.3,0.7" means "AugName,[aggr,]p,threshold".
    parser.add_argument('--aug1', type=str, default="DropEdge,mean,0.2,0.7")
    parser.add_argument('--aug2', type=str, default="NodeDrop,0.2,0.7")

    parser.add_argument('--use_unlabel', type=str2bool, default=False)
    parser.add_argument('--use_unsup_loss', type=str2bool, default=True)

    parser.add_argument('--k', type=int, default=10000)

    args = parser.parse_args()
    return args
