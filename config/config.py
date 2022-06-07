# class Config(object):
#     env = 'default'
#     backbone = 'resnet18'
#     classify = 'softmax'
#     num_classes = 5749
#     metric = 'arc_margin'
#     easy_margin = False
#     use_se = False
#     loss = 'focal_loss'
#
#     display = False
#     finetune = False
#
#     train_root = './lfw-align-128'
#     train_list = './train_data_13938.txt'
#     val_list = './val_data_13938.txt'
#
#     test_root = './lfw-align-128'
#     test_list = 'lfw_test_pair.txt'
#
#     lfw_root = './lfw-align-128'
#     lfw_test_list = './lfw_test_pair.txt'
#
#     checkpoints_path = 'checkpoints'
#     load_model_path = 'models/resnet18.pth'
#     test_model_path = 'checkpoints/resnet18_110.pth'
#     save_interval = 10
#
#     train_batch_size = 32  # batch size
#     test_batch_size = 60
#
#     input_shape = (1, 128, 128)
#
#     optimizer = 'sgd'
#
#     use_gpu = True  # use GPU or not
#     gpu_id = '0, 1'
#     num_workers = 0  # how many workers for loading data
#     print_freq = 100  # print info every N batch
#
#     debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
#     result_file = 'result.csv'
#
#     max_epoch = 30
#     lr = 1e-1  # initial learning rate
#     lr_step = 10
#     lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
#     weight_decay = 5e-4



class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 5750
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    train_root = './lfw-align-128'
    train_list = '/data/Datasets/webface/train_data_13938.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = '/home/pxxy2417/lfw'
    lfw_test_list = './lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'checkpoints/resnet18_110.pth'
    save_interval = 10

    train_batch_size = 32  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 0  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 30
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
