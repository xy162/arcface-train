from __future__ import print_function
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
from test import *
from torchvision import datasets
from torchvision import transforms as T
from PIL import Image
import shutil
from models.focal_loss import FocalLoss
from models.metrics import SphereProduct,ArcMarginProduct,AddMarginProduct


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    open(save_name)
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    opt = Config()
    device = torch.device("cuda")

    normalize = T.Normalize(mean=[0.5], std=[0.5])

    transforms = T.Compose([
        T.RandomCrop(opt.input_shape[1:]),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])
    train_dataset = datasets.ImageFolder(root=opt.train_root,
                                         transform=transforms,
                                         loader=lambda path: Image.open(path).convert('L'))
    opt.num_classes = len(train_dataset.classes)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers,
                                  drop_last=True)

    identity_list = get_lfw_list(opt.lfw_test_list)
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    train_start_time = time.time()
    start = time.time()
    best_loss = float(100)
    for i in range(opt.max_epoch):
        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print(
                    '{} train epoch: {} iter: {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(),
                                                                                   acc))

                start = time.time()
        print("Epoch:", i, "losss", loss.item())
        scheduler.step()

        # if i % opt.save_interval == 0 or i == opt.max_epoch-1:
        #     save_model(model, opt.checkpoints_path, opt.backbone, i)
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_model(model, opt.checkpoints_path, opt.backbone, "best"+str(i))

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
    print("trian cost", time.time() - train_start_time, "second")
