"""RetinaFace 학습 스크립트 (A모듈).

RetinaFace 원본 `Pytorch_Retinaface/train.py` 와 동일한 학습 절차를 FaceDemo
패키지 구조로 옮긴 것이다. 학습 데이터는 `utils/prepare_data.py` 가 WIDER FACE
와 WFLW 를 통일 포맷으로 합쳐 `data/prepared/` 에 저장한 **통합 데이터셋**
(canonical 15-value 텍스트)을 사용한다.

기본은 두 source 를 배치 내 `mix_ratio` 로 섞는 통합 학습(option A)이며,
`--source wider` / `--source wflw` 로 한쪽만 학습할 수도 있다.

실행 (FaceDemo 루트에서):
    # WIDER+WFLW 통합 학습 (기본) — data/prepared 사용
    python -m modules.detection.training.train --network mobile0.25

    # WIDER 만 / WFLW 만
    python -m modules.detection.training.train --network mobile0.25 --source wider
    python -m modules.detection.training.train --network mobile0.25 --source wflw

    # 통합 배치 구성비 조정 (WIDER:WFLW)
    python -m modules.detection.training.train --network mobile0.25 --mix_ratio 4.0

    # 체크포인트에서 재개
    python -m modules.detection.training.train --network mobile0.25 \
        --resume_net ./weights/mobilenet0.25_epoch_50.pth --resume_epoch 50
"""

from __future__ import print_function

import argparse
import datetime
import math
import os
import time
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

from ..config import cfg_mnet, cfg_re50
from ..network import RetinaFace
from ..prior_box import PriorBox
from .combined import MixedBatchSampler
from .data_augment import preproc
from .multibox_loss import MultiBoxLoss
from .prepared_dataset import CanonicalFaceDataset, detection_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Retinaface Training")
    parser.add_argument(
        "--prepared_dir",
        default="./data/prepared",
        help="통합 데이터셋 디렉토리 (utils.prepare_data 산출물)",
    )
    parser.add_argument(
        "--wider_images",
        default="./data/widerface/train/images",
        help="WIDER FACE 이미지 루트",
    )
    parser.add_argument(
        "--wflw_images", default="./data/WFLW_images", help="WFLW 이미지 루트"
    )
    parser.add_argument(
        "--source",
        default="combined",
        choices=["combined", "wider", "wflw"],
        help="학습에 사용할 source (기본: WIDER+WFLW 통합)",
    )
    parser.add_argument(
        "--network", default="mobile0.25", help="Backbone network mobile0.25 or resnet50"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="Number of workers used in dataloading"
    )
    parser.add_argument(
        "--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate"
    )
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--resume_net", default=None, help="resume net for retraining")
    parser.add_argument(
        "--resume_epoch", default=0, type=int, help="resume iter for retraining"
    )
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="Weight decay for SGD"
    )
    parser.add_argument("--gamma", default=0.1, type=float, help="Gamma update for SGD")
    parser.add_argument(
        "--save_folder", default="./weights/", help="Location to save checkpoint models"
    )
    parser.add_argument(
        "--epoch", default=None, type=int, help="Override max epoch from config"
    )
    parser.add_argument(
        "--mix_ratio",
        default=4.0,
        type=float,
        help="WIDER:WFLW per-batch sampling ratio (combined mode)",
    )
    return parser.parse_args()


def adjust_learning_rate(optimizer, initial_lr, gamma, epoch, step_index, iteration, epoch_size):
    """학습률 스케줄 (PyTorch imagenet 예제 기반)."""
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr - 1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train(args, cfg, net, optimizer, criterion, priors):
    rgb_mean = (104, 117, 123)  # bgr order
    img_dim = cfg["image_size"]
    batch_size = cfg["batch_size"]
    max_epoch = cfg["epoch"]
    num_workers = args.num_workers
    save_folder = args.save_folder

    net.train()
    epoch = 0 + args.resume_epoch
    print("Loading Dataset...")

    prepared_dir = args.prepared_dir
    wider_txt = os.path.join(prepared_dir, "wider_canonical.txt")
    wflw_txt = os.path.join(prepared_dir, "wflw_canonical.txt")
    pp = preproc(img_dim, rgb_mean)

    def load_wider():
        return CanonicalFaceDataset(wider_txt, args.wider_images, pp)

    def load_wflw():
        return CanonicalFaceDataset(wflw_txt, args.wflw_images, pp)

    if args.source == "wider":
        dataset = load_wider()
        epoch_size = math.ceil(len(dataset) / batch_size)

        def make_loader():
            return data.DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=detection_collate,
            )
    elif args.source == "wflw":
        dataset = load_wflw()
        epoch_size = math.ceil(len(dataset) / batch_size)

        def make_loader():
            return data.DataLoader(
                dataset,
                batch_size,
                shuffle=True,
                num_workers=num_workers,
                collate_fn=detection_collate,
            )
    else:
        # Combined WIDER+WFLW training (option A): 배치 내 비율(mix_ratio) 고정.
        wider_dataset = load_wider()
        wflw_dataset = load_wflw()
        dataset = data.ConcatDataset([wider_dataset, wflw_dataset])
        batch_sampler = MixedBatchSampler(
            len(wider_dataset), len(wflw_dataset), batch_size, mix_ratio=args.mix_ratio
        )
        epoch_size = len(batch_sampler)
        print(
            "Combined dataset: WIDER={} WFLW={} | per-batch {}:{} (mix_ratio={}) | {} batches/epoch".format(
                len(wider_dataset),
                len(wflw_dataset),
                batch_sampler.b_wider,
                batch_sampler.b_wflw,
                args.mix_ratio,
                epoch_size,
            )
        )

        def make_loader():
            return data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=detection_collate,
            )

    max_iter = max_epoch * epoch_size

    stepvalues = (cfg["decay1"] * epoch_size, cfg["decay2"] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    batch_iterator = None
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(make_loader())
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg["decay1"]):
                torch.save(
                    net.state_dict(),
                    save_folder + cfg["name"] + "_epoch_" + str(epoch) + ".pth",
                )
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(
            optimizer, args.lr, args.gamma, epoch, step_index, iteration, epoch_size
        )

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg["loc_weight"] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print(
            "Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} "
            "Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}".format(
                epoch,
                max_epoch,
                (iteration % epoch_size) + 1,
                epoch_size,
                iteration + 1,
                max_iter,
                loss_l.item(),
                loss_c.item(),
                loss_landm.item(),
                lr,
                batch_time,
                str(datetime.timedelta(seconds=eta)),
            )
        )

    torch.save(net.state_dict(), save_folder + cfg["name"] + "_Final.pth")


def main():
    args = parse_args()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    else:
        raise ValueError(f"Unknown network: {args.network}")

    if args.epoch is not None:
        cfg["epoch"] = args.epoch

    num_gpu = cfg["ngpu"]
    gpu_train = cfg["gpu_train"]

    net = RetinaFace(cfg=cfg)
    print("Printing net...")
    print(net)

    if args.resume_net is not None:
        print("Loading resume network...")
        try:
            state_dict = torch.load(args.resume_net, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(args.resume_net, map_location="cpu", weights_only=False)
        # create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == "module.":
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)

    if num_gpu > 1 and gpu_train:
        net = torch.nn.DataParallel(net).cuda()
    else:
        net = net.cuda()

    cudnn.benchmark = True

    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = MultiBoxLoss(2, 0.35, True, 0, True, 7, 0.35, False)

    priorbox = PriorBox(cfg, image_size=(cfg["image_size"], cfg["image_size"]))
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    train(args, cfg, net, optimizer, criterion, priors)


if __name__ == "__main__":
    main()
