"""Model training."""
import argparse
import copy
import glob
import os
import time
from itertools import cycle
from typing import List

import higher
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import common
import made
import transformer
from utils import dataset_util
from utils.arg_util import add_common_arguments, ArgType
from utils.end2end_utils import communicator
from utils.model_util import save_torch_model
from utils.path_util import get_absolute_path
from utils.torch_util import get_torch_device


DEVICE = get_torch_device()


def create_parser():
    parser = argparse.ArgumentParser()

    # 添加通用参数
    common_args: List[ArgType] = [
        ArgType.DATASET,
        ArgType.END2END,
        ArgType.MODEL_UPDATE
    ]
    add_common_arguments(parser, arg_types=common_args)

    # Training.
    # parser.add_argument("--dataset", type=str, default="census", help="Dataset.")
    parser.add_argument("--num-gpus", type=int, default=0, help="#gpus.")
    parser.add_argument("--bs", type=int, default=1024, help="Batch size.")
    parser.add_argument(
        "--warmups",
        type=int,
        default=0,
        help="Learning rate warmup steps.  Crucial for Transformer.",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train for."
    )
    parser.add_argument("--constant-lr", type=float, default=None, help="Constant LR?")
    parser.add_argument(
        "--column-masking",
        action="store_true",
        help="Column masking training, which permits wildcard skipping"
             " at querying time.",
    )

    # MADE.
    parser.add_argument("--fc-hiddens", type=int, default=256, help="Hidden units in FC.")
    parser.add_argument("--layers", type=int, default=4, help="# layers in FC.")
    parser.add_argument("--residual", action="store_true", help="ResMade?")
    parser.add_argument("--direct-io", action="store_true", help="Do direct IO?")
    parser.add_argument(
        "--inv-order",
        action="store_true",
        help="Set this flag iff using MADE and specifying --order. Flag --order "
             "lists natural indices, e.g., [0 2 1] means variable 2 appears second."
             "MADE, however, is implemented to take in an argument the inverse "
             "semantics (element i indicates the position of variable i).  Transformer"
             " does not have this issue and thus should not have this flag on.",
    )
    parser.add_argument(
        "--input-encoding",
        type=str,
        default="binary",
        help="Input encoding for MADE/ResMADE, {binary, one_hot, embed}.",
    )
    parser.add_argument(
        "--output-encoding",
        type=str,
        default="one_hot",
        help="Iutput encoding for MADE/ResMADE, {one_hot, embed}.  If embed, "
             "then input encoding should be set to embed as well.",
    )

    # Transformer.
    parser.add_argument(
        "--heads",
        type=int,
        default=0,
        help="Transformer: num heads.  A non-zero value turns on Transformer"
             " (otherwise MADE/ResMADE).",
    )
    parser.add_argument("--blocks", type=int, default=2, help="Transformer: num blocks.")
    parser.add_argument("--dmodel", type=int, default=32, help="Transformer: d_model.")
    parser.add_argument("--dff", type=int, default=128, help="Transformer: d_ff.")
    parser.add_argument(
        "--transformer-act", type=str, default="gelu", help="Transformer activation."
    )

    # Ordering.
    parser.add_argument("--num-orderings", type=int, default=1, help="Number of orderings.")
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        required=False,
        help="Use a specific ordering.  "
             "Format: e.g., [0 2 1] means variable 2 appears second.",
    )
    return parser


args = create_parser().parse_args()


def Entropy(name, data, bases=None):
    import scipy.stats

    # s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        # s += " {:.4f} {}".format(e, unit)
    # print(s)
    return ret


def KD_loss(pm_hat, nm_hat):
    loss = nn.MSELoss()
    output = loss(pm_hat, nm_hat)
    return output


def RunEpoch(
        split,
        model,
        opt,
        scheduler,
        train_data,
        val_data=None,
        batch_size=100,
        upto=None,
        epoch_num=None,
        verbose=False,
        log_every=10,
        return_losses=False,
        table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    dataset = train_data if split == "train" else val_data
    losses = []

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, xb in enumerate(loader):
        if split == "train":
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                    param_group["lr"] = lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model ** -0.5) * min(
                        (global_steps ** -0.5), global_steps * (t ** -1.5)
                    )
                    param_group["lr"] = lr

        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        model_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_logits.append(model_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
            xbhat += model_out

        if xbhat.shape == xb.shape:
            if mean:
                xb = (xb * std) + mean
            loss = (
                    F.binary_cross_entropy_with_logits(xbhat, xb, size_average=False)
                    / xbhat.size()[0]
            )
            # loss = loss
        else:
            if model.input_bins is None:
                # NOTE: we have to view() it in this order due to the mask
                # construction within MADE.  The masks there on the output unit
                # determine which unit sees what input vars.
                xbhat = xbhat.view(-1, model.nout // model.nin, model.nin)
                # Equivalent to:
                loss = (
                    F.cross_entropy(xbhat, xb.long(), reduction="none").sum(-1).mean()
                )
                # loss = loss
            else:
                if num_orders_to_forward == 1:
                    loss = model.nll(xbhat, xb).mean()
                    loss = loss
                else:
                    # Average across orderings & then across minibatch.
                    #
                    #   p(x) = 1/N sum_i p_i(x)
                    #   log(p(x)) = log(1/N) + log(sum_i p_i(x))
                    #             = log(1/N) + logsumexp ( log p_i(x) )
                    #             = log(1/N) + logsumexp ( - nll_i (x) )
                    #
                    # Used only at test time.
                    logps = []  # [batch size, num orders]
                    assert len(model_logits) == num_orders_to_forward, len(model_logits)
                    for logits in model_logits:
                        # Note the minus.
                        logps.append(-model.nll(logits, xb))
                    logps = torch.stack(logps, dim=1)
                    logps = logps.logsumexp(dim=1) + torch.log(
                        torch.tensor(1.0 / nsamples, device=logps.device)
                    )
                    loss = (-logps).mean()
                    # loss = loss

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                if epoch_num % 10 == 0:
                    print(
                        "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                            epoch_num,
                            step,
                            split,
                            loss.item() / np.log(2) - table_bits,
                            loss.item() / np.log(2),
                            table_bits,
                            opt.param_groups[0]["lr"],
                        )
                    )
            else:
                print(
                    "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                        epoch_num, step, split, loss.item(), loss.item() / np.log(2)
                    )
                )

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if split == "train":
        x = "uncomment"
        scheduler.step()
    if return_losses:
        return losses
    return np.mean(losses)


def RunUpdateEpoch(
        split,
        model,
        pmodel,
        opt,
        scheduler,
        train_data,
        transfer_data,
        omega=0.0001,
        val_data=None,
        batch_size=100,
        upto=None,
        epoch_num=None,
        verbose=False,
        log_every=10,
        return_losses=False,
        table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    pmodel.eval()
    dataset = train_data if split == "train" else val_data
    losses = []

    # print(train_data.size())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size * 2, shuffle=(split == "train")
    )
    trloader = torch.utils.data.DataLoader(
        transfer_data, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, (trb, xb) in enumerate(zip(trloader, cycle(loader))):
        """
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(loader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr
        """
        if upto and step >= upto:
            break

        xb = xb.to(DEVICE).to(torch.float32)
        trb = trb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.
        xbhat = None
        trbhat = None
        pmtrbhat = None
        model_logits = []
        model_tr_logits = []
        pmodel_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_out = model(xb)
            model_tr_out = model(trb)
            pmodel_out = pmodel(trb)
            model_logits.append(model_out)
            model_tr_logits.append(model_tr_out)
            pmodel_logits.append(pmodel_out)
            if xbhat is None:
                xbhat = torch.zeros_like(model_out)
                trbhat = torch.zeros_like(model_tr_out)
                pmtrbhat = torch.zeros_like(pmodel_out)
            xbhat += model_out
            trbhat += model_tr_out
            pmtrbhat += pmodel_out

        if num_orders_to_forward == 1:
            nll1 = model.nll(xbhat, xb)
            nll2 = model.nll(trbhat, trb)
            nll3 = pmodel.nll(pmtrbhat, trb)
            loss1 = nll1.mean()
            loss2 = nll2.mean()
            # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
            loss3 = model.kd_loss(trbhat, pmtrbhat)

            loss = (1 - omega) * (loss3.mean() / 2 + loss2 / 2) + omega * loss1
            # loss = (1-omega)*loss3.mean() + omega*loss1

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                if epoch_num % 10 == 0:
                    print(
                        "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                            epoch_num,
                            step,
                            split,
                            loss.item() / np.log(2) - table_bits[0],
                            loss.item() / np.log(2),
                            table_bits[0],
                            opt.param_groups[0]["lr"],
                        )
                    )
            else:
                print(
                    "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                        epoch_num, step, split, loss.item(), loss.item() / np.log(2)
                    )
                )

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if split == "train":
        scheduler.step()
    if return_losses:
        return losses
    return np.mean(losses)


def InnerForward(model, pmodel, omega, xb, trb):
    xbhat = None
    trbhat = None
    pmtrbhat = None
    model_logits = []
    model_tr_logits = []
    pmodel_logits = []

    if hasattr(model, "update_masks"):
        # We want to update_masks even for first ever batch.
        model.update_masks()

    model_out = model(xb)
    model_tr_out = model(trb)
    pmodel_out = pmodel(trb)
    model_logits.append(model_out)
    model_tr_logits.append(model_tr_out)
    pmodel_logits.append(pmodel_out)
    if xbhat is None:
        xbhat = torch.zeros_like(model_out)
        trbhat = torch.zeros_like(model_tr_out)
        pmtrbhat = torch.zeros_like(pmodel_out)
    xbhat += model_out
    trbhat += model_tr_out
    pmtrbhat += pmodel_out

    nll1 = model.nll(xbhat, xb)
    nll2 = model.nll(trbhat, trb)
    # nll3 = pmodel.nll(pmtrbhat, trb)
    loss1 = nll1.mean()
    loss2 = nll2.mean()
    # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
    loss3 = model.kd_loss(trbhat, pmtrbhat)

    # loss = (1 - omega) * (loss3.mean() / 2 + loss2 / 2) + omega * loss1
    loss = (1 - omega) * loss2 + omega * loss1

    return loss


def RunAdaptEpoch(
        split,
        model,
        pmodel,
        opt,
        scheduler,
        train_data,
        transfer_data,
        omega=0.0001,
        val_data=None,
        batch_size=100,
        upto=None,
        epoch_num=None,
        verbose=False,
        log_every=10,
        return_losses=False,
        table_bits=None,
        task_num=4,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    dataset = train_data if split == "train" else val_data
    losses = []

    if split == "train":
        opt.zero_grad()

        for task in range(task_num):
            lenth1 = dataset.__len__()
            lenth2 = transfer_data.__len__()
            indecis1 = np.random.choice(range(lenth1), int(lenth1 / task_num), replace=False)
            indecis2 = np.random.choice(range(lenth2), int(lenth2 / task_num), replace=False)
            task_dataset = copy.deepcopy(dataset)
            task_trdataset = copy.deepcopy(transfer_data)
            task_dataset.tuples = dataset.tuples[indecis1]
            task_trdataset.tuples = transfer_data.tuples[indecis2]
            # tasks_dataset.append((task_dataset, task_trdataset))

            # print(train_data.size())
            loader = torch.utils.data.DataLoader(
                task_dataset, batch_size=batch_size * 2, shuffle=(split == "train")
            )
            trloader = torch.utils.data.DataLoader(
                task_trdataset, batch_size=batch_size, shuffle=(split == "train")
            )

            # How many orderings to run for the same batch?
            nsamples = 1
            if hasattr(model, "orderings"):
                nsamples = len(model.orderings)

            inner_opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            with higher.innerloop_ctx(
                    model,
                    inner_opt,
                    copy_initial_weights=False,
            ) as (fmodel, diffopt):
                for step, (trb, xb) in enumerate(zip(trloader, cycle(loader))):
                    xb = xb.to(DEVICE).to(torch.float32)
                    trb = trb.to(DEVICE).to(torch.float32)

                    loss = InnerForward(fmodel, pmodel, omega, xb.detach(), trb.detach())
                    diffopt.step(loss)

                for step1, (trb1, xb1) in enumerate(zip(trloader, cycle(loader))):
                    xb1 = xb1.to(DEVICE).to(torch.float32)
                    trb1 = trb1.to(DEVICE).to(torch.float32)

                    # test_loss = InnerForward(fmodel, pmodel, omega, xb1.detach(), trb1.detach())
                    xbhat1 = None
                    trbhat1 = None
                    pmtrbhat1 = None
                    model_logits1 = []
                    model_tr_logits1 = []
                    pmodel_logits1 = []

                    if hasattr(model, "update_masks"):
                        # We want to update_masks even for first ever batch.
                        model.update_masks()

                    model_out = model(xb1)
                    model_tr_out = model(trb1)
                    pmodel_out = pmodel(trb1)
                    model_logits1.append(model_out)
                    model_tr_logits1.append(model_tr_out)
                    pmodel_logits1.append(pmodel_out)
                    if xbhat1 is None:
                        xbhat1 = torch.zeros_like(model_out)
                        trbhat1 = torch.zeros_like(model_tr_out)
                        pmtrbhat1 = torch.zeros_like(pmodel_out)
                    xbhat1 += model_out
                    trbhat1 += model_tr_out
                    pmtrbhat1 += pmodel_out

                    nll1 = model.nll(xbhat1, xb1)
                    nll2 = model.nll(trbhat1, trb1)
                    # nll3 = pmodel.nll(pmtrbhat, trb)
                    loss1 = nll1.mean()
                    loss2 = nll2.mean()
                    # loss3 = model.semiparam_kd_loss(trbhat, pmtrbhat,loss2)
                    loss3 = model.kd_loss(trbhat1, pmtrbhat1)

                    test_loss = (1 - omega) * (loss3.mean() / 2 + loss2 / 2) + omega * loss1
                    # test_loss = (1 - omega) * loss2  + omega * loss1

                    losses.append(test_loss.item())

                    if step1 % log_every == 0 and task == 0:
                        if split == "train":
                            if epoch_num % 10 == 0:
                                print(
                                    "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                                        epoch_num,
                                        step1,
                                        split,
                                        test_loss.item() / np.log(2) - table_bits[0],
                                        test_loss.item() / np.log(2),
                                        table_bits[0],
                                        opt.param_groups[0]["lr"],
                                    )
                                )
                        else:
                            print(
                                "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                                    epoch_num,
                                    step1,
                                    split,
                                    test_loss.item(),
                                    test_loss.item() / np.log(2),
                                )
                            )
                    # opt.zero_grad()
                    test_loss.backward()

        opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if split == "train":
        scheduler.step()
    if return_losses:
        return losses

    return np.mean(losses)


def RunDistillEpoch(
        split,
        model,
        pmodel,
        opt,
        scheduler,
        transfer_data,
        omega=0.001,
        val_data=None,
        batch_size=100,
        upto=None,
        epoch_num=None,
        verbose=False,
        log_every=10,
        return_losses=False,
        table_bits=None,
):
    torch.set_grad_enabled(split == "train")
    model.train() if split == "train" else model.eval()
    pmodel.eval()
    dataset = transfer_data if split == "train" else val_data
    losses = []

    trloader = torch.utils.data.DataLoader(
        transfer_data, batch_size=batch_size, shuffle=(split == "train")
    )

    # How many orderings to run for the same batch?
    nsamples = 1
    if hasattr(model, "orderings"):
        nsamples = len(model.orderings)

    for step, trb in enumerate(trloader):
        """
        if split == 'train':
            base_lr = 8e-4
            for param_group in opt.param_groups:
                if args.constant_lr:
                    lr = args.constant_lr
                elif args.warmups:
                    t = args.warmups
                    d_model = model.embed_size
                    global_steps = len(trloader) * epoch_num + step + 1
                    lr = (d_model**-0.5) * min(
                        (global_steps**-.5), global_steps * (t**-1.5))
                else:
                    lr = 1e-2

                param_group['lr'] = lr

        if upto and step >= upto:
            break
        """
        trb = trb.to(DEVICE).to(torch.float32)

        # Forward pass, potentially through several orderings.

        trbhat = None
        pmtrbhat = None
        model_logits = []
        model_tr_logits = []
        pmodel_logits = []
        num_orders_to_forward = 1
        if split == "test" and nsamples > 1:
            # At test, we want to test the 'true' nll under all orderings.
            num_orders_to_forward = nsamples

        for i in range(num_orders_to_forward):
            if hasattr(model, "update_masks"):
                # We want to update_masks even for first ever batch.
                model.update_masks()

            model_tr_out = model(trb)
            pmodel_out = pmodel(trb)

            model_tr_logits.append(model_tr_out)
            pmodel_logits.append(pmodel_out)
            if trbhat is None:
                trbhat = torch.zeros_like(model_tr_out)
                pmtrbhat = torch.zeros_like(pmodel_out)

            trbhat += model_tr_out
            pmtrbhat += pmodel_out

        if trbhat.shape == trb.shape:
            if mean:
                trb = (trb * std) + mean

            loss2 = (
                    F.binary_cross_entropy_with_logits(trbhat, trb, size_average=False)
                    / trbhat.size()[0]
            )

            loss3 = KD_loss(trbhat, pmtrbhat)

            # loss = loss3#loss3/2 + loss2/2

        if num_orders_to_forward == 1:
            nll2 = model.nll(trbhat, trb)
            # nll3 = pmodel.nll(pmtrbhat, trb)

            loss2 = nll2.mean()

            loss3 = model.kd_loss(trbhat, pmtrbhat)  # KD_loss(trbhat, pmtrbhat)
            loss = loss3.mean() / 2 + loss2 / 2  # 5*loss3.mean()/6 + 1*loss2/6

        losses.append(loss.item())

        if step % log_every == 0:
            if split == "train":
                print(
                    "Epoch {} Iter {}, {} entropy gap {:.4f} bits (loss {:.3f}, data {:.3f}) {:.5f} lr".format(
                        epoch_num,
                        step,
                        split,
                        loss.item() / np.log(2) - table_bits[0],
                        loss.item() / np.log(2),
                        table_bits[0],
                        opt.param_groups[0]["lr"],
                    )
                )
            else:
                print(
                    "Epoch {} Iter {}, {} loss {:.4f} nats / {:.4f} bits".format(
                        epoch_num, step, split, loss.item(), loss.item() / np.log(2)
                    )
                )

        if split == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        if verbose:
            print("%s epoch average loss: %f" % (split, np.mean(losses)))

    if split == "train":
        scheduler.step()
    if return_losses:
        return losses
    return np.mean(losses)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print("Number of model parameters: {} (~= {:.1f}MB)".format(num_params, mb))
    # print(model)
    return mb


def InvertOrder(order):
    if order is None:
        return None
    # 'order'[i] maps nat_i -> position of nat_i
    # Inverse: position -> natural idx.  This it the 'true' ordering -- it's how
    # heuristic orders are generated + (less crucially) how Transformer works.
    nin = len(order)
    inv_ordering = [None] * nin
    for natural_idx in range(nin):
        inv_ordering[order[natural_idx]] = natural_idx
    return inv_ordering


def MakeMade(scale, cols_to_train, seed, fixed_ordering=None):
    if args.inv_order:
        print("Inverting order!")
        fixed_ordering = InvertOrder(fixed_ordering)

    model = made.MADE(
        nin=len(cols_to_train),
        hidden_sizes=[scale] * args.layers
        if args.layers > 0
        else [512, 256, 512, 128, 1024],
        nout=sum([c.DistributionSize() for c in cols_to_train]),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        input_encoding=args.input_encoding,
        output_encoding=args.output_encoding,
        embed_size=32,
        seed=seed,
        do_direct_io_connections=args.direct_io,
        natural_ordering=False if seed is not None and seed != 0 else True,
        residual_connections=args.residual,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
    ).to(DEVICE)

    return model


def MakeTransformer(cols_to_train, fixed_ordering, seed=None):
    return transformer.Transformer(
        num_blocks=args.blocks,
        d_model=args.dmodel,
        d_ff=args.dff,
        num_heads=args.heads,
        nin=len(cols_to_train),
        input_bins=[c.DistributionSize() for c in cols_to_train],
        use_positional_embs=True,
        activation=args.transformer_act,
        fixed_ordering=fixed_ordering,
        column_masking=args.column_masking,
        seed=seed,
    ).to(DEVICE)


def InitWeight(m):
    if type(m) == made.MaskedLinear or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    if type(m) == nn.Embedding:
        nn.init.normal_(m.weight, std=0.02)


def AdaptTask(pre_model, new_model_path=None, seed=0, end2end: bool = False):
    torch.manual_seed(0)
    np.random.seed(0)

    # 读取数据集
    if not end2end:
        table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(dataset=args.dataset, permute=False)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(path=abs_dataset_path)
        split_indices = communicator.SplitIndicesCommunicator().get()

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print(table.data.info())

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    # print("split index: {}".format(split_indices))
    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.heads > 0:
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )
            pmodel = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
            model.load_state_dict(torch.load(pre_model))

        mb = ReportModel(model)

        if False:  # not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
            decay_rate = 0.98
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )
        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)
        print("dataset size: {}".format(train_data.size()))

        transfer_data = TransferDataPrepare(
            train_data=train_data, split_indices=split_indices, update_step=update_step
        )

        train_data.tuples = train_data.tuples[
                            split_indices[update_step]: split_indices[update_step + 1]
                            ]

        print(
            "train data shape: {}, transfer data shape:{}".format(
                train_data.tuples.shape, transfer_data.tuples.shape
            )
        )
        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunAdaptEpoch(
                "train",
                model,
                pmodel,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                transfer_data=transfer_data,
                val_data=train_data,
                omega=0.1,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

                check_point_path = "./Naru/checkpoints/update_batch_{}_epoch_{}.pt".format(update_step + 1, epoch)
                absolute_checkpoint_path = get_absolute_path(check_point_path)
                torch.save(model.state_dict(), absolute_checkpoint_path)
            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/adapt-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )

        if not end2end:
            save_torch_model(model, PATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def TrainTask(seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    table = dataset_util.DatasetLoader.load_dataset(dataset=args.dataset)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )[0]
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    print(table.data.info())

    table_train = table

    if args.heads > 0:
        model = MakeTransformer(
            cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
        )
    else:
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print("Applying InitWeight()")
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), lr=2e-4)
        decay_rate = 0.96
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=decay_rate
        )

    bs = args.bs
    log_every = 200

    train_data = common.TableDataset(table)
    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunEpoch(
            "train",
            model,
            opt,
            scheduler=lr_scheduler,
            train_data=train_data,
            val_data=train_data,
            batch_size=bs,
            epoch_num=epoch,
            log_every=log_every,
            table_bits=table_bits,
        )

        if epoch % 1 == 0:
            print(
                "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                    epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                )
            )
            since_start = time.time() - train_start
            print("time since start: {:.1f} secs".format(since_start))

        train_losses.append(mean_epoch_train_loss)

    print("Training done; evaluating likelihood on full data:")
    all_losses = RunEpoch(
        "test",
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        scheduler=lr_scheduler,
        batch_size=1024,
        log_every=500,
        table_bits=table_bits,
        return_losses=True,
    )
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if fixed_ordering is None:
        if seed is not None:
            PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
            )
        else:
            PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
                time.time(),
            )
    else:
        annot = ""
        if args.inv_order:
            annot = "-invOrder"

        PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
            args.dataset,
            mb,
            model.model_bits,
            table_bits,
            model.name(),
            args.epochs,
            seed,
            "_".join(map(str, fixed_ordering)),
            annot,
        )
    save_torch_model(model, PATH)


def TransferDataPrepare(train_data, split_indices: List[int], update_step):
    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data.tuples[:split_indices[0]]
    rndindices = torch.randperm(len(tuples1))[:10000]
    transfer_data.tuples = tuples1[rndindices]  # 从原始数据中随机抽取样本放进迁移数据集

    # 从更新数据切片中抽样数据放进迁移数据集
    for i in range(update_step):
        tuples1 = train_data.tuples[split_indices[i]: split_indices[i + 1]]
        rndindices = torch.randperm(len(tuples1))[:500]
        transfer_data.tuples = torch.cat(
            [transfer_data.tuples, tuples1[rndindices]], dim=0
        )

    return transfer_data


def UpdateTask(pre_model, new_model_path=None, seed=0, end2end: bool = False):
    torch.manual_seed(0)
    np.random.seed(0)

    # 读取数据集
    if not end2end:
        table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(dataset=args.dataset, permute=False)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(path=abs_dataset_path)
        split_indices = communicator.SplitIndicesCommunicator().get()

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )

    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print("data info: {}".format(table.data.info()))

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    # print("split index: {}".format(split_indices))
    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.heads > 0:
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel.load_state_dict(torch.load(pre_model))
            model.load_state_dict(torch.load(pre_model))
            pmodel.eval()

        mb = ReportModel(model)

        if False:  # not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
            decay_rate = 0.98
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )
        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)
        print("dataset size: {}".format(train_data.size()))

        transfer_data = TransferDataPrepare(
            train_data=train_data, split_indices=split_indices, update_step=update_step
        )

        train_data.tuples = train_data.tuples[
                            split_indices[update_step]: split_indices[update_step + 1]
                            ]

        print(
            "train data shape: {}, transfer data shape:{}".format(
                train_data.tuples.shape, transfer_data.tuples.shape
            )
        )
        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunUpdateEpoch(
                "train",
                model,
                pmodel,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                transfer_data=transfer_data,
                val_data=train_data,
                omega=0.1,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

                check_point_path = "./Naru/checkpoints/update_batch_{}_epoch_{}.pt".format(update_step + 1, epoch)
                absolute_checkpoint_path = get_absolute_path(check_point_path)
                torch.save(model.state_dict(), absolute_checkpoint_path)
            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/update-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/update-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/update-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        if not end2end:
            save_torch_model(model, PATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def FineTuneTask(pre_model, new_model_path=None, seed=0, end2end: bool = False):
    torch.manual_seed(0)
    np.random.seed(0)

    # 读取数据集
    if not end2end:
        table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(dataset=args.dataset, permute=False)
    else:
        abs_dataset_path = communicator.DatasetPathCommunicator().get()
        table = dataset_util.NpyDatasetLoader.load_npy_dataset_from_path(path=abs_dataset_path)
        split_indices = communicator.SplitIndicesCommunicator().get()

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )
    table_bits = table_bits[0]
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    # print(table.data.info())

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    for update_step in range(len(split_indices) - 1):
        table_main = table

        if args.heads > 0:
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            model.load_state_dict(torch.load(pre_model))
            model.train()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            model.load_state_dict(torch.load(pre_model))
            model.train()

        mb = ReportModel(model)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
            decay_rate = 0.98
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )

        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)

        train_data.tuples = train_data.tuples[
                            split_indices[update_step]: split_indices[update_step + 1]
                            ]

        train_losses = []
        train_start = time.time()
        print(
            "finetuning, initial lr = {}, update batch {}".format(
                opt.param_groups[0]["lr"], update_step
            )
        )
        for epoch in range(args.epochs):
            print(train_data.tuples.shape)
            mean_epoch_train_loss = RunEpoch(
                "train",
                model,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                val_data=train_data,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if (epoch + 1) % 10 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/FT-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/FT-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/FT-{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        if not end2end:
            save_torch_model(model, PATH)
            pre_model = PATH
        else:
            save_torch_model(model, new_model_path)


def DistillTask(pre_model, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    table = dataset_util.DatasetLoader.load_dataset(dataset=args.dataset)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )

    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    print(table.data.info())

    """
    Loop over update batches (split indexes) and create new models base on previous model
    """

    for update_step in range(5):
        table_main = table

        if args.heads > 0:
            model = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel = MakeTransformer(
                cols_to_train=table.columns, fixed_ordering=fixed_ordering, seed=seed
            )

            pmodel.load_state_dict(torch.load(pre_model))
            pmodel.eval()
        else:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
            pmodel.load_state_dict(torch.load(pre_model))
            model.load_state_dict(torch.load(pre_model))
            pmodel.eval()
            model.train()

        mb = ReportModel(model)

        if False:  # not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=2e-4)
            decay_rate = 0.96
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )

        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table_main)

        # transfer_data = copy.deepcopy(train_data)
        rndindices = torch.randperm(len(train_data.tuples))[:50000]
        train_data.tuples = train_data.tuples[rndindices]

        # samples = pmodel.sample(num=25000)
        # train_data.tuples = samples

        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            print("data = {}, batch={}".format(train_data.tuples.shape, bs))
            mean_epoch_train_loss = RunDistillEpoch(
                "train",
                model,
                pmodel,
                opt,
                scheduler=lr_scheduler,
                transfer_data=train_data,
                val_data=train_data,
                omega=0.1,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if epoch % 1 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/{}distilled{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                    str(update_step + 1).zfill(2),
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    model.name(),
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/{}distilled{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                    str(update_step + 1).zfill(2),
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits[0],
                    model.name(),
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/{}distilled{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
                str(update_step + 1).zfill(2),
                args.dataset,
                mb,
                model.model_bits,
                table_bits[0],
                model.name(),
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        save_torch_model(model, PATH)
        pre_model = PATH
        os.sys.exit()


def BayesCardExp(pre_model=None, seed=0):
    torch.manual_seed(0)
    np.random.seed(0)

    # Load dataset
    table, split_indices = dataset_util.DatasetLoader.load_permuted_dataset(dataset=args.dataset, permute=False)

    table_bits = Entropy(
        table,
        table.data.fillna(value=0).groupby([c.name for c in table.columns]).size(),
        [2],
    )[0]
    fixed_ordering = None

    if args.order is not None:
        print("Using passed-in order:", args.order)
        fixed_ordering = args.order

    print(table.data.info())

    if pre_model == None:
        if args.dataset in ["dmv-tiny", "dmv", "tpcds"]:
            model = MakeMade(
                scale=args.fc_hiddens,
                cols_to_train=table.columns,
                seed=seed,
                fixed_ordering=fixed_ordering,
            )
        else:
            assert False, args.dataset

        mb = ReportModel(model)

        if not isinstance(model, transformer.Transformer):
            print("Applying InitWeight()")
            model.apply(InitWeight)

        if isinstance(model, transformer.Transformer):
            opt = torch.optim.Adam(
                list(model.parameters()),
                2e-4,
                betas=(0.9, 0.98),
                eps=1e-9,
            )
        else:
            opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
            decay_rate = 0.96
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=opt, gamma=decay_rate
            )

        bs = args.bs
        log_every = 200

        train_data = common.TableDataset(table)
        split_idx = int(len(train_data.tuples) * 0.2)
        train_data.tuples = train_data.tuples[:split_idx, :]
        train_losses = []
        train_start = time.time()
        for epoch in range(args.epochs):
            mean_epoch_train_loss = RunEpoch(
                "train",
                model,
                opt,
                scheduler=lr_scheduler,
                train_data=train_data,
                val_data=train_data,
                batch_size=bs,
                epoch_num=epoch,
                log_every=log_every,
                table_bits=table_bits,
            )

            if epoch % 1 == 0:
                print(
                    "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                        epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                    )
                )
                since_start = time.time() - train_start
                print("time since start: {:.1f} secs".format(since_start))

            train_losses.append(mean_epoch_train_loss)

        print("Training done; evaluating likelihood on full data:")
        all_losses = RunEpoch(
            "test",
            model,
            train_data=train_data,
            val_data=train_data,
            opt=None,
            scheduler=lr_scheduler,
            batch_size=1024,
            log_every=500,
            table_bits=table_bits,
            return_losses=True,
        )
        model_nats = np.mean(all_losses)
        model_bits = model_nats / np.log(2)
        model.model_bits = model_bits

        if fixed_ordering is None:
            if seed is not None:
                PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    model.name(),
                    args.epochs,
                    seed,
                )
            else:
                PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                    args.dataset,
                    mb,
                    model.model_bits,
                    table_bits,
                    model.name(),
                    args.epochs,
                    seed,
                    time.time(),
                )
        else:
            annot = ""
            if args.inv_order:
                annot = "-invOrder"

            PATH = "./models/00{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
                "_".join(map(str, fixed_ordering)),
                annot,
            )
        save_torch_model(model, PATH)
        pre_model = PATH

    if args.dataset in ["dmv-tiny", "dmv", "tpcds"]:
        model = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
        pmodel = MakeMade(
            scale=args.fc_hiddens,
            cols_to_train=table.columns,
            seed=seed,
            fixed_ordering=fixed_ordering,
        )
        pmodel.load_state_dict(torch.load(pre_model))
        pmodel.eval()
    else:
        assert False, args.dataset

    mb = ReportModel(model)

    if not isinstance(model, transformer.Transformer):
        print("Applying InitWeight()")
        model.apply(InitWeight)

    if isinstance(model, transformer.Transformer):
        opt = torch.optim.Adam(
            list(model.parameters()),
            2e-4,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
    else:
        opt = torch.optim.Adam(list(model.parameters()), lr=1e-3)
        decay_rate = 0.96
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opt, gamma=decay_rate
        )
    bs = args.bs
    log_every = 200

    train_data = common.TableDataset(table)
    split_idx = int(len(train_data.tuples) * 0.2)

    transfer_data = copy.deepcopy(train_data)
    tuples1 = transfer_data.tuples[:split_idx]
    rndindices = torch.randperm(len(tuples1))[:20000]
    transfer_data.tuples = tuples1[rndindices]

    train_data.tuples = train_data.tuples[split_idx:]

    print(train_data.tuples.shape, transfer_data.tuples.shape)
    train_losses = []
    train_start = time.time()
    for epoch in range(args.epochs):
        mean_epoch_train_loss = RunUpdateEpoch(
            "train",
            model,
            pmodel,
            opt,
            scheduler=lr_scheduler,
            train_data=train_data,
            transfer_data=transfer_data,
            val_data=train_data,
            omega=0.1,
            batch_size=bs,
            epoch_num=epoch,
            log_every=log_every,
            table_bits=table_bits,
        )

        if epoch % 1 == 0:
            print(
                "epoch {} train loss {:.4f} nats / {:.4f} bits".format(
                    epoch, mean_epoch_train_loss, mean_epoch_train_loss / np.log(2)
                )
            )
            since_start = time.time() - train_start
            print("time since start: {:.1f} secs".format(since_start))

        train_losses.append(mean_epoch_train_loss)

    print("Training done; evaluating likelihood on full data:")
    all_losses = RunEpoch(
        "test",
        model,
        train_data=train_data,
        val_data=train_data,
        opt=None,
        scheduler=lr_scheduler,
        batch_size=1024,
        log_every=500,
        table_bits=table_bits,
        return_losses=True,
    )
    model_nats = np.mean(all_losses)
    model_bits = model_nats / np.log(2)
    model.model_bits = model_bits

    if fixed_ordering is None:
        if seed is not None:
            PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
            )
        else:
            PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-{}.pt".format(
                args.dataset,
                mb,
                model.model_bits,
                table_bits,
                model.name(),
                args.epochs,
                seed,
                time.time(),
            )
    else:
        annot = ""
        if args.inv_order:
            annot = "-invOrder"

        PATH = "models/01{}-{:.1f}MB-model{:.3f}-data{:.3f}-{}-{}epochs-seed{}-order{}{}.pt".format(
            args.dataset,
            mb,
            model.model_bits,
            table_bits[0],
            model.name(),
            args.epochs,
            seed,
            "_".join(map(str, fixed_ordering)),
            annot,
        )
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    torch.save(model.state_dict(), PATH)
    print("Saved to:")
    print(PATH)
    pre_model = PATH


def main():
    if not args.end2end:
        # 原逻辑：用3种增量训练方法，训练所有模型
        relative_model_paths = "./models/origin-{}*MB-model*-data*-*epochs-seed*.pt".format(args.dataset)
        absolute_model_paths = get_absolute_path(relative_model_paths)
        model_paths = glob.glob(str(absolute_model_paths))
        for model_path in model_paths:
            AdaptTask(pre_model=model_path)
            UpdateTask(pre_model=model_path)
            FineTuneTask(pre_model=model_path)
    else:
        # end2end实验：用1种增量训练方法，训练1个模型
        model_path = communicator.ModelPathCommunicator().get()
        new_model_path = model_path  # 保存在原模型的路径下  TODO: 保存在新模型的路径下
        if args.model_update == "update":
            print("UpdateTask - START")
            UpdateTask(pre_model=model_path, new_model_path=new_model_path, end2end=True)
            print("UpdateTask - END")
        elif args.model_update == "adapt":
            print("AdaptTask - START")
            AdaptTask(pre_model=model_path, new_model_path=new_model_path, end2end=True)
            print("AdaptTask - END")
        elif args.model_update == "finetune":
            print("FineTuneTask - START")
            FineTuneTask(pre_model=model_path, new_model_path=new_model_path, end2end=True)
            print("FineTuneTask - END")
        else:
            raise ValueError("model update method not supported")

    # TrainTask()
    # BayesCardExp(pre_model="models/00tpcds-12.9MB-model10.410-data13.514-made-resmade-hidden128_128_128_128-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-10epochs-seed0.pt")
    # UpdateTask(
    #     pre_model="./models/census-38.5MB-model30.303-data15.573-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt"
    # )

    # if args.dataset=="census":
    #     pre_model="./models/origin-census-22.5MB-model26.797-data14.989-200epochs-seed0.pt"
    #     UpdateTask(pre_model=pre_model)
    #     FineTuneTask(pre_model=pre_model)
    # elif args.dataset=="forest":
    #     forest_path="./models/origin-forest-16.5MB-model62.991-data19.148-100epochs-seed0.pt"
    #     UpdateTask(pre_model=forest_path)
    #     FineTuneTask(pre_model=forest_path)
    # DistillTask(pre_model='models/00forest-23.7MB-model59.664-data19.148-made-resmade-hidden256_256_256_256_256-emb32-directIo-binaryInone_hotOut-inputNoEmbIfLeq-colmask-20epochs-seed0.pt')


if __name__ == "__main__":
    main()
