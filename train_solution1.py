import torch 
import random
import numpy as np
import argparse
import logging
import datetime
import os
import json
from reconstruction.embedding_dataset import EmbeddingSet
import torch.utils.data as data_utils
from networks.encoder import Encoder
import time
import tqdm
from tensorboardX import SummaryWriter 

import deep_sdf
import deep_sdf.workspace as ws


def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


def main_function(experiment_directory, data_source, continue_from, batch_split, batch_size):

    logging.info("running " + experiment_directory)
    # backup code
    now = datetime.datetime.now()
    code_bk_path = os.path.join(
        experiment_directory, 'code_bk_%s.tar.gz' % now.strftime('%Y_%m_%d_%H_%M_%S'))
    ws.create_code_snapshot('./', code_bk_path,
                            extensions=('.py', '.json', '.cpp', '.cu', '.h', '.sh'),
                            exclude=('examples', 'third-party', 'bin'))

    specs = ws.load_experiment_specifications(experiment_directory)
    # data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    checkpoints = list(
        range(
            specs["SnapshotFrequency"],
            specs["NumEpochs"] + 1,
            specs["SnapshotFrequency"],
        )
    )
    for checkpoint in specs["AdditionalSnapshots"]:
        checkpoints.append(checkpoint)
    checkpoints.sort()

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 10)
    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    latent_size = specs["CodeLength"]
    code_bound = get_spec_with_default(specs, "CodeBound", None)
    def save_latest(epoch):
        ws.save_optimizer(os.path.join(experiment_directory,'pretrained_embedding'), "latest.pth", optimizer, epoch)
        ws.save_latent_vectors(os.path.join(experiment_directory,'pretrained_embedding'), "latest.pth", encoder, epoch)

    def save_checkpoints(epoch):

        ws.save_optimizer(os.path.join(experiment_directory,'pretrained_embedding'), str(epoch) + ".pth", optimizer, epoch)
        ws.save_latent_vectors(os.path.join(experiment_directory,'pretrained_embedding'), str(epoch) + ".pth", encoder, epoch)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    embedding_dataset = EmbeddingSet(data_source, train_split, level = 'easy')

    embedding_loader = data_utils.DataLoader(
        embedding_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=True,
    )
    logging.debug("torch num_threads: {}".format(torch.get_num_threads()))

    encoder = Encoder(latent_size=latent_size).cuda()
    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    lr_schedules = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5 ,gamma=0.5, last_epoch=-1)


    if not os.path.isdir(os.path.join(experiment_directory, 'pretrained_embedding')):
        os.mkdir(os.path.join(experiment_directory, 'pretrained_embedding'))
    if not os.path.isdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.latent_codes_subdir)):
        os.mkdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.latent_codes_subdir))
    if not os.path.isdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.optimizer_params_subdir)):
        os.mkdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.optimizer_params_subdir))
    if not os.path.isdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.tensorboard_log_subdir)):
        os.mkdir(os.path.join(experiment_directory, 'pretrained_embedding', ws.tensorboard_log_subdir))
        
    tensorboard_saver = writer = SummaryWriter(os.path.join(experiment_directory, 'pretrained_embedding', ws.tensorboard_log_subdir))
    start_epoch = 1
    loss_log = []
    lr_log = []
    lat_mag_log = []
    timing_log = []
    param_mag_log = {}

    if continue_from is not None:
        if not os.path.exists(os.path.join(experiment_directory, 'pretrained_embedding', ws.latent_codes_subdir, continue_from + ".pth")) or \
        not os.path.exists(os.path.join(experiment_directory, 'pretrained_embedding', ws.optimizer_params_subdir, continue_from + ".pth")):
            logging.warning('"{}" does not exist! Ignoring this argument...'.format(continue_from))
        else:
            logging.info('continuing from "{}"'.format(continue_from))

            lat_epoch = ws.load_encoder_parameters(
                os.path.join(experiment_directory,ws.latent_codes_subdir), continue_from, encoder
            )

            optimizer_epoch = ws.load_optimizer(
                os.path.join(experiment_directory, ws.optimizer_params_subdir), continue_from + ".pth", optimizer_all
            )

            loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, log_epoch = ws.load_logs(
                experiment_directory
            )

            if not log_epoch == lat_epoch:
                loss_log, lr_log, timing_log, lat_mag_log, param_mag_log = ws.clip_logs(
                    loss_log, lr_log, timing_log, lat_mag_log, param_mag_log, lat_epoch
                )

            if not (lat_epoch == optimizer_epoch):
                raise RuntimeError(
                    "epoch mismatch: {} vs {} vs {}".format(
                        lat_epoch, optimizer_epoch, log_epoch
                    )
                )

            start_epoch = lat_epoch + 1

            logging.debug("loaded")

    logging.info("starting from epoch {}".format(start_epoch))

    logging.info(
        "Number of encoder parameters: {}".format(
            sum(p.data.nelement() for p in encoder.parameters())
        )
    )

    it = 0
    logging.info(encoder)
    for epoch in range(start_epoch, 200):

        logging.info("epoch {}...".format(epoch))
        encoder.train()


        batch_num = len(embedding_loader)
        loader = tqdm.tqdm(embedding_loader,desc="training epoch {}".format(epoch), position=0, leave=True)
        epoch_loss = 0.0
        for bi, (images, embedding_gt) in enumerate(loader):
            #===========Input==============
            #images: (B, 3, W, H)
            #embedding: (B, latent_size)
            #==============================
            input_imgs = images.cuda()
            embedding_gt = embedding_gt.cuda()

            optimizer.zero_grad()
            batch_embedding = encoder(input_imgs)
            #batch_embedding = torch.clamp(batch_embedding, max = code_bound)
            batch_loss = loss_l1(batch_embedding, embedding_gt)
            batch_loss.backward()

            epoch_loss+=batch_loss.item()

            tensorboard_saver.add_scalar('batch_loss', batch_loss, it)
            loss_log.append(batch_loss)
            optimizer.step()
            it+=1
            # release memory
            del batch_loss

        epoch_loss/=batch_num
        logging.info("epoch_loss = {:.9f}".format(epoch_loss))
        tensorboard_saver.add_scalar('epoch_loss', epoch_loss, epoch)
        lr_schedules.step()
        


        if epoch in checkpoints:
            save_checkpoints(epoch)

        if epoch % log_frequency == 0:

            save_latest(epoch)
            ws.save_logs(
                os.path.join(experiment_directory,'pretrained_embedding'),
                loss_log,
                lr_log,
                timing_log,
                lat_mag_log,
                param_mag_log,
                epoch,
            )





if __name__ == "__main__":
    random.seed(31359)
    torch.random.manual_seed(31359)
    np.random.seed(31359)

    arg_parser = argparse.ArgumentParser(description="Train an encoder.")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        dest="data_source",
        required=True,
        help="The data source directory.",
    )
    arg_parser.add_argument(
        "--continue",
        "-c",
        dest="continue_from",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--batch_size",
        dest="batch_size",
        default=512,
        type=int
    )
    arg_parser.add_argument(
        "--batch_split",
        dest="batch_split",
        default=1,
        help="This splits the batch into separate subbatches which are "
        + "processed separately, with gradients accumulated across all "
        + "subbatches. This allows for training with large effective batch "
        + "sizes in memory constrained environments.",
    )

    deep_sdf.add_common_args(arg_parser)

    args = arg_parser.parse_args()

    deep_sdf.configure_logging(args)

    main_function(args.experiment_directory, args.data_source, args.continue_from, int(args.batch_split), int(args.batch_size))