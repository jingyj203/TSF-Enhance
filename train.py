import logging
import dataset
import utils
import loss

import os
import sklearn
import torch
import numpy as np
import matplotlib
matplotlib.use('agg', force=True)
import matplotlib.pyplot as plt
import time
import argparse
import json
import random
from utils import JSONEncoder, json_dumps

parser = argparse.ArgumentParser(description='Training TSF-Enhacne')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='cub.json')
parser.add_argument('--batch-size', default = 32, type=int, dest = 'sz_batch')
parser.add_argument('--epochs', default = 40, type=int, dest = 'nb_epochs')
parser.add_argument('--log-filename', default = 'example')
parser.add_argument('--workers', default = 4, type=int, dest = 'nb_workers')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--mode', default='train', choices=['train', 'test'],
    help='train with train data or train with trainval')
parser.add_argument('--lr_steps', default=[200,300], nargs='+', type=int)
parser.add_argument('--source_dir', default='', type=str),
parser.add_argument('--root_dir', default='', type=str)
parser.add_argument('--eval_nmi', default=False, action='store_true')
parser.add_argument('--recall', default=[1,2,4,8], nargs='+', type=int)
parser.add_argument('--init_eval', default=False, action='store_true')
parser.add_argument('--no_warmup', default=False, action='store_true')
parser.add_argument('--apex', default=False, action='store_true')
parser.add_argument('--warmup_k', default=5, type=int)
parser.add_argument('--pt_path', default='./results/cub/cub_755_847_904_944.pt', type=str)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # set random seed for all gpus

if not os.path.exists("./log"):
    os.makedirs("./log")
if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists(os.path.join('./results',args.dataset)):
    os.makedirs(os.path.join('./results',args.dataset))

#curr_fn = os.path.basename(args.config).split(".")[0]


out_results_fn = "./log/%s_%s_%d.json" % (args.dataset, args.mode, args.seed)

config = utils.load_config(args.config)

dataset_config = utils.load_config('dataset/config.json')


if args.source_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['source'])
    dataset_config['dataset'][args.dataset]['source'] = os.path.join(args.source_dir, bs_name)
if args.root_dir != '':
    bs_name = os.path.basename(dataset_config['dataset'][args.dataset]['root'])
    dataset_config['dataset'][args.dataset]['root'] = os.path.join(args.root_dir, bs_name)

if args.apex:
    from apex import amp

#set NMI or recall accordingly depending on dataset. note for cub and cars R=1,2,4,8
if args.dataset == 'sop':
    args.recall = [1, 10, 100, 1000]


args.nb_epochs = config['nb_epochs']
args.sz_batch = config['sz_batch']
(dim3,dim4) = config['dims']

if 'warmup_k' in config:
    args.warmup_k = config['warmup_k']

transform_key = 'transform_parameters'
if 'transform_key' in config.keys():
    transform_key = config['transform_key']

run_time = str(time.strftime('%m-%d_%H',time.localtime(time.time())))
args.log_filename = "%s_%s_%s" % (args.dataset,args.mode,run_time)

best_epoch = args.nb_epochs
pt_path = args.pt_path

model = config['model']['type'](dim3=dim3,dim4=dim4)
if not args.apex:
    model = torch.nn.DataParallel(model)
model = model.cuda()




def save_best_checkpoint(model):
    torch.save(model.state_dict(), './results/' + args.log_filename + '.pt')



def load_best_checkpoint(model,pt_path):
    #pt = os.path.join('results',os.path.join(args.dataset, pt_name))
    model.load_state_dict(torch.load(pt_path),strict=False)
    model = model.cuda()
    return model



if args.mode == 'train':
    train_results_fn = "log/%s_%s_%d.json" % (args.dataset,'train', args.seed)
    if os.path.exists(train_results_fn):
        with open(train_results_fn, 'r') as f:
            train_results = json.load(f)
        args.lr_steps = train_results['lr_steps']
        best_epoch = train_results['best_epoch']

train_transform = dataset.utils.make_transform(
            **dataset_config[transform_key]
        )
print('best_epoch', best_epoch)

results = {}

if ('inshop' not in args.dataset ): 
    dl_ev = torch.utils.data.DataLoader(
        dataset.load(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            )
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )
else:
    #inshop trainval mode 
    dl_query = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            ),
            dset_type = 'query'
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )
    dl_gallery = torch.utils.data.DataLoader(
        dataset.load_inshop(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['eval'],
            transform = dataset.utils.make_transform(
                **dataset_config[transform_key],
                is_train = False
            ),
            dset_type = 'gallery'
        ),
        batch_size = args.sz_batch,
        shuffle = False,
        num_workers = args.nb_workers,
        #pin_memory = True
    )


logging.basicConfig(
    format="%(asctime)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("log/{0}.log".format(args.log_filename.split(".")[0])),
        logging.StreamHandler()
    ]
)

if args.mode == 'train' or args.mode == 'test':
    tr_dataset = dataset.load(
            name = args.dataset,
            root = dataset_config['dataset'][args.dataset]['root'],
            source = dataset_config['dataset'][args.dataset]['source'],
            classes = dataset_config['dataset'][args.dataset]['classes']['train'],
            transform = train_transform
        )



num_class_per_batch = config['num_class_per_batch']
num_gradcum = config['num_gradcum']
is_random_sampler = config['is_random_sampler']


if is_random_sampler:
    batch_sampler = dataset.utils.RandomBatchSampler(tr_dataset.ys, args.sz_batch, True, num_class_per_batch, num_gradcum)
else:
    
    batch_sampler = dataset.utils.BalancedBatchSampler(torch.Tensor(tr_dataset.ys), num_class_per_batch, int(args.sz_batch / num_class_per_batch))


dl_tr = torch.utils.data.DataLoader(
    tr_dataset,
    batch_sampler = batch_sampler,
    num_workers = args.nb_workers,
    #pin_memory = True
)


'''
    criterion3 ==> loss-3 in paper
    criterion4 ==> loss-4 in paper
'''
criterion2 = config['criterion']['type'](
    nb_classes = dl_tr.dataset.nb_classes(),
    sz_embed = int(341),
    **config['criterion']['args']
).cuda()
criterion3 = config['criterion']['type'](
    nb_classes = dl_tr.dataset.nb_classes(),
    sz_embed = int(dim3),
    **config['criterion']['args']
).cuda()
criterion4 = config['criterion']['type'](
    nb_classes = dl_tr.dataset.nb_classes(),
    sz_embed = int(dim4),
    **config['criterion']['args']
).cuda()



opt_warmup = config['opt']['type'](
    [
        { 
            **{'params': list(model.parameters()
                )
            }, 
            'lr': 0
        },
        { 
            **{'params': criterion4.parameters()}
            ,
            **config['opt']['args']['proxynca']
        },
        {
            **{'params': criterion3.parameters()}
            ,
            **config['opt']['args']['proxynca']
        },
{
            **{'params': criterion2.parameters()}
            ,
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base'] 
)

opt = config['opt']['type'](
    [
        { 
            **{'params': list(model.parameters()
                )
            }, 
            **config['opt']['args']['backbone']
        },
        { 
            **{'params': criterion4.parameters()},
            **config['opt']['args']['proxynca']
        },
        {
            **{'params': criterion3.parameters()}
            ,
            **config['opt']['args']['proxynca']
        },
{
            **{'params': criterion2.parameters()}
            ,
            **config['opt']['args']['proxynca']
        },

    ],
    **config['opt']['args']['base'] 
)

if args.apex:
    [model, criterion], [opt, opt_warmup, opt_warmup1] = amp.initialize([model, criterion], [opt, opt_warmup, opt_warmup1], opt_level='O1')
    model = torch.nn.DataParallel(model)
if args.mode == 'test':
    with torch.no_grad():
        logging.info("**Evaluating...(test mode)**")
        model = load_best_checkpoint(model,pt_path)
        if 'inshop' in args.dataset:
            utils.evaluate_inshop(model, dl_query, dl_gallery)
        else:
            utils.evaluate(model, dl_ev, args.recall)
    exit()

if args.mode == 'train':
    scheduler = config['lr_scheduler2']['type'](
        opt,
        milestones=args.lr_steps,
        gamma=0.1
    )

logging.info("Training parameters: {}".format(vars(args)))
logging.info("Training for {} epochs.".format(args.nb_epochs))
losses = []
scores = []
scores_tr = []

t1 = time.time()




it = 0

def batch_lbl_stats(y):
    print(torch.unique(y))
    kk = torch.unique(y)
    kk_c = torch.zeros(kk.size(0))
    for kx in range(kk.size(0)):
        for jx in range(y.size(0)):
            if y[jx] == kk[kx]:
                kk_c[kx] += 1

def get_centers(dl_tr):
    c_centers = torch.zeros(dl_tr.dataset.nb_classes(), args.sz_embedding).cuda()
    n_centers = torch.zeros(dl_tr.dataset.nb_classes()).cuda()
    for ct, (x,y,_) in enumerate(dl_tr):
        with torch.no_grad():
            m = model(x.cuda())
        for ix in range(m.size(0)):
            c_centers[y] += m[ix]
            n_centers[y] += 1
    for ix in range(n_centers.size(0)):
        c_centers[ix] = c_centers[ix] / n_centers[ix]
    
    return c_centers

prev_lr = opt.param_groups[0]['lr']
lr_steps = []


if not args.no_warmup:
    logging.info("**warm up for %d epochs.**" % args.warmup_k)
    for e in range(0,args.warmup_k):
        for ct, (x, y, _, _) in enumerate(dl_tr):
            opt_warmup.zero_grad()
            with torch.no_grad():
                embedding2,embedding3, embedding4 = model(x.cuda())
            loss2 = criterion2(embedding2, y.cuda())
            loss3 = criterion3(embedding3, y.cuda())
            loss4 = criterion4(embedding4, y.cuda())
            loss = loss2 + loss3 + loss4
            #loss = loss3 + loss4
            '''
            loss2 = criterion2(embedding2, y.cuda())
            #loss3 = criterion3(embedding3,y.cuda())
            # loss3 = criterion3(embedding3, y.cuda())
            loss4 = criterion4(embedding4, y.cuda())
            # loss = loss3 + loss4 + loss2
            loss = loss2 + loss4
            '''
            if args.apex:
                with amp.scale_loss(loss, opt_warmup) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 10)
            opt_warmup.step()
        logging.info('warm up ends in %d epochs' % (args.warmup_k - e))


for e in range(0, args.nb_epochs+10):
    print("epoch:",e+1," training...")
    time_per_epoch_1 = time.time()
    losses_per_epoch = []
    tnmi = []

    opt.zero_grad()
    for ct, (x, y, _, _) in enumerate(dl_tr):
        it += 1
        embedding2, embedding3, embedding4 = model(x.cuda())
        loss2 = criterion2(embedding2, y.cuda())
        loss3 = criterion3(embedding3, y.cuda())
        loss4 = criterion4(embedding4, y.cuda())
        loss = loss2 + loss3 + loss4
        #loss = loss2 + loss3 + loss4
        if args.apex:
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
       
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())

        if (ct + 1) % 1 == 0:
            opt.step()
            opt.zero_grad()

        

    time_per_epoch_2 = time.time()
    losses.append(np.mean(losses_per_epoch[-20:]))
    print('it: {}'.format(it))
    print(opt)
    logging.info(
        "Epoch: {}, loss: {:.3f}, time (seconds): {:.2f}.".format(
            e,
            losses[-1],
            time_per_epoch_2 - time_per_epoch_1
        )
    )

    model.losses = losses
    model.current_epoch = e

    #if e == best_epoch:
    #    break

    if args.mode =='train' and ((e+1)>=args.nb_epochs-20 or e==0) :
        with torch.no_grad():
            logging.info("**Validation... epoch:"+str(e+1)+"**")
            if 'inshop' in args.dataset:
                (best_test_r1, best_test_r10, best_test_r20, best_test_r30, best_test_r40,
                                best_test_r50) = utils.evaluate_inshop(model, dl_query, dl_gallery)
                results['R1'] = best_test_r1
                results['R10'] = best_test_r10
                results['R20'] = best_test_r20
                results['R30'] = best_test_r30
                results['R40'] = best_test_r40
                results['R50'] = best_test_r50
            elif 'sop' in args.dataset:
                (best_test_r1, best_test_r10, best_test_r100, best_test_r1000) = utils.evaluate(model, dl_ev, args.recall)
                results['R1'] = best_test_r1
                results['R10'] = best_test_r10
                results['R100'] = best_test_r100
                results['R1000'] = best_test_r1000
            else:
                (best_test_r1, best_test_r2, best_test_r4, best_test_r8) = utils.evaluate(model, dl_ev, args.recall)
                results['R1'] = best_test_r1
                results['R2'] = best_test_r2
                results['R4'] = best_test_r4
                results['R8'] = best_test_r8
                print(str(best_test_r1)+"_"+str(best_test_r2)+"_"+str(best_test_r4)+"_"+str(best_test_r8))
            #torch.save(model.state_dict(),'results/'+str(args.dataset)+"/"+str(e+1)+".pt")
            torch.save(model.state_dict(),'./log/'+str(e+1)+".pt")

        scheduler.step(e)



with open(out_results_fn,'w') as outfile:
    json.dump(results, outfile)

t2 = time.time()
logging.info("Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
