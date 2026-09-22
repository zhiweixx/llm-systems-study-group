"""Figure-led overview slides for DP, serving replicas, ZeRO, and FSDP.

This module belongs only to the separate overview deck. The full deck is unchanged.
"""
from .common import *

DDP = ('PyTorch DDP', 'https://docs.pytorch.org/docs/stable/notes/ddp.html')
SERVE = ('vLLM: data-parallel serving', 'https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/')
ZERO = ('ZeRO paper', 'https://arxiv.org/abs/1910.02054')
FSDP = ('PyTorch FSDP2', 'https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works')
COLL = ('NCCL collectives', 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html')
TPALE = '#d6e6de'
BPALE = '#d8e6ef'


def _training():
    body = text(75,194,'Same initial weights. Different training examples.',31)
    for gpu,x,fill,examples,g in [(0,100,PALE,'Examples A, B','g₀'),(1,865,TPALE,'Examples C, D','g₁')]:
        body += rect(x,237,635,283,'white')
        body += text(x+26,282,f'GPU {gpu}',31,700,color=BLUE)
        body += label_box(x+25,309,230,67,examples,fill,30)
        body += arrow(x+270,342,x+339,342)
        body += rect(x+355,309,250,100,PALE)
        body += text(x+480,348,'Full model W',30,600,anchor='middle')
        body += text(x+480,391,'Forward/backward',28,anchor='middle')
        body += arrow(x+480,419,x+319,451)
        body += text(x+319,489,'Local gradient '+g,31,600,anchor='middle')
    body += arrow(418,521,610,562)+arrow(1183,521,990,562)
    body += label_box(480,562,640,76,'Average: g = (g₀ + g₁) / 2',PALE,34)
    body += arrow(610,638,418,676)+arrow(990,638,1183,676)
    body += text(418,711,'Same update → W′',31,700,color=BLUE,anchor='middle')
    body += text(1183,711,'Same update → W′',31,700,color=BLUE,anchor='middle')
    body += takeaway('Data-parallel training synchronizes gradients to keep model copies consistent.',
                      'Equal-sized local minibatches and local mean losses assumed.')
    return slide('Data-parallel training (DDP)',body,
        'Read the diagram from the local examples at the top to the matching updated weights at the bottom. GPU 0 receives examples A and B, while GPU 1 receives C and D. Each worker has a complete copy of the same weights and runs its own forward and backward computation. Their gradients differ because their examples differ. The average gradient is then available to both workers, so identical initial parameters and optimizer state produce identical updates. An AllReduce with sum followed by division by two implements this average; the diagram shows its semantics, not a required implementation order. Equal local example weighting and local mean losses are assumed. Different valid-token counts in language modeling may require weighting by those counts to represent the desired global token-mean loss. DDP does not partition the model weights across workers. Production implementations can overlap gradient communication with backward through gradient buckets. The boxes show ownership and dependencies rather than elapsed time.',
        [DDP],section='Data parallelism')


def _replicas():
    body = text(75,194,'Dense-model inference: route whole requests to independent model copies.',31)
    body += text(800,271,'Incoming requests',31,600,anchor='middle')
    for i,label in enumerate(['A','B','C','D']):
        body += label_box(605+i*100,294,90,58,label,PALE if i<2 else TPALE,31)
    body += arrow(680,361,418,421)+arrow(930,361,1183,421)
    for gpu,x,requests,fill in [(0,100,'A, B',PALE),(1,865,'C, D',TPALE)]:
        body += rect(x,423,635,268,'white')
        body += text(x+26,467,f'GPU {gpu}',31,700,color=BLUE)
        body += label_box(x+25,491,585,71,'Full model weights W',PALE,32)
        body += label_box(x+25,582,270,67,'Requests '+requests,fill,30)
        body += label_box(x+315,582,295,67,'Own request KV',fill,29)
    body += takeaway('Replicas serve more independent requests without per-token gradient synchronization.',
                      'Each request stays with its model instance and its KV cache in this example.')
    return slide('Data parallelism for serving',body,
        'Both GPUs load the same trained dense-model checkpoint. The dispatcher assigns whole requests A and B to GPU 0 and C and D to GPU 1. Each worker then performs prefill and decode for its own requests, maintaining their request-specific KV caches. These are independent model instances: inference does not compute training gradients or synchronize those gradients after a token. Additional replicas can increase aggregate service capacity when there is sufficient traffic and balanced routing, but they do not divide a single request’s layer computation across the two GPUs. Each replica stores a full weight copy, so replication uses more aggregate weight memory. In a larger deployment, one model instance could itself be a tensor- or pipeline-parallel group. The figure deliberately assumes ordinary dense inference. MoE deployments can instead replicate attention while distributing experts across workers, which creates communication between otherwise separate request groups. That expert-parallel arrangement appears later in the overview.',
        [SERVE],section='Serving replicas')


def _state_bar(x,y,owner,sharded):
    out=''
    for part in range(4):
        local=not sharded or part==owner
        color=TPALE if sharded and local else BPALE if local else 'white'
        out += rect(x+part*59,y,56,21,color,TEAL if sharded and local else LINE)
    return out


def _zero():
    body = text(75,192,'P = weights     G = gradients     O = optimizer state',31)
    body += text(75,239,'Persistent storage on four data-parallel GPUs',29,color=MUTED)
    xs=[367,657,947,1237]
    for gpu,x in enumerate(xs):body += text(x+116,292,f'GPU {gpu}',30,700,color=BLUE,anchor='middle')
    for row,(label,shards) in enumerate([
        ('DDP',(False,False,False)),('ZeRO-1',(False,False,True)),
        ('ZeRO-2',(False,True,True)),('ZeRO-3',(True,True,True))]):
        yy=318+row*89
        body += text(75,yy+46,label,31,700,color=BLUE)
        for state,(name,sharded) in enumerate(zip(['P','G','O'],shards)):
            body += text(310,yy+19+state*24,name,28,600,anchor='middle')
            for gpu,x in enumerate(xs): body += _state_bar(x,yy+state*24,gpu,sharded)
        if row<3:body += line(75,yy+77,1525,yy+77,'#d9e1e5')
    body += rect(75,695,28,21,TPALE,TEAL)+text(118,715,'Filled = local storage',28)
    body += rect(498,695,28,21,'white',LINE)+text(541,715,'Outline = stored elsewhere',28)
    body += takeaway('ZeRO progressively shards optimizer state, gradients, and then weights.',
                      'Equal blocks represent parameter ranges, not byte sizes. Full parameters may be gathered temporarily.')
    return slide('ZeRO: less replicated training state',body,
        'ZeRO means Zero Redundancy Optimizer. Each horizontal group of four small blocks represents four disjoint parameter-index ranges. Read a row across all four GPUs. In DDP, every GPU retains all weight, gradient, and optimizer ranges. In ZeRO-1, each GPU retains only its assigned optimizer range. ZeRO-2 also distributes the gradient ranges. ZeRO-3 distributes the parameter ranges as well. Pale filled blocks are local allocations; white outlines represent ranges held by other workers. Every sharded state collectively remains complete across the group. The blocks have equal schematic size to show ownership, not bytes: actual optimizer state can occupy much more memory than low-precision parameters. The view describes persistent state between computations, not the instantaneous memory peak. Gathering parameters, holding activations, and communicating gradients can need additional allocations. The examples processed by each data-parallel worker still differ. FSDP on the next slide shows how a worker can execute a layer when that layer’s weights are persistently distributed.',
        [ZERO],section='Training-state sharding')


def _weight_pair(x,y,owner=None):
    out=''
    for part in range(2):
        local=owner is None or part==owner
        fill=([BPALE,TPALE][part] if local else 'white')
        out+=rect(x+part*116,y,112,62,fill,LINE)
        if local:out+=text(x+part*116+56,y+42,'W₀' if part==0 else 'W₁',30,600,anchor='middle')
    return out


def _fsdp():
    body=text(75,193,'FSDP = Fully Sharded Data Parallel. Trace one layer’s weights.',31)
    centers=[310,635,960,1285]
    for x,label in zip(centers,['Stored shards','AllGather','Compute','Reshard']):
        body+=text(x,275,label,31,700,color=BLUE,anchor='middle')
    for gpu,y in [(0,327),(1,463)]:
        body+=text(75,y+40,f'GPU {gpu}',30,700,color=BLUE)
        for step,cx in enumerate(centers):
            body+=_weight_pair(cx-114,y,gpu if step in (0,3) else None)
            if step<3:body+=arrow(cx+124,y+31,cx+194,y+31)
        body+=text(960,y+99,'Examples A, B' if gpu==0 else 'Examples C, D',28,anchor='middle')
    body+=line(75,588,1525,588)
    body+=text(75,635,'Backward',30,700,color=BLUE)
    for x,w,label in [(281,369,'Gather + compute'),(748,336,'ReduceScatter'),(1182,340,'Update local shards')]:
        body+=label_box(x,603,w,68,label,PALE,29)
    body+=arrow(660,637,736,637)+arrow(1094,637,1170,637)
    body+=text(916,713,'Combine gradients; keep a shard',27,color=MUTED,anchor='middle')
    body+=takeaway('Gather the current layer, compute locally, and release the full weights.',
                    'The other layers stay sharded. Live activations and temporary tensors add to peak memory.')
    return slide('FSDP gathers one layer at a time',body,
        'The two colored blocks are parameter shards W0 and W1 of one layer, not two entire model layers. GPU 0 owns W0 and GPU 1 owns W1. AllGather reconstructs this layer’s full weights on both workers without adding the values together. Each then runs the layer on different local examples, as in data-parallel training. In the depicted configuration, workers release the full gathered copy after forward and retain only their owned shard. Backward therefore gathers this layer again, computes local gradients, and uses ReduceScatter to combine gradient contributions while retaining a different gradient shard on each worker. Each optimizer updates its corresponding parameter shard and optimizer-state shard. This logical trace omits other layers and communication overlap. Frameworks can prefetch another layer or retain selected full tensors, trading extra live memory for less exposed communication. The diagram explains why sharded persistent storage reduces memory but does not equal peak memory: live activations and temporary gathered tensors still occupy space.',
        [FSDP,COLL],section='Training-state sharding')


def get_slides():
    return [_training(),_replicas(),_zero(),_fsdp()]
