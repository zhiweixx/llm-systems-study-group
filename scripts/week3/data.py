"""Serving replicas, data parallel training, and state sharding."""
from .common import *

DDP = ('PyTorch: DDP design', 'https://docs.pytorch.org/docs/stable/notes/ddp.html')
FSDP = ('PyTorch: FSDP2 execution', 'https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#how-fsdp2-works')
ZERO = ('ZeRO paper, §3', 'https://arxiv.org/abs/1910.02054')
PLAY = ('Ultra-Scale: ZeRO memory', 'https://nanotron-ultrascale-playbook.static.hf.space/index.html#zero-redundancy-optimizer-zero')
COLL = ('NCCL: collective operations', 'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html')
SERVE = ('vLLM: data-parallel deployment', 'https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/')


def get_replica_slides():
    body = text(75,200,'Inference: place the same trained model on two GPUs; route different requests to each.',29)
    body += label_box(590,239,420,60,'Incoming requests A, B, C, D')
    body += arrow(695,299,427,352) + arrow(905,299,1175,352)
    for x, gpu, requests, kv in [(100,'GPU 0','Requests A, B','KV cache for A and B'),(855,'GPU 1','Requests C, D','KV cache for C and D')]:
        body += rect(x,354,645,285) + text(x+25,396,gpu,31,700,color=BLUE)
        body += label_box(x+25,422,595,65,'Full model weights W',size=29)
        body += text(x+25,536,requests,29,600) + text(x+25,581,kv,29)
        body += text(x+25,618,'Independent request scheduler',25,color=MUTED)
    body += text(75,695,'Each replica can itself use several GPUs through tensor or pipeline parallelism.',28)
    body += takeaway('Replicas add request capacity; one request still runs on one model instance.',
                      'Dense-model serving here: no backward pass and no gradient synchronization between replicas.')
    return [slide('Serving replicas divide requests across model instances', body,
        'Use a dense, read-only model. Both replicas start from the same trained checkpoint. The routing arrows assign whole requests, not different tokens of one request. A replica owns the KV cache of its assigned requests; the two replica caches do not automatically form one larger shared cache. More replicas can increase aggregate throughput if there is demand and the router balances it, but do not intrinsically shorten one request’s computation. A model instance may be a TP or PP group rather than a single GPU. Some MoE deployments couple data-parallel workers through expert parallelism; that is a separate configuration discussed later.',
        sources=[SERVE], section='Serving replicas')]


def _ddp_slide():
    body = text(75,198,'Training: split a global batch into local minibatches, but start from the same model.',29)
    for x,gpu,batch,g in [(100,'GPU 0','Examples 1–4','g₀'),(855,'GPU 1','Examples 5–8','g₁')]:
        body += rect(x,237,645,307) + text(x+25,279,gpu,31,700,color=BLUE)
        body += text(x+25,324,'Full weights W + optimizer state',29,600)
        body += label_box(x+25,346,595,58,batch,size=28)
        body += text(x+25,445,'Forward → local mean loss → backward',27)
        body += text(x+25,498,'Local gradient '+g,29,600)
    body += arrow(420,544,420,573) + arrow(1177,544,1177,573)
    body += label_box(275,573,1050,68,'Average gradients: g = (g₀ + g₁) / 2',size=31)
    body += text(800,690,'Both GPUs apply the same optimizer update to the same W.',30,600,anchor='middle')
    body += takeaway('DDP splits the training examples; each GPU still holds the whole model.',
                      'Equal-sized minibatches and local mean losses assumed. DDP = DistributedDataParallel.')
    return slide('Data-parallel training must keep the model copies consistent',body,
        'DDP means DistributedDataParallel. Each worker processes different examples with an identical model and optimizer state. Backward produces a local gradient; synchronizing gradients makes the following optimizer step consistent across workers. The diagram assumes equal numbers of equally weighted training targets and local mean losses. In language modeling, different valid-token counts can require weighted normalization for the intended global token-mean objective. DDP can overlap gradient communication with backward through gradient buckets; this diagram shows the logical dependency, not a serialized implementation. The input batch must be partitioned by the data pipeline.',
        sources=[DDP],section='Data parallelism')


def _gradient_slide():
    body = text(75,196,'Toy model: two parameters, W = [10, 20]; SGD learning rate = 0.1.',30)
    body += text(75,244,'Each GPU computes a mean gradient from the same number of training examples.',28)
    body += table(75,279,1450,['Step','GPU 0','GPU 1'],[
        ['Local backward','g₀ = [2, 4]','g₁ = [6, 8]'],
        ['AllReduce(SUM)','[8, 12]','[8, 12]'],
        ['Divide by 2 workers','g = [4, 6]','g = [4, 6]'],
        ['SGD: W ← W − 0.1g','W = [9.6, 19.4]','W = [9.6, 19.4]'],
    ],ratios=[.39,.305,.305],row_h=75,size=28)
    body += text(75,712,'Without synchronization: GPU 0 gets [9.8, 19.6]; GPU 1 gets [9.4, 19.2].',28,color=WARM)
    body += takeaway('Averaging local gradients gives the gradient of the combined mean loss.',
                      'SUM alone does not compute a mean. DDP supplies the appropriate averaging by default.')
    return slide('The optimizer update explains why DDP averages gradients',body,
        'All numbers are an authored arithmetic example. The model is deliberately a two-parameter vector so the operation is visible. Local gradients [2,4] and [6,8] are supplied, not derived from a particular network. Sum and division are shown separately for clarity; a backend may combine scaling with communication. Both workers must start from equal weights and compatible optimizer state. With Adam rather than SGD, matching gradients and matching moment states similarly lead to matching updates. Unequal local loss denominators change the correct weighting; equal worker averaging is the intended objective only under the stated assumptions.',
        sources=[DDP,COLL],section='Data parallelism')


def _shard_strip(x,y,w,sharded):
    # Four parameter-index ranges; only one range persists on this illustrative GPU when sharded.
    out=''
    for j in range(4):
        fill=PALE if not sharded or j==0 else 'white'
        out+=rect(x+j*w/4,y,w/4-4,46,fill,LINE)
        out+=text(x+(j+.5)*w/4-2,y+32,str(j) if not sharded or j==0 else '—',25,600,
                  color=INK if not sharded or j==0 else MUTED,anchor='middle')
    return out


def _zero_slide():
    body = text(75,199,'ZeRO (Zero Redundancy Optimizer): state ownership on GPU 0 of 4.',30)
    body += text(75,245,'Each numbered segment covers a different quarter of the model’s parameter indices.',28)
    body += table(75,280,1450,['Method','Model weights','Gradients','Optimizer state'],[],ratios=[.19,.27,.27,.27],row_h=62,size=28)
    for i,(name,flags) in enumerate([('DDP',(False,False,False)),('ZeRO-1',(False,False,True)),('ZeRO-2',(False,True,True)),('ZeRO-3',(True,True,True))]):
        yy=362+i*79
        body += text(88,yy+31,name,29,700,color=BLUE)
        for j,shard in enumerate(flags): body += _shard_strip(360+j*391.5,yy,345,shard)
        body += line(75,yy+61,1525,yy+61,'#c7cdd1')
    body += text(75,710,'GPU 1 retains shard 1, GPU 2 shard 2, and GPU 3 shard 3 wherever state is partitioned.',27)
    body += takeaway('ZeRO removes duplicated training state across data-parallel workers.',
                      'State ownership is shown between computations; temporary full parameters can still be needed.')
    return slide('ZeRO progressively shards the state that DDP replicates',body,
        'ZeRO stages progressively partition optimizer state, gradients, and parameters across the data-parallel group. Each strip represents the same parameter-index ranges, not layers. The stage-3 diagram is a persistent ownership view: it does not mean a local matrix multiplication can execute using an arbitrary quarter of its required weights. Workers reconstruct parameters when needed. This differs from tensor parallelism, which explicitly partitions the layer computation. ZeRO does not divide the local-example activation memory by the data-parallel degree. Fully sharded data parallel implementations follow a similar state-sharding principle.',
        sources=[ZERO],section='Training-state sharding')


def _budget_slide():
    body = text(75,197,'Example: 8B parameters, 4 GPUs; one explicit mixed-precision Adam layout.',29)
    body += lines(75,242,[
        'Weights: BF16, 2 bytes/parameter.  Gradients: BF16, 2 bytes/parameter.',
        'Optimizer state: FP32 master weights (4) + Adam moments m and v (8) = 12 bytes.'
    ],size=27,gap=42)
    body += table(75,316,1450,['Per GPU','Weights','Gradients','Optimizer','Total'],[
        ['DDP','16 GB','16 GB','96 GB','128 GB'],
        ['ZeRO-1','16 GB','16 GB','96 / 4 = 24 GB','56 GB'],
        ['ZeRO-2','16 GB','16 / 4 = 4 GB','24 GB','44 GB'],
        ['ZeRO-3','16 / 4 = 4 GB','4 GB','24 GB','32 GB'],
    ],ratios=[.18,.205,.215,.25,.15],row_h=66,size=27)
    body += text(75,701,'The 32 GB figure does not include activations, gathered parameters, or communication buffers.',27,color=WARM)
    body += takeaway('Budget peak memory, not just the persistent state divided by the GPU count.',
                      'Decimal GB. This is a specified layout, not the memory usage of every Adam/FSDP configuration.')
    return slide('Sharding the same 8B model: 128 → 56 → 44 → 32 GB per GPU',body,
        'The numbers are derived from the same illustrative state layout used in Week 1. For P=8 billion parameters, 2P bytes is 16 GB, 12P is 96 GB, and 16P is 128 GB. Dividing only optimizer state gives 16+16+24=56 GB; additionally dividing gradients gives 16+4+24=44 GB; dividing all persistent state gives 4+4+24=32 GB. This is not an exact prediction for PyTorch FSDP2: master-parameter representation, gradient precision, accumulation, buffer reuse, and allocator behavior vary. Activations scale with the local workload. Gathering and prefetching introduce temporary allocations, so a 32 GB device is not sufficient merely because this state budget equals 32 GB.',
        sources=[PLAY],section='Training-state sharding')


def _fsdp_collectives_slide():
    body = text(75,198,'Fully Sharded Data Parallel (FSDP): gather a layer’s weights only when needed.',29)
    body += text(75,256,'Before a layer: AllGather its parameter shards',31,700,color=BLUE)
    body += table(75,281,1450,['Worker','Stored parameter shard','After AllGather'],[
        ['GPU 0','[w₀, w₁]','[w₀, w₁, w₂, w₃]'],
        ['GPU 1','[w₂, w₃]','[w₀, w₁, w₂, w₃]'],
    ],ratios=[.19,.38,.43],row_h=58,size=27)
    body += text(75,503,'After backward: ReduceScatter gradients, then scale for the mean',30,700,color=BLUE)
    body += table(75,525,1450,['Worker','Local full gradient','Mean-gradient shard retained'],[
        ['GPU 0','[1, 2, 3, 4]','[3, 4]'],
        ['GPU 1','[5, 6, 7, 8]','[5, 6]'],
    ],ratios=[.19,.38,.43],row_h=58,size=27)
    body += takeaway('AllGather assembles pieces; ReduceScatter adds contributions and distributes pieces.',
                      'Toy mean: ([1,2,3,4] + [5,6,7,8]) / 2 = [3,4,5,6]. Equal local loss weighting.')
    return slide('Gather weights for computation; scatter gradients for the update',body,
        'AllGather copies disjoint parameter slices into a full tensor on every participating GPU; it does not sum those slices. ReduceScatter sums matching gradient entries across workers and leaves different output slices on different workers. In this authored example, the summed vector is [6,8,10,12], with sum shards [6,8] and [10,12]. Division by two produces the displayed mean shards. The optimizer on GPU 0 needs only the first gradient shard because it owns the first parameter and optimizer-state shard; GPU 1 updates the second shard. Scaling can be implemented within or around the collective.',
        sources=[COLL,FSDP],section='Training-state sharding')


def _fsdp_trace_slide():
    body = text(75,195,'One fully sharded Transformer layer; two GPUs process different local minibatches.',29)
    body += text(75,239,'Trace the layer’s weights over one training step (reshard after forward).',28)
    # Logical trace, not an elapsed-time chart: each box is an event, not a proportional duration.
    events=[('1  Forward','AllGather weights','Compute on local data','Release full weights'),
            ('2  Backward','AllGather weights again','Compute local gradients','ReduceScatter gradients'),
            ('3  Optimizer','Keep local state shard','Update owned parameters','Next step gathers updates')]
    for i,(title,a,b,c) in enumerate(events):
        x=75+i*495
        body += rect(x,281,460,282)
        body += rect(x,281,460,61,PALE)
        body += text(x+22,322,title,30,700,color=BLUE)
        body += lines(x+22,390,[a,b,c],size=27,gap=61)
        if i<2: body += arrow(x+460,421,x+487,421)
    body += line(75,599,1525,599)
    body += text(75,643,'Peak memory = persistent shards + live activations + temporary full tensors/buffers.',28,600)
    body += text(75,696,'Prefetching can hide communication, but keeps more gathered weights live at once.',28)
    body += takeaway('FSDP changes when full weights are present; each worker still computes its own examples.',
                      'TP instead splits one layer’s computation across GPUs. The two techniques can be combined.')
    return slide('FSDP saves memory by gathering weights when needed',body,
        'This is a logical dependency trace for one layer, not a proportional timing chart. The example reshards weights after forward, so backward must gather them again. Real schedules can prefetch the next layer while the current layer computes, and can keep some parameters unsharded depending on configuration. Other layers remain sharded while this layer executes. Temporary full weights, full local gradients before reduction, and communication workspaces contribute to peak memory. Data-parallel workers still process distinct local data. A separate tensor-parallel group can partition the layer computation inside each data-parallel model instance.',
        sources=[FSDP,PLAY],section='Training-state sharding')


def get_training_slides():
    return [_ddp_slide(),_gradient_slide(),_zero_slide(),_budget_slide(),_fsdp_collectives_slide(),_fsdp_trace_slide()]


def get_slides():
    return get_replica_slides()+get_training_slides()
