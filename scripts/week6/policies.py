"""Slides 12–22: visual-language policies and the original pi0.5 system.

All diagrams and equations are editable SVG/MathML. Published evaluation
plots are embedded separately, with their original metrics and scope.
"""
from .common import *

PALIGEMMA = ('PaliGemma §3', 'https://arxiv.org/html/2407.07726v1#S3')
RT1 = ('RT-1 §5', 'https://robotics-transformer1.github.io/assets/rt1.pdf')
RT2 = ('RT-2 §3', 'https://robotics-transformer2.github.io/assets/rt2.pdf')
OPENVLA = ('OpenVLA §3', 'https://arxiv.org/html/2406.09246v3#S3')
PI0 = ('π0 v4 §IV', 'https://arxiv.org/html/2410.24164v4#S4')
PI05 = ('π0.5 paper', 'https://www.physicalintelligence.company/download/pi05.pdf')
PI05_BLOG = ('π0.5 explanation', 'https://www.pi.website/blog/pi05')
OPENPI = ('OpenPI release', 'https://github.com/Physical-Intelligence/openpi')
CS_IL = ('CS224R, lecture 2', 'https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf')
CS_RL = ('CS224R, lecture 17', 'https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf')


def mi(s): return '<mi>' + esc(s) + '</mi>'
def mn(s): return '<mn>' + esc(str(s)) + '</mn>'
def mo(s): return '<mo>' + esc(s) + '</mo>'
def mt(s): return '<mtext>' + esc(s) + '</mtext>'
def row(s): return '<mrow>' + s + '</mrow>'
def sub(a, b): return '<msub>' + row(a) + row(b) + '</msub>'
def sup(a, b): return '<msup>' + row(a) + row(b) + '</msup>'
def par(s): return row(mo('(') + s + mo(')'))
def norm2(s): return sup(row(mo('‖') + s + mo('‖')), mn(2))
def at(): return sub(mi('A'), mi('t'))
def ot(): return sub(mi('o'), mi('t'))
def ct(): return sub(mi('c'), mi('t'))
def xt(): return sub(mi('X'), mi('τ'))
def vel(): return sub(mi('v'), mi('θ'))


def module(x, y, w, h, heading, detail=(), color=BLUE, fill=PALE):
    """A labeled computational module, rather than a decorative text card."""
    b = rect(x, y, w, h, fill, LINE)
    b += text(x+w/2, y+40, heading, 29, 700, color=color, anchor='middle')
    if detail:
        b += lines(x+w/2, y+79, detail, 25, gap=34, color=INK, anchor='middle')
    return b


def token_strip(x, y, labels, width=56, height=39, fill=PALE):
    b = ''
    for k, label in enumerate(labels):
        b += rect(x+k*(width+5), y, width, height, fill)
        b += text(x+k*(width+5)+width/2, y+27, label, 23, anchor='middle')
    return b


def vlm():
    b = text(75,190,'Vision–language model (VLM): images + text → an answer.',30)
    b += text(1525,190,'PaliGemma-style architecture',24,color=MUTED,anchor='end')
    b += module(75,250,225,110,'Image', ['pixels Iₜ'])
    b += arrow(300,305,355,305)
    b += module(355,250,250,110,'SigLIP encoder',['patch features'])
    b += arrow(605,305,660,305)
    b += module(660,250,230,110,'Projection',['LM dimensions'])
    b += arrow(890,305,950,305)
    b += module(950,250,285,290,'Gemma decoder',[],fill='#e7f0ed')
    b += token_strip(980,313,['I','I','…'],59,40,fill='#dbe8f0')
    b += token_strip(980,378,['Q','…'],59,40,fill='#e2ece8')
    b += lines(1092,471,['Image + question','form the prefix'],26,gap=34,anchor='middle')
    b += module(75,430,225,110,'Question',['where is the cup?'])
    b += arrow(300,485,355,485)
    b += module(355,430,535,110,'Tokenizer + text embeddings',['question tokens in the LM embedding space'])
    b += arrow(890,485,950,485)
    b += arrow(1235,390,1290,390)
    b += lines(1400,365,['“The cup is','on the table.”'],28,gap=38,anchor='middle')
    b += text(1400,455,'Generated answer',24,color=MUTED,anchor='middle')
    b += line(75,587,1525,587)
    b += text(75,637,'Training examples teach captioning, question answering, and structured visual outputs.',29)
    b += text(75,688,'The answer is an illustrative text output. Robot command units and controllers enter next.',26,color=MUTED)
    b += takeaway('Visual and language tokens share a model that can transfer across prediction tasks.',
                   'PaliGemma uses full attention within its input prefix and autoregressive attention for the answer.')
    return slide('A VLM connects images to language representations', b,
        'This is an editable architectural illustration, not a model-generated example. PaliGemma combines the SigLIP ViT-So400m vision encoder, a linear projection, and Gemma-2B. The projection aligns feature dimensions, rather than converting image patches into vocabulary IDs. Its image and prompt prefix uses bidirectional attention, while the generated answer suffix uses autoregressive attention. The model can emit natural language and serialized structured outputs such as boxes. Its base pretraining is not chat instruction tuning. Recognition and semantic representations do not by themselves specify a robot action coordinate system, scale, or execution controller. OpenVLA uses a different VLM backbone: Prismatic, with DINOv2 and SigLIP features plus Llama 2. The schematic introduces the reusable three-part VLM structure, not a claim that every VLA uses PaliGemma. Source: PaliGemma §3.1 and Figures 1–2.',
        [PALIGEMMA,OPENVLA], 'VLMs and robot actions')


def discrete_vla():
    b = text(75,190,'Vision–language–action (VLA): learn from images, instructions, and action labels.',30,700,color=BLUE)
    b += text(75,237,'Illustrative base-frame displacement: Δx ∈ [−0.10, +0.10] m, with 256 bins.',27)
    values=[('Recorded action','+0.0200 m'),('Scale to [−1, 1]','+0.20'),('Quantize','bin 153'),('Token label','<action_153>')]
    for i,(h,d) in enumerate(values):
        x=75+i*370
        b += module(x,270,325,108,h,[d])
        if i<3:b += arrow(x+325,324,x+355,324)
    b += text(75,424,'Cross-entropy trains the VLM to predict the demonstrated action-token sequence.',28)
    b += line(75,459,1525,459)
    b += text(75,505,'Inference: generate tokens, then decode them back into physical commands.',30,700,color=BLUE)
    b += module(75,542,300,102,'Image + instruction',[],fill='white')
    b += module(445,542,300,102,'VLM action tokens',[],fill=PALE)
    b += module(815,542,300,102,'Action decoder',[],fill='white')
    b += module(1185,542,340,102,'Robot controller',[],fill='#e7f0ed')
    b += arrow(375,594,430,594)+arrow(745,594,800,594)+arrow(1115,594,1170,594)
    b += text(75,700,'Original OpenVLA: one 7D command = Δposition, Δorientation, and gripper control.',27)
    b += takeaway('Action tokens encode numerical commands whose meaning comes from the robot dataset.',
                   'OpenVLA uses 1st–99th percentile bounds. RT-2 also encodes an episode-termination command.')
    return slide('A VLA turns those representations into robot actions',b,
        'The top row is a deliberately authored quantization example, not an observed OpenVLA training value. Mapping +0.0200 m from [−0.10,+0.10] to [−1,1] gives +0.20. With 256 uniform bins, floor(256×0.60)=153, and decoding to that bin center gives approximately +0.01992 m. A token index is not a physical displacement. Original OpenVLA fine-tunes Prismatic-7B on 970k demonstrations, predicts one 7D end-effector/gripper action from one image and an instruction, and computes next-token cross-entropy on action tokens only. It uses 256 bins per coordinate between the training dataset’s first and 99th percentiles and repurposes 256 vocabulary entries. It does not have the chunking, proprioceptive input, or multiple-camera features of later derivatives. RT-2 co-fine-tunes on robot data and web vision-language tasks, uses an eight-value encoding including termination, and constrains output vocabulary for action prompts. Both generate discrete action codes autoregressively. A robot-specific decoder and controller still execute the resulting numeric command. Sources: OpenVLA §§3.1–3.3 and RT-2 §§3.2–3.3.',
        [OPENVLA,RT2], 'VLMs and robot actions')


def experience():
    b=text(75,190,'Data coverage and the action model are different design choices.',30)
    b+=table(75,235,1450,['Method','What transfers or scales','Action representation'],[
        ['RT-1','Many robot tasks and demonstrations','Discretized single actions'],
        ['RT-2 / OpenVLA','Pretrained vision-language representations','Autoregressive action tokens'],
        ['ACT','Task demonstrations, temporal structure','Continuous action chunks'],
        ['Diffusion Policy','Task demonstrations, action distributions','Denoised continuous chunks'],
    ],ratios=[.20,.47,.33],row_h=77,size=27)
    b+=text(75,674,'Web examples broaden visual and semantic coverage. Robot examples ground commands in motion.',27)
    b+=takeaway('Pretraining, supervision, action distribution, and control interface are separate axes.',
                 'These rows summarize the original methods. They are a design map, not a common benchmark ranking.')
    return slide('Which capability comes from which experience?',b,
        'Read across each row rather than reading downward as a leaderboard or a strict family tree. RT-1 scales language-conditioned robot imitation and uses a pretrained visual encoder and language embeddings, but it is not a conversational VLM adapted into a policy. Its final action dimensions are predicted non-autoregressively. RT-2 and OpenVLA adapt VLM representations to action prediction. RT-2 reports semantic reuse of motor skills supported by robot data, not arbitrary new motor skills learned solely from web text. ACT and Diffusion Policy are behavioral-cloning methods with expressive temporal/action representations; original ACT is task-specific and has no language conditioning. Their chunking and generative modeling choices can coexist with pretrained VLMs, as π0 will demonstrate. ACT’s CVAE is a conditional variational autoencoder; the present lesson focuses on chunk prediction and temporal ensembling rather than deriving its latent-variable objective. The table is an authored synthesis from RT-1 §5 and Appendix D.4, RT-2 §§3–5, OpenVLA §3, ACT §IV, and Diffusion Policy §3. Additional primary links: https://arxiv.org/html/2304.13705v1 and https://arxiv.org/html/2303.04137v5.',
        [RT1,RT2,OPENVLA], 'VLMs and robot actions')


def pi0_architecture():
    b=text(75,190,'π0 adds a smaller action expert with its own weights and continuous outputs.',30)
    b+=module(75,249,290,132,'Images + text',['multiple camera views','language ℓ'],fill='white')
    b+=arrow(365,315,425,315)
    b+=module(425,249,410,132,'PaliGemma backbone',['pretrained vision + language','≈3B parameters'])
    b+=arrow(835,315,915,315)
    b+=text(875,233,'attention',24,color=MUTED,anchor='middle')
    b+=module(915,249,400,262,'Action expert',['300M new parameters','predicts a vector field','over H = 50 action slots'],fill='#e7f0ed')
    b+=module(425,461,410,114,'Robot inputs',['qₜ, noisy action chunk Xτ, time τ'],fill='white')
    b+=arrow(835,477,915,477)
    b+=arrow(1315,380,1370,380)
    b+=lines(1445,354,['Update','Xτ'],30,gap=43,anchor='middle')
    b+=path('M 1445 429 L 1445 561 L 1115 561 L 1115 525',TEAL,2)
    b+=arrow(1115,525,1115,511,TEAL)
    b+=text(1445,607,'10 updates',26,700,color=TEAL,anchor='middle')
    b+=text(75,646,'Cache the visual/language prefix. Recompute the action expert as the noisy chunk changes.',28)
    b+=text(75,694,'After generation, decode the chunk into robot commands and execute it through feedback control.',27)
    b+=takeaway('The VLM supplies context while the action expert constructs a continuous command sequence.',
                 'An “action token” here is a continuous sequence slot. It is not a vocabulary bin.')
    return slide('π0 combines a VLM with a continuous action expert',b,
        'π0 initializes its visual-language weights from PaliGemma and adds a randomly initialized 300M-parameter action expert, giving roughly 3.3B parameters in total. The robot configuration q_t and action slots use robotics-specific expert weights. Image and text slots use the VLM weights, and attention provides the conditioning connections. This is a simplified computational diagram, not a separate large-language planner calling a black-box motion module. In the original π0, q_t is projected continuously. The original π0.5 changes proprioception to discretized text input, which will be distinguished later. The flow model predicts a vector field on an H=50 action chunk; ten Euler steps generate a chunk in the reported experiments. These ten iterations are generation computation, not ten physical servo cycles. During inference, prefix attention keys/values can be cached while the suffix is updated. Prefix representations are fixed within this sampling call, not permanently frozen model weights. π0 trains on diverse robot demonstrations and then uses curated downstream post-training; some long tasks use an additional high-level VLM. Sources: π0 v4 §IV, Figure 3, Appendices B/D, and CS224R lecture 17 p.6.',
        [PI0,CS_RL], 'Continuous VLA actions')


def flow_matching():
    b=text(75,190,'Training: draw a demonstrated chunk A, noise ε ∼ N(0, I), and a generation time τ.',28,700,color=BLUE)
    interpolation=xt()+mo('=')+par(mn(1)+mo('−')+mi('τ'))+mi('ε')+mo('+')+mi('τ')+mi('A')
    b+=mathml(75,218,760,70,interpolation,38)
    b+=text(875,262,'Known training target: A − ε',30)
    vf=vel()+par(xt()+mo(',')+mi('τ')+mo(',')+ot()+mo(',')+mi('ℓ'))
    loss=sub(mi('ℒ'),mt('FM'))+mo('=')+mi('𝔼')+norm2(vf+mo('−')+par(mi('A')+mo('−')+mi('ε')))
    b+=mathml(75,303,1450,87,loss,36)
    b+=text(75,417,'The loss fits a conditional vector field. The final action samples need not form a Gaussian.',26,color=MUTED)
    b+=line(75,448,1525,448)
    b+=text(75,494,'Inference: start with fresh noise and integrate the learned field.',29,700,color=BLUE)
    update=sub(mi('X'),mi('τ')+mo('+')+mi('δ'))+mo('=')+xt()+mo('+')+mi('δ')+vf
    b+=mathml(75,510,1450,80,update,36)
    for i,(x,l,d) in enumerate([(75,'X₀ = noise','τ = 0'),(520,'intermediate chunk','0 < τ < 1'),(1060,'X₁ = generated actions','τ = 1')]):
        w=365 if i<2 else 465
        b+=label_box(x,615,w,58,l,size=27)
        b+=text(x+w/2,710,d,25,color=MUTED,anchor='middle')
    b+=arrow(440,644,505,644)+arrow(885,644,1045,644)
    b+=takeaway('Generation time τ controls numerical sampling. Execution time t indexes robot commands.',
                 'Noise-to-action convention from π0 v4. Its reported sampler uses 10 Euler steps, δ = 0.1.')
    return slide('Flow matching: learn a path from noise to actions',b,
        'A is a demonstrated H×d_a action chunk and ε is an independent Gaussian tensor of the same shape. Sample τ from the training schedule, interpolate X_τ=(1−τ)ε+τA, and regress the network’s vector field against A−ε. The expectation is over demonstrated context/chunk pairs, independent noise, and generation time. Under squared error, the optimal field at a noisy point is the conditional mean of these local velocity targets; integrating this field transforms an entire noise distribution into an action distribution. This does not require the output action distribution to be Gaussian or to equal a single conditional mean action. Training can sample one τ per example rather than unrolling the full inference trajectory. At inference no demonstrated A is available: sample X_0 and repeatedly apply the learned field with observation/instruction fixed during the call. π0 v4 uses ten Euler steps and a training τ distribution emphasizing noisier times. The equations consistently use noise-to-action orientation. Older π0/π0.5 texts contain an inconsistent target sign, so the teaching equations follow the corrected π0 v4 §IV. The field has units of normalized action per unit generative time, not necessarily physical end-effector velocity. CS224R lecture 2 pp.11–14 gives the corresponding teaching derivation.',
        [PI0,CS_IL], 'Continuous VLA actions')


def room_plan(x,y,w,h,variant=0):
    """An annotated scene/layout diagram, explicitly an authored illustration."""
    b=rect(x,y,w,h,'white',LINE)
    if variant==0:
        b+=rect(x+25,y+30,w-50,42,PALE)+text(x+w/2,y+59,'counter',24,anchor='middle')
        b+=rect(x+w-145,y+103,115,75,'#e7f0ed')+text(x+w-88,y+150,'drawer',23,anchor='middle')
        b+=circle(x+90,y+125,20,PALE)+text(x+90,y+168,'cup',23,anchor='middle')
    else:
        b+=rect(x+25,y+30,110,135,PALE)+text(x+80,y+105,'counter',23,anchor='middle')
        b+=rect(x+190,y+30,w-220,60,'#e7f0ed')+text(x+190+(w-220)/2,y+70,'drawer',23,anchor='middle')
        b+=circle(x+210,y+114,20,PALE)+text(x+210,y+153,'cup',23,anchor='middle')
    return b


def transfer():
    b=text(75,190,'Task illustration: “Put the cup in the drawer.”',30)
    b+=text(75,252,'Training environments',29,700,color=BLUE)
    b+=room_plan(75,278,605,194,0)
    b+=text(75,527,'An unseen home',29,700,color=BLUE)
    b+=room_plan(75,550,605,164,1)
    b+=line(740,238,740,713)
    entries=[(258,'Recognize the scene','Which object is the cup?','Where is a plausible storage location?'),
             (421,'Choose the next subtask','Open the drawer or pick up the cup?','The current scene determines the next step.'),
             (584,'Execute the movement','Reach, grasp, and place under changed geometry.','Use physical feedback to correct the motion.')]
    for y,h,a,c in entries:
        b+=text(805,y,h,30,700,color=BLUE)
        b+=lines(805,y+47,[a,c],27,gap=39)
    b+=takeaway('π0.5 studies how several sources of supervision support transfer to new environments.',
                 'These floor plans are illustrations. The evaluation uses real held-out homes and separate mock rooms.')
    return slide('What must transfer to a new home?',b,
        'This schematic isolates semantic recognition, subtask choice, and motor execution. A new environment can change background appearance, object instances, layout, and local geometry simultaneously. The diagram is an authored cup-to-drawer illustration and is not a reconstruction of a training or evaluation home. π0.5’s mobile training subset contains about 400 hours across about 100 home environments, supplemented with other robots, laboratory tasks, high-level labels, web tasks, and later verbal instruction demonstrations. Its central claim is transfer of skills to new settings. The official explanation explicitly distinguishes that from learning wholly new physical skills without relevant robot experience. The later evaluation slide separates real-home task progress from controlled mock-environment ablations. Sources: original π0.5 Introduction, §IV-C, §V, and official project explanation.',
        [PI05,PI05_BLOG], 'π0.5 case study')


def supervision():
    b=text(75,190,'Different examples train motion, visual understanding, and semantic subtask selection.',29)
    b+=table(75,230,1450,['Data source','Example input','Supervised target'],[
        ['Mobile robots (MM)','Scene + robot state + instruction','Recorded action chunk'],
        ['Other homes/robots (ME)','Non-mobile robot demonstration','Recorded action chunk'],
        ['Lab robots + OXE','Cross-embodiment demonstration','Recorded action chunk'],
        ['High-level labels (HL)','Current scene + overall task','Subtask + relevant boxes'],
        ['Web data (WD)','Image + question or task prompt','Answer, caption, or boxes'],
        ['Verbal instruction (VI)','Scene during robot execution','Human-selected subtask'],
    ],ratios=[.29,.42,.29],row_h=63,size=26)
    b+=text(75,711,'VI enters post-training. The mobile subset is about 400 hours across about 100 home environments.',25,color=MUTED)
    b+=takeaway('Robot actions and semantic labels supervise different outputs within a shared model.',
                 'Source: π0.5 Figure 4 and §§IV-C/D. Dataset roles are distinct from undisclosed full mixture ratios.')
    return slide('π0.5 learns from several kinds of supervision',b,
        'Read each row as an input/target pair. MM is mobile manipulation in homes. ME collects demonstrations from easier-to-transport non-mobile robots in diverse homes. The laboratory cross-embodiment category includes other robot configurations and Open X-Embodiment. HL manually labels robot observations with semantic subtask descriptions and relevant object boxes. Web data supplies captioning, VQA, and localization supervision. VI is collected after an initial low-level policy exists: an expert guides it by selecting verbal subtask commands, producing demonstrations of high-level decisions. These are not additional teleoperated joint-action labels from speech alone. The mobile subset is roughly 400 hours in roughly 100 environments. The paper reports that 97.6% of first-phase examples come from sources other than the mobile household subset, but does not disclose a complete numeric mixing table. The boxes/targets here are schematic categories rather than copied training records. Sources: original π0.5 Figure 4 on PDF p.6 and §§IV-C/D.',
        [PI05], 'π0.5 case study')


def pi05_training():
    b=text(75,190,'Original paper: initialize from a web-pretrained VLM, then train in two stages.',29)
    b+=text(75,253,'1. Broad pretraining',32,700,color=BLUE)
    b+=text(865,253,'2. Focused post-training',32,700,color=BLUE)
    b+=module(75,280,605,110,'Mobile + other robots + lab/OXE',['high-level labels and web tasks also included'],fill='white')
    b+=module(865,280,660,110,'Mobile + other-home robot actions',['retain high-level/web data, add verbal instruction'],fill='white')
    b+=arrow(377,390,377,432)+arrow(1195,390,1195,432)
    b+=module(75,432,605,113,'Discrete output tokens',['text, boxes, compressed action chunks'])
    b+=module(865,432,660,113,'Discrete outputs + new action expert',['action/text tokens and continuous flow outputs'],fill='#e7f0ed')
    b+=arrow(680,488,850,488)
    b+=text(765,466,'transfer',24,color=MUTED,anchor='middle')
    b+=text(377,599,'Loss: cross-entropy',30,700,anchor='middle')
    b+=text(1195,599,'Loss: cross-entropy + flow matching',30,700,anchor='middle')
    b+=text(377,648,'280k training steps',26,color=MUTED,anchor='middle')
    b+=text(1195,648,'80k additional steps',26,color=MUTED,anchor='middle')
    b+=text(75,706,'The current OpenPI release supports the flow head; it does not reproduce this entire household system.',24,color=MUTED)
    b+=takeaway('The original recipe keeps discrete supervision while adding efficient continuous action generation.',
                 'Training stages and runtime decoding are separate. The next page shows the paper’s runtime hierarchy.')
    return slide('Separate the training recipe from runtime',b,
        'This diagram follows the original 19-page π0.5 paper rather than treating current OpenPI code as an exact reproduction. First, pretrain for 280k gradient steps with next-token prediction over text, boxes, and compressed FAST action tokens. This phase uses mobile manipulation, multi-environment non-mobile data, laboratory cross-embodiment data including OXE, high-level labels, and web data. Second, add a randomly initialized action expert and train another 80k steps with joint cross-entropy and flow matching, using flow-loss coefficient 10.0. Post-training action data use MM/ME successful episodes filtered by a length threshold. It retains web and relevant high-level data, adds verbal instruction examples, and omits laboratory cross-embodiment data. FAST discrete action outputs and flow action outputs are masked from attending to one another. FAST compresses a chunk; it is not identical to OpenVLA’s independent 256-bin coordinates. Original π0.5 discretizes proprioceptive state into text input, unlike π0’s continuous state projection. At inference, the system uses autoregressive subtask text followed by flow sampling. Current OpenPI README describes π0.5 trained with knowledge insulation and only supports the flow-matching head for π0.5 training/inference; do not assume downloading it reproduces this entire household hierarchy. Sources: π0.5 Figure 3 (p.4), Figure 4 (p.6), §§IV-A–D, and OpenPI README.',
        [PI05,OPENPI], 'π0.5 case study')


def hierarchy():
    b=text(75,190,'Runtime in the original paper: one model supports semantic and motor decisions.',29)
    left=mi('p')+par(at()+mo(',')+ct()+mo('|')+ot()+mo(',')+mi('ℓ'))
    right=mi('p')+par(ct()+mo('|')+ot()+mo(',')+mi('ℓ'))+mi('p')+par(at()+mo('|')+ot()+mo(',')+ct())
    b+=mathml(75,220,1450,85,left+mo('=')+right,38)
    b+=text(75,332,'Model factorization: the motor policy conditions on cₜ, the inferred subtask.',27,color=MUTED)
    b+=text(555,392,'Overall task ℓ',26,700,color=BLUE,anchor='middle')
    b+=arrow(555,403,555,435)
    b+=module(75,435,285,120,'Observation oₜ',['images + robot state'],fill='white')
    b+=module(420,435,285,120,'Subtask prediction',['“pick up the cup”'])
    b+=module(785,435,325,120,'Action expert',['50-step action chunk'],fill='#e7f0ed')
    b+=module(1190,435,335,120,'Robot + controller',['execute commands'],fill='white')
    b+=arrow(360,491,405,491)+arrow(705,491,770,491)+arrow(1110,491,1175,491)
    b+=text(745,464,'cₜ',26,700,color=BLUE,anchor='middle')
    b+=text(1150,464,'Aₜ',26,700,color=TEAL,anchor='middle')
    b+=path('M 330 555 L 330 608 L 947 608 L 947 569',BLUE,2)
    b+=arrow(947,569,947,555)
    b+=text(575,598,'current observation also conditions motion',24,color=MUTED,anchor='middle')
    b+=path('M 1357 555 L 1357 679 L 160 679 L 160 625',TEAL,2)
    b+=arrow(160,625,160,555,TEAL)
    b+=text(850,713,'new physical observations close the loop',25,color=TEAL,anchor='middle')
    b+=takeaway('Subtask text selects the behavior. The action expert supplies the numeric command chunk.',
                 'The 50-step chunk spans one second at 50 Hz. Model calls, text decoding, and servo feedback have different rates.')
    return slide('Decide the subtask, then generate the motion',b,
        'The equation is the paper’s model factorization, not the unconstrained probability chain rule: its action distribution depends on the inferred subtask c_t and current observation, but not directly on the overall task prompt ℓ. The same model supports autoregressive semantic output and continuous flow-based action output with different expert weights. In the actual system high-level inference uses four camera views, while low-level inference uses the wrist and forward cameras; o_t denotes available observation context rather than asserting identical camera subsets. Proprioception includes joints, grippers, lift and base velocity. The mobile platforms have 18 or 19 action dimensions. They issue arm/gripper/lift targets and base velocities at 50 Hz with action chunking; simple PD controllers track the targets. The paper reports no additional trajectory planner or collision detector in this execution path. Ten flow denoising steps generate the 50-step chunk. A 50 Hz command stream does not mean the entire VLM generates a semantic subtask 50 times per second. The official explanation describes a one-second chunk. Distinguish high-level demonstration labels used in training from inferred subtasks used at deployment. Sources: π0.5 §§IV-A/B/E and official explanation.',
        [PI05,PI05_BLOG], 'π0.5 case study')


def evidence():
    b1=text(75,190,'Held-out real homes: task progress, averaged over 10 trials per task/environment.',28,700,color=BLUE)
    b1+=image(75,205,925,445,'pi05-fig7.png','Original pi0.5 Figure 7b: task progress for mock environments and three real homes')
    b1+=text(1050,274,'3 homes',35,700,color=BLUE)
    b1+=lines(1050,320,['3 kitchens + 3 bedrooms','Each home is absent from training.'],25,gap=37)
    b1+=text(1050,421,'Progress ≠ full completion',28,700)
    b1+=lines(1050,467,['A rubric credits completed steps.','Half the dishes can earn partial credit.'],25,gap=37)
    b1+=text(1050,573,'Many trials last 2–5 minutes.',25,color=MUTED)
    b1+=text(1050,626,'Figure 7(b), original π0.5 paper.',23,color=MUTED)
    b2=text(75,190,'High-level inference ablation: distinguish training supervision from runtime hierarchy.',28,700,color=BLUE)
    b2+=image(75,205,925,445,'pi05-fig13.png','Original pi0.5 Figure 13: high-level training and inference ablations')
    b2+=text(1050,270,'Full π0.5',29,700,color=BLUE)
    b2+=lines(1050,311,['High-level labels in training','Explicit subtask inference'],25,gap=35)
    b2+=text(1050,414,'“implicit HL”',29,700,color=BLUE)
    b2+=lines(1050,455,['High-level labels in training','No explicit subtask at runtime'],25,gap=35)
    b2+=text(1050,545,'“no HL” removes both.',26,700)
    b2+=text(1050,590,'Full vs implicit: p = 0.144',24,color=MUTED)
    b2+=text(1050,626,'Figure 13, mock environments.',23,color=MUTED)
    b=steps('pi05-evidence',[b1,b2])
    b+=takeaway('The evidence supports environment transfer and the value of heterogeneous supervision.',
                 'Remaining failures include unfamiliar handles, occlusion, and repeated or distracted subtask choices.')
    return slide('What did the π0.5 experiments establish?',b,
        'Frame 1 reproduces Figure 7(b). §V-A explicitly says three held-out real homes with bedroom and kitchen tasks in each. Thus the caption’s three kitchens plus three bedrooms describe six room environments in three homes. The plot compares task progress, averaged over ten trials per task/environment, with a rubric roughly reflecting completed steps. It is not a universal task-completion success rate. Many quantitative trials last 2–5 minutes; longer 10–15-minute demonstrations in the introduction are separate evidence. Appendix B notes exclusions of some canceled episodes due to robot failures, time limits, or other causes, so do not infer a larger exact total trial count from the caption alone. Frame 2 reproduces Figure 13 in controlled mock environments. The full model uses semantic inference. The implicit-HL variant keeps high-level training examples but omits high-level inference at runtime. No-HL removes both. The strong implicit-HL result shows that training supervision accounts for a substantial part of the gain; the result should not be reduced to “explicit reasoning is essential.” The figure also contains no-web, no-verbal-instruction, GPT-4 high-level, and human high-level comparisons, which are defined in §V-E. These share the low-level inference process, and the main paper discusses the full model outperforming the chosen human high-level baseline. This does not establish a broad human-ability comparison. §VI describes failure on difficult unfamiliar handles, occlusion, repeated subtask choices, simple-prompt limitations, and restricted memory. Source: original π0.5 §§V-A/E, VI, and evaluation appendix.',
        [PI05], 'π0.5 case study')


def improvement():
    b=text(75,190,'Imitation fits demonstrated behavior. Reinforcement learning uses outcomes to improve a policy.',29)
    b+=module(75,260,300,125,'Demonstrations',['initial behavioral cloning'],fill='white')
    b+=module(445,260,300,125,'Initial VLA',['useful motor behaviors'])
    b+=module(815,260,300,125,'Robot rollouts',['outcomes + interventions'],fill='white')
    b+=module(1185,260,340,125,'Policy improvement',['reward/value-informed','update'],fill='#e7f0ed')
    b+=arrow(375,323,430,323)+arrow(745,323,800,323)+arrow(1115,323,1170,323)
    b+=path('M 1355 385 L 1355 435 L 595 435 L 595 398',TEAL,2)+arrow(595,398,595,385,TEAL)
    b+=text(978,468,'evaluate the updated policy on new rollouts',26,color=TEAL,anchor='middle')
    b+=line(75,503,1525,503)
    b+=text(75,551,'Three obstacles for large generative policies',30,700,color=BLUE)
    entries=[(75,'Physical data','Rollouts and resets cost time.'),
             (575,'Large models','Updates and iteration are expensive.'),
             (1075,'Action generators','Chunks and sampling complicate RL.')]
    for x,h,d in entries:
        b+=text(x,608,h,28,700)
        b+=text(x,655,d,25)
    b+=text(75,709,'Options include batch data collection, value-guided updates, and learned residuals or edits.',26,color=MUTED)
    b+=takeaway('A pretrained policy supplies a starting point. Improvement needs feedback and evaluation.',
                 'Intervention learning and reward-based learning can coexist. This diagram is a family of workflows, not one algorithm.')
    return slide('Demonstrations provide a starting point for improvement',b,
        'Behavioral cloning learns from demonstrated actions. Reinforcement learning optimizes expected cumulative reward and can exploit feedback about what happened after actions. A pretrained VLA supplies useful exploration and representations but generally does not supply a trained critic/value function. This diagram is an authored overview, not the exact algorithm of π0.5 and not a claim that π0.5 itself used this RL stage. CS224R lecture 17 distinguishes online updates from iterated offline learning over large rollout batches, describes value-guided supervised policy updates, and explores latent steering and residual/edit policies. These methods make different compromises; we do not teach them as interchangeable or guaranteed improvements. Human interventions can supply useful imitation data and coexist with RL. Large models incur substantial gradient-update costs, physical resets and rollouts can dominate collection time, and diffusion/flow action generators plus chunking complicate direct use of algorithms built around tractable per-step policy probabilities. Avoid transferring the lecture’s informal “80% plateau” into a universal reliability law or reading π0.5’s task-progress metric as success probability. Sources: CS224R lecture 17 pp.5–15 and 20–29, lecture 2 on online interventions.',
        [CS_RL,CS_IL], 'Policy improvement')


def get_slides():
    return [vlm(), discrete_vla(), experience(), pi0_architecture(), flow_matching(),
            transfer(), supervision(), pi05_training(), hierarchy(), evidence(), improvement()]
