"""Twenty-two visual-first slides on VLMs, VLAs, flow, and pi0.5.

The preceding compact module remains intact. Source photos are cropped from
primary PDFs; schematic models and the analytic flow demo remain editable.
"""
from math import exp, tanh
from statistics import NormalDist
from .common import *
from .policies import (PALIGEMMA, RT1, RT2, OPENVLA, PI0, PI05, PI05_BLOG,
                       OPENPI, CS_IL, CS_RL, mi, mn, mo, mt, row, sub, sup,
                       par, norm2, at, ot, ct, xt, vel)
from . import policies as compact


def block(x,y,w,h,label,detail=None,fill=PALE,size=32):
    b=rect(x,y,w,h,fill)
    b+=text(x+w/2,y+h/2+(3 if detail else size*.34),label,size,700,color=BLUE,anchor='middle')
    if detail:b+=text(x+w/2,y+h/2+44,detail,27,anchor='middle')
    return b


def caption(value,y=790,size=28,color=INK):
    return text(75,y,value,size,color=color)


def tokens(x,y,labels,w=100,h=65,fill=PALE,size=28):
    return ''.join(rect(x+i*(w+9),y,w,h,fill)+text(x+i*(w+9)+w/2,y+h/2+size*.34,s,size,anchor='middle') for i,s in enumerate(labels))


def observed_scene():
    b=image(75,230,670,510,'pol-vlm-scene.png','Mock kitchen photograph from pi0.5 Figure 6')
    b+=text(835,285,'Image + question',34,700,color=BLUE)
    b+=lines(835,348,['“Describe the kitchen.”'],34)
    b+=arrow(1090,397,1090,485)
    b+=block(835,510,540,120,'Vision–language model')
    b+=arrow(1090,630,1090,690)
    b+=text(1090,750,'Text or structured prediction',30,anchor='middle')
    b+=caption('Illustrative query. Photograph: π0.5 Figure 6.',size=24,color=MUTED)
    return slide('What does a vision–language model predict?',b,
        'A VLM accepts visual inputs and text prompts and produces predictions such as a description, an answer, or serialized boxes. The prompt shown here is an authored illustration, not a model run or a measured result. The photograph is a mock kitchen evaluation environment from π0.5 Figure 6. This page establishes the interface before showing the encoder and Transformer. PaliGemma is a representative generative VLM; it is not the only architecture called a VLM. Calibrated robot commands require an additional action representation and robot experience.',[PALIGEMMA,PI05],'Visual language to action')


def inside_vlm():
    b=block(75,240,270,105,'Image encoder','SigLIP')
    b+=arrow(345,292,420,292)+block(420,240,260,105,'Projection')
    b+=tokens(760,259,['I','I','I','…'],74,66,fill='#dbe8f0')
    b+=arrow(680,292,760,292)
    b+=text(914,225,'Image embeddings',27,anchor='middle')
    b+=block(75,505,605,105,'Text tokenizer + embeddings')
    b+=tokens(760,525,['Q','…'],74,66,fill='#e2ece8')
    b+=arrow(680,558,760,558)
    b+=text(841,491,'Question embeddings',27,anchor='middle')
    b+=path('M 1100 292 L 1160 292 L 1160 445',BLUE,2)
    b+=path('M 940 558 L 1160 558 L 1160 445',BLUE,2)
    b+=arrow(1160,445,1220,445)
    b+=block(1220,370,305,145,'Language model','Gemma decoder',fill='#e7f0ed')
    b+=caption('Projection aligns feature dimensions. It does not turn an image into vocabulary IDs.',y=760)
    return slide('Images become input tokens',b,compact.vlm()['notes'],[PALIGEMMA],'Visual language to action')


def quantization():
    b=text(75,200,'One illustrative base-frame displacement: Δx = +0.0200 m',32)
    x0,x1,y=150,1430,430
    b+=line(x0,y,x1,y,BLUE,3)
    for i in range(33):
        x=x0+(x1-x0)*i/32
        b+=line(x,y-15,x,y+15,LINE,1.5)
    for x,l in [(x0,'−0.10 m'),((x0+x1)/2,'0'),(x1,'+0.10 m')]:
        b+=text(x,493,l,29,anchor='middle')
    xp=x0+(x1-x0)*.60
    b+=arrow(xp,310,xp,408)+text(xp,286,'+0.0200 m',32,700,color=TEAL,anchor='middle')
    b+=text(790,564,'256 bins across the chosen range',30,anchor='middle')
    b+=tokens(430,640,['bin 153'],w=290,h=85,size=33)+arrow(735,682,860,682)
    b+=tokens(880,640,['action token'],w=400,h=85,size=33,fill='#e7f0ed')
    b+=caption('Original OpenVLA sets each coordinate’s bounds from the 1st and 99th percentiles.',size=26)
    return slide('A numerical action can become a token',b,
        compact.discrete_vla()['notes']+' The tick marks shown are a sparse schematic of the quantization interval, not all 256 bin boundaries. The dataset defines frame, units, bounds and gripper conventions. This slide isolates conversion of one coordinate before learning and deployment.',[OPENVLA,RT2],'Discrete action policies')


def discrete_training():
    b=text(75,205,'Training example',32,700,color=BLUE)
    b+=block(75,280,380,125,'Image + instruction')
    b+=arrow(455,343,555,343)+block(555,280,450,125,'Pretrained VLM',fill='#e7f0ed')
    b+=arrow(1005,343,1115,343)
    b+=tokens(1115,298,['153','…','g'],w=110,h=88,size=32)
    b+=text(1310,443,'Action-token targets',30,anchor='middle')
    b+=mathml(240,525,1120,100,mo('−')+sub(mo('∑'),mi('j'))+mt('log ')+sub(mi('p'),mi('θ'))+par(sub(mi('z'),mi('j'))+mo('|')+mi('I')+mo(',')+mi('ℓ')+mo(',')+sub(mi('z'),mo('<')+mi('j'))),44)
    b+=text(800,685,'I: image; ℓ: instruction; zⱼ: action token.',29,anchor='middle')
    b+=caption('Robot demonstrations supervise action-token prediction. This is behavioral cloning.',size=29)
    return slide('Train the VLM on demonstrated action tokens',b,
        'This equation is next-token cross-entropy on an action-token sequence z. I is the image and ℓ is the instruction; previous action-coordinate tokens condition later tokens within the same command. Original OpenVLA evaluates its cross-entropy loss on action tokens only and fine-tunes a Prismatic-7B VLM. The diagram’s token values are illustrative rather than an actual robot record; g denotes the gripper coordinate’s token. RT-2 additionally co-fine-tunes on original web vision-language examples. It constrains decoding to action tokens when prompted for actions, while retaining language outputs for language tasks. Do not attribute RT-2’s precise data mixture to OpenVLA or imply that original OpenVLA has chunking or proprioceptive input.',[OPENVLA,RT2],'Discrete action policies')


def discrete_inference():
    b=block(75,310,350,145,'Image + instruction')
    b+=arrow(425,382,510,382)+block(510,310,390,145,'VLA','predict action tokens',fill='#e7f0ed')
    b+=arrow(900,382,985,382)+block(985,310,540,145,'Decode + restore physical units')
    b+=path('M 1255 455 L 1255 625 L 690 625',BLUE,2)+arrow(690,625,620,625)
    b+=block(75,555,545,145,'Robot controller','execute the numeric command',fill='white')
    b+=path('M 345 555 L 345 490 L 250 490',TEAL,2)+arrow(250,490,250,455,TEAL)
    b+=text(75,755,'Observe again after executing the command.',28,color=TEAL)
    b+=text(860,755,'Original OpenVLA: one 7D action per call.',28)
    return slide('Deploy a vision–language–action (VLA) model',b,compact.discrete_vla()['notes'],[OPENVLA],'Discrete action policies')


def semantic_transfer():
    b=image(75,235,760,560,'pol-rt2-germany.png','RT-2 Figure 2 demonstration with banana and national flags')
    b+=lines(930,300,['“Move banana','to Germany”'],40,gap=58,weight=700,color=BLUE)
    b+=line(930,442,1525,442)
    b+=lines(930,508,['Web experience:','associate the country','with its flag.'],31,gap=43)
    b+=lines(930,667,['Robot experience:','move an object to a target.'],31,gap=43)
    return slide('Web knowledge can select a familiar motor skill',b,
        'This photograph is cropped from RT-2 Figure 2, which shows semantic generalization examples. The task uses a familiar physical manipulation skill with a target indicated through a semantic concept. The research distinction is between richer interpretation/selection and arbitrary new physical competence. RT-2 explicitly limits its motor skill distribution to robot experience even when web knowledge changes which object or destination it chooses. These particular illustrative demonstrations are not a common leaderboard against OpenVLA or later policies. RT-1 is a useful predecessor that scaled language-conditioned robot imitation; it was not a conversational VLM fine-tuned into a robot. Source: RT-2 Figure 2, §§3–5.',[RT2,RT1],'Visual language to action')


def pi0_model():
    b=block(75,250,330,125,'Images + instruction')
    b+=arrow(405,312,500,312)+block(500,250,450,125,'PaliGemma backbone','visual/language context')
    b+=arrow(950,312,1050,312)+block(1050,250,475,300,'Action expert','300M parameters',fill='#e7f0ed',size=38)
    b+=block(500,505,450,125,'Robot inputs','qₜ, noisy chunk Xτ, time τ',fill='white')
    b+=arrow(950,528,1050,528)
    b+=arrow(1287,550,1287,675)
    b+=text(1287,735,'Vector field for the action chunk',30,anchor='middle')
    b+=text(75,769,'π0 uses separate action-expert weights with attention to the VLM context.',29)
    return slide('π0 adds a continuous action expert',b,compact.pi0_architecture()['notes'],[PI0,CS_RL],'Continuous action policies')


def whole_chunk():
    b=text(75,205,'One generated sample contains H commands, each with dₐ coordinates.',32)
    x,y,cw,ch=260,290,160,110
    labels=[sub(mi('a'),mi('t')),sub(mi('a'),mi('t')+mo('+')+mn(1)),
            sub(mi('a'),mi('t')+mo('+')+mn(2)),mo('…'),
            sub(mi('a'),mi('t')+mo('+')+mi('H')+mo('−')+mn(1))]
    for j,l in enumerate(labels):
        b+=mathml(x+j*cw,y-75,cw-9,60,l,32)
        for i in range(3):
            b+=rect(x+j*cw,y+i*ch,cw-9,ch-9,('#edf3f7','#e5efec','#f4eadf')[i])
            if j==3:b+=text(x+j*cw+cw/2,y+i*ch+64,'…',32,anchor='middle')
    for i,l in enumerate(['coordinate 1','coordinate 2','…']):
        b+=text(218,y+i*ch+64,l,27,anchor='end')
    b+=arrow(280,690,1050,690)+text(665,742,'Physical execution time t',30,anchor='middle')
    b+=text(1190,410,'π0: H = 50',36,700,color=BLUE)
    b+=lines(1190,490,['All slots participate','in each generation step.'],27,gap=41)
    return slide('One sample is a whole action chunk',b,
        'The matrix schematically shows action-coordinate rows and physical-time columns. A_t is mathematically stored as H×d_a in this deck; the display transposes it for a horizontal time axis. Entries are intentionally blank, rather than fabricated robot measurements. Each column corresponds to one physical command. π0 uses H=50 and bidirectional attention across continuous action slots. Every flow-generation iteration updates the entire chunk; the action slots are not generated as 50 independent language-model conversations. Prediction horizon, executed prefix, physical control rate and the number of flow iterations are distinct. The next three slides explain generation time τ separately.',[PI0],'Continuous action policies')


def interpolation():
    b=text(75,200,'How can a demonstration provide a direction at an intermediate noisy point?',31)
    p0,p1=(270,545),(1290,320)
    b+=line(*p0,*p1,LINE,4)
    b+=circle(*p0,15,'#dbe8f0',BLUE)+circle(*p1,15,'#e2ece8',TEAL)
    p=(p0[0]+.42*(p1[0]-p0[0]),p0[1]+.42*(p1[1]-p0[1]))
    b+=circle(*p,17,PALE,BLUE)+arrow(p[0]+25,p[1]-5,p[0]+177,p[1]-39,TEAL)
    b+=text(p0[0],610,'ε: sampled noise',31,anchor='middle')
    b+=text(p1[0],275,'A: demonstrated chunk',31,anchor='middle')
    b+=text(p0[0],650,'τ = 0',26,color=MUTED,anchor='middle')
    b+=text(p1[0],374,'τ = 1',26,color=MUTED,anchor='middle')
    b+=text(p[0],p[1]+65,'Xτ',39,700,color=BLUE,anchor='middle')
    b+=text(p[0]+160,p[1]-80,'A − ε',35,700,color=TEAL,anchor='middle')
    b+=mathml(75,675,1450,105,xt()+mo('=')+par(mn(1)+mo('−')+mi('τ'))+mi('ε')+mo('+')+mi('τ')+mi('A'),46)
    return slide('Construct a flow-matching training example',b,
        'The line is a two-dimensional schematic of a path in the much higher-dimensional action-chunk space. Sample a demonstrated chunk A and an independent Gaussian ε with the same shape, then draw τ according to the training schedule. Interpolate X_τ=(1−τ)ε+τA. The derivative of this particular linear path is A−ε, which is known during training. We can therefore supervise a local velocity without simulating a robot or unrolling a generated trajectory. This is the noise-to-action convention in corrected π0 v4. The drawing’s position τ≈0.42 is illustrative. Its geometry is not an end-effector path in the room. CS224R lecture 2 pp.12–14 motivates the same construction.',[PI0,CS_IL],'Flow matching')


def field_loss():
    b=text(75,215,'Predict the local direction from the noisy chunk and the observation.',32)
    b+=block(75,295,380,125,'Xτ, τ, observation, ℓ',fill='white')
    b+=arrow(455,358,555,358)+block(555,295,430,125,'Action expert',fill='#e7f0ed')
    b+=arrow(985,358,1085,358)+text(1290,372,'Predicted vector field',32,anchor='middle')
    vf=vel()+par(xt()+mo(',')+mi('τ')+mo(',')+ot()+mo(',')+mi('ℓ'))
    loss=sub(mi('ℒ'),mt('FM'))+mo('=')+mi('𝔼')+norm2(vf+mo('−')+par(mi('A')+mo('−')+mi('ε')))
    b+=mathml(75,490,1450,135,loss,43)
    b+=lines(75,711,['The squared loss fits a vector field.', 'Integrating that field can generate a multimodal action distribution.'],30,gap=48)
    return slide('Fit the vector field, not a single mean action',b,
        compact.flow_matching()['notes']+' The statistical distinction is important: squared-error regression here estimates the conditional mean of local velocity targets at X_τ, τ, and context. It is not direct regression from the observation to one mean action. The expectation averages over data pairs, independent Gaussian noise and generation time. Expressiveness of the resulting sampled action law follows from transforming a noise distribution. It does not imply generative methods always beat robust direct regression in every task or dataset.',[PI0,CS_IL],'Flow matching')


def _flow_paths():
    """Analytic conditional flow for A∈{−2,+2}; no learned robot policy."""
    starts=[NormalDist().inv_cdf((i+.5)/18) for i in range(18)]
    paths=[]
    for x in starts:
        history=[x]
        for k in range(99):
            tau=k/100
            posterior_mean=2*tanh(2*tau*x/(1-tau)**2)
            x+=.01*(posterior_mean-x)/(1-tau)
            history.append(x)
        paths.append(history)
    return paths


def flow_demo():
    histories=_flow_paths()
    frames=[]
    for k in [0,25,50,75,99]:
        tau=k/100
        b=text(75,205,'Hold the observation and task fixed. Draw noise once, then update the sample.',31)
        b+=text(1370,267,f'τ = {tau:.2f}',35,700,color=BLUE,anchor='end')
        x0,x1,ymid,scale=245,1335,470,77
        b+=line(x0,260,x0,685,LINE,2)+line(x0,685,x1,685,LINE,2)
        for a in [-2,0,2]:
            yy=ymid-a*scale
            b+=line(x0,yy,x1,yy,'#dde3e7',1.5,'5 7')+text(x0-23,yy+10,str(a),28,anchor='end')
        b+=text(75,255,'Analytic 1D example with two action modes. No robot policy is running.',24,color=MUTED)
        b+=text(75,350,'Action',27)+text(75,387,'coordinate',27)
        for t in [0,.5,1]:b+=text(x0+t*(x1-x0),727,str(t),27,anchor='middle')
        b+=text(1450,727,'τ',29,anchor='middle')
        for j,h in enumerate(histories):
            color=BLUE if j<9 else TEAL
            pts=[(x0+(i/100)*(x1-x0),ymid-h[i]*scale) for i in range(k+1)]
            if k:
                b+=path('M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in pts),color,2)
            xx,yy=pts[-1]
            b+=circle(xx,yy,6,color,color)
        frames.append(b)
    b=steps('policy-flow-sampling',frames,y=737)
    return slide('Follow samples through the flow',b,
        'Use Next step to advance generation time while retaining the same initial noise samples. This is an analytic demonstration, not a trained model, robot trajectory, empirical benchmark, or visualization of π0 hidden states. Conditional on a fixed observation, let the data action A equal −2 or +2 with equal probability and ε~N(0,1). The interpolating law is an equal mixture of Gaussians with means ±2τ and standard deviation 1−τ. Its conditional flow is v(x,τ)=[2 tanh(2τx/(1−τ)^2)−x]/(1−τ). This follows from E[A|X_τ=x] and ε=(x−τA)/(1−τ). We integrate this known field using Euler δ=0.01 to τ=0.99. The paths illustrate how initial Gaussian variation separates into two modes. The action coordinate uses arbitrary units; horizontal τ is generation time and does not represent robot execution time. A learned policy approximates a conditional field from demonstrations, and a real action chunk has many coordinates. π0’s reported sampler uses ten Euler steps; the finer analytic demonstration is chosen for a clear teaching picture. The discrete endpoint law makes the field singular exactly at τ=1, so the display stops at 0.99. The control is a discrete step-through, not live numerical inference.',[PI0,CS_IL],'Flow matching')


def new_home():
    b=text(75,210,'π0.5: reuse learned skills in a home absent from training.',34)
    b+=image(75,305,1450,330,'pol-pi05-kitchen.png','pi0.5 Figure 2 sequence of cleaning tasks in a new kitchen')
    labels=[(260,'Close cabinets'),(620,'Put items away'),(1030,'Wipe a spill'),(1375,'Move dishes')]
    for x,l in labels:b+=text(x,690,l,28,anchor='middle')
    b+=caption('Published demonstration sequence. Quantitative evaluation comes later.',size=27,color=MUTED)
    return slide('A household task extends beyond one grasp',b,
        'The photograph strip is π0.5 Figure 2, cropped to the photographs while retaining their order. It shows a robot cleaning a previously unseen kitchen. The brief labels paraphrase tasks shown in the original caption. The research challenge combines object recognition, selecting useful subtasks, and executing motor behavior under changed geometry. This is demonstration evidence, not a measurement that every task succeeds in every home. The task-progress plot is presented separately after the training recipe. The official explanation distinguishes transferring learned skills to new settings from learning arbitrary new physical skills without relevant robot experience. Sources: original π0.5 Introduction, §V, Figure 2, and official project explanation.',[PI05,PI05_BLOG],'π0.5: heterogeneous supervision')


def robot_data():
    b=text(75,205,'Robot demonstrations supply actions across environments and embodiments.',32)
    b+=text(315,287,'Mobile home robots',31,700,color=BLUE,anchor='middle')
    b+=text(840,287,'Other home robots',31,700,color=BLUE,anchor='middle')
    b+=text(1335,287,'Laboratory robots',31,700,color=BLUE,anchor='middle')
    b+=image(75,325,480,400,'pol-pi05-mobile.png','Mobile manipulation examples from pi0.5 Figure 4')
    b+=image(600,330,480,390,'pol-pi05-other-homes.png','Non-mobile robots in diverse homes from pi0.5 Figure 4')
    b+=image(1165,325,340,400,'pol-pi05-lab.png','Laboratory robot examples from pi0.5 Figure 4')
    b+=caption('The mobile subset contains about 400 hours across about 100 home environments.',size=28)
    return slide('Robot data broaden the physical experience',b,
        'These are cropped examples from π0.5 Figure 4, not a newly collected dataset. MM supplies mobile household demonstrations, around 400 hours in about 100 home environments. ME supplies demonstrations from non-mobile robots in diverse homes; these robots are easier to transport but differ from the target mobile platforms. Laboratory cross-embodiment data include single/dual arms and static/mobile robots, with public Open X-Embodiment data as an additional component. The pictured lab examples are only part of this category. Action targets require each dataset’s control-mode, units, normalization and embodiment conventions. These physical sources complement semantic supervision rather than duplicating it. The full numeric mixing ratios are not disclosed; 97.6% of first-phase examples come from sources other than the target mobile household subset. Source: π0.5 §IV-C and Figure 4.',[PI05],'π0.5: heterogeneous supervision')


def semantic_data():
    b=text(75,205,'Some useful training targets are words and boxes rather than robot motion.',32)
    b+=image(75,300,430,300,'pol-pi05-hl.png','High-level bedroom example from pi0.5 Figure 4')
    b+=image(640,300,260,300,'pol-pi05-web.png','Web VQA pie image example from pi0.5 Figure 4')
    b+=image(1030,300,495,300,'pol-pi05-verbal.png','Human verbal coaching example from pi0.5 Figure 4')
    for x,h,ls in [(290,'High-level labels',['Scene → next subtask']),(770,'Web supervision',['Image → answer / boxes']),(1277,'Verbal instruction',['Human selects the next subtask'])]:
        b+=text(x,666,h,31,700,color=BLUE,anchor='middle')
        b+=lines(x,721,ls,27,gap=39,anchor='middle')
    b+=caption('Verbal instruction examples enter post-training.',size=26,color=MUTED)
    return slide('Semantic data teach what to do next',b,
        'The three images are crops from π0.5 Figure 4. The bedroom illustrates high-level supervision: manual subtask labels and relevant bounding boxes accompany robot observations. The pie illustrates a web visual-question-answering example; the broader web mixture also includes captioning and localization. The human beside the robot illustrates verbal instruction demonstrations collected after a low-level policy exists. An expert chooses useful next subtask commands during execution, yielding labels for high-level decisions. These are not newly inferred joint commands extracted from speech alone. VI enters post-training; high-level labels and web examples participate in both phases. The short input/output annotations paraphrase the data roles rather than claiming the displayed images constitute newly run model tests. Source: π0.5 §§IV-C/D and Figure 4.',[PI05],'π0.5: heterogeneous supervision')


def pretraining():
    b=text(75,205,'Stage 1: broad pretraining with discrete outputs',33,700,color=BLUE)
    b+=block(75,300,350,180,'Robot examples','home + laboratory',fill='white')
    b+=block(75,565,350,150,'Semantic examples','high-level + web',fill='white')
    b+=path('M 425 390 L 520 390 L 520 520',BLUE,2)+path('M 425 640 L 520 640 L 520 520',BLUE,2)
    b+=arrow(520,520,610,520)+block(610,405,450,230,'Web-pretrained VLM','280k training steps',fill='#e7f0ed')
    b+=arrow(1060,520,1150,520)
    b+=lines(1340,436,['Text','Boxes','FAST tokens'],32,gap=70,weight=700,color=BLUE,anchor='middle')
    b+=text(1340,620,'compressed action chunks',26,color=MUTED,anchor='middle')
    b+=caption('Objective: next-token cross-entropy, including tokenized action chunks.',size=30)
    return slide('π0.5 first learns with discrete tokens',b,compact.pi05_training()['notes'],[PI05],'π0.5: training')


def posttraining():
    b=text(75,205,'Stage 2: retain discrete supervision and add a continuous action expert.',32)
    b+=block(75,310,445,170,'Pretrained VLA','text + FAST outputs')
    b+=arrow(520,395,615,395)+block(615,310,445,170,'New action expert','continuous flow outputs',fill='#e7f0ed')
    b+=text(1270,355,'80k',48,700,color=BLUE,anchor='middle')
    b+=text(1270,410,'additional steps',30,anchor='middle')
    b+=mathml(75,520,1450,100,mi('ℒ')+mo('=')+sub(mi('ℒ'),mt('cross-entropy'))+mo('+')+mn(10)+sub(mi('ℒ'),mt('flow')),45)
    b+=lines(75,700,['Keep mobile / other-home actions, high-level labels, and web data.',
                     'Add verbal instruction. Omit laboratory cross-embodiment data.'],29,gap=49)
    return slide('Then add efficient continuous action generation',b,compact.pi05_training()['notes'],[PI05],'π0.5: training')


def hierarchy_visual():
    b=text(75,205,'Overall task: “Put the dishes away.”',34)
    b+=block(75,330,400,165,'Current observation','images + robot state',fill='white')
    b+=arrow(475,413,565,413)+block(565,330,430,165,'Semantic prediction','“pick up the cup”')
    b+=arrow(995,413,1085,413)+block(1085,330,440,165,'Action expert','50-step action chunk',fill='#e7f0ed')
    b+=path('M 275 330 L 275 275 L 1305 275',BLUE,2)+arrow(1305,275,1305,330)
    b+=text(800,258,'Current images + robot state',26,color=BLUE,anchor='middle')
    b+=path('M 1305 495 L 1305 650 L 275 650 L 275 525',TEAL,2)+arrow(275,525,275,495,TEAL)
    b+=text(800,697,'Execute commands, then observe the changed scene.',30,color=TEAL,anchor='middle')
    b+=caption('The same model supports semantic prediction and continuous action generation.',size=29)
    return slide('Choose a subtask, then generate its motion',b,compact.hierarchy()['notes'],[PI05,PI05_BLOG],'π0.5: inference')


def hierarchy_equation():
    b=text(75,205,'Let cₜ denote a semantic subtask and Aₜ a chunk of robot commands.',32)
    high=mi('p')+par(ct()+mo('|')+ot()+mo(',')+mi('ℓ'))
    low=mi('p')+par(at()+mo('|')+ot()+mo(',')+ct())
    lhs=mi('p')+par(at()+mo(',')+ct()+mo('|')+ot()+mo(',')+mi('ℓ'))
    b+=mathml(75,300,1450,120,lhs+mo('=')+f'<mstyle mathcolor="{BLUE}">'+high+'</mstyle>'+f'<mstyle mathcolor="{TEAL}">'+low+'</mstyle>',48)
    b+=text(430,560,'Semantic decision',34,700,color=BLUE,anchor='middle')
    b+=text(430,625,'Overall task + current scene',30,anchor='middle')
    b+=line(800,520,800,680)
    b+=text(1160,560,'Motor decision',34,700,color=TEAL,anchor='middle')
    b+=text(1160,625,'Inferred subtask + current scene',30,anchor='middle')
    b+=caption('This is the model’s factorization: the low-level factor receives cₜ instead of the overall task ℓ.',y=805,size=25,color=MUTED)
    return slide('The hierarchy separates two conditional decisions',b,compact.hierarchy()['notes'],[PI05],'π0.5: inference')


def home_evidence():
    b=image(75,190,1050,610,'pi05-fig7.png','pi0.5 Figure 7b mean task progress by real home and task, with mock environments')
    b+=text(1160,305,'3 held-out homes',32,700,color=BLUE)
    b+=lines(1160,362,['3 kitchens','3 bedrooms'],30,gap=48)
    b+=text(1160,510,'Metric: task progress',28,700)
    b+=lines(1160,566,['10 trials per','task / environment'],28,gap=43)
    b+=lines(1160,709,['Partial credit differs','from full completion.'],27,gap=41,color=MUTED)
    return slide('How much of each task did the robot complete?',b,
        'This reproduces original π0.5 Figure 7(b). §V-A specifies three held-out real homes, with kitchen and bedroom tasks in each. Thus the caption’s three kitchens plus three bedrooms refer to six room environments within three homes. The y-axis is average task progress, not binary episode success. The rubric roughly credits completed task steps; moving half the dishes can receive partial credit. The caption reports ten trials per task/environment. Appendix B describes canceled-episode exclusions, so do not derive a larger exact total from the bar count. Many quantitative trials last 2–5 minutes; longer 10–15-minute demonstrations are separate evidence. This is an environment-generalization result, not a measurement of arbitrary unseen motor skills. Source: π0.5 §V-A, Figure 7 and evaluation appendix.',[PI05],'π0.5: evidence')


def hierarchy_evidence():
    b=image(75,185,1035,625,'pi05-fig13.png','pi0.5 Figure 13 high-level training and inference ablation')
    b+=text(1150,295,'“implicit HL”',33,700,color=BLUE)
    b+=lines(1150,355,['High-level training labels.','No explicit subtask','at runtime.'],27,gap=42)
    b+=lines(1150,565,['Full vs implicit HL','p = 0.144'],31,gap=50)
    b+=lines(1150,740,['Mock environments.','Metric: task progress.'],26,gap=39,color=MUTED)
    return slide('Training labels help even without an explicit hierarchy',b,
        'This reproduces original π0.5 Figure 13 in controlled mock environments. Full π0.5 uses high-level labels and explicit subtask inference. The implicit-HL variant keeps the high-level training examples but omits that inference at runtime; no-HL removes both. The full-versus-implicit comparison is labeled p=0.144, so the observed mean gap should not be presented as a statistically established advantage at conventional thresholds. High-level supervision remains useful even without explicit high-level inference. The entire plot retains no-web, no-verbal-instruction, GPT-4 high-level, and human high-level ablations, with uncertainty and reported p-values. Their definitions are in §V-E. The comparison uses the full low-level inference process with high-level variants. The full model outperforming this particular human high-level baseline does not establish general human-level ability. Metric: task progress. Source: π0.5 §V-E and Figure 13.',[PI05],'π0.5: evidence')


def scope_limits():
    b=image(75,235,850,510,'pol-pi05-real-rooms.png','Real kitchen and bedroom evaluation environments from pi0.5 Figure 6')
    b+=text(995,285,'What transfers?',33,700,color=BLUE)
    b+=lines(995,344,['Learned skills across','new layouts and objects.'],31,gap=44)
    b+=text(995,500,'What still fails?',33,700,color=BLUE)
    b+=lines(995,559,['Unfamiliar handles.','Occluded objects.','Repeated subtask choices.'],30,gap=47)
    return slide('Generalization still has a physical scope',b,
        'The photographs show real evaluation rooms from π0.5 Figure 6; they are not labeled failure episodes. The failure classes come from the paper’s discussion: difficult or unfamiliar handles, partial observability such as occluded spills, and distracted high-level choices such as repeated drawer opening and closing. Longer-context memory and complex user preferences remain limitations. The paper and official explanation concern transferring learned motor skills into unfamiliar environments, not arbitrary new physical skills. The current OpenPI π0.5 release is trained with knowledge insulation and supports only the flow-matching head for training and inference. It should not be equated with downloading a complete reproduction of the paper’s hierarchical mobile household system.',[PI05,OPENPI],'π0.5: scope')


def rl_improvement():
    b=text(75,205,'Demonstrations provide an initial policy. Outcomes can guide further learning.',32)
    b+=block(75,330,380,165,'Pretrained policy','learned motor behavior')
    b+=arrow(455,413,560,413)+block(560,330,430,165,'Robot experience','rollouts + interventions',fill='white')
    b+=arrow(990,413,1095,413)+block(1095,330,430,165,'Policy update','reward / value feedback',fill='#e7f0ed')
    b+=path('M 1310 495 L 1310 650 L 265 650 L 265 525',TEAL,2)+arrow(265,525,265,495,TEAL)
    b+=text(800,709,'Evaluate the updated policy in new rollouts.',31,color=TEAL,anchor='middle')
    b+=caption('Physical data cost, large updates, and generative action interfaces complicate RL.',size=28)
    return slide('Experience can improve an imitation-trained policy',b,compact.improvement()['notes'],[CS_RL,CS_IL],'Beyond behavioral cloning')


def get_slides():
    slides=[observed_scene(),inside_vlm(),quantization(),discrete_training(),discrete_inference(),
            semantic_transfer(),pi0_model(),whole_chunk(),interpolation(),field_loss(),flow_demo(),
            new_home(),robot_data(),semantic_data(),pretraining(),posttraining(),hierarchy_visual(),
            hierarchy_equation(),home_evidence(),hierarchy_evidence(),scope_limits(),rl_improvement()]
    assert len(slides)==22
    return slides
