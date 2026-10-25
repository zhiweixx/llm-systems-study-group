"""Twenty-four teaching steps: image answers, robot commands, chunks, and pi0.5.

All numeric teaching records are explicitly illustrative. The old literature
and visual-first modules remain intact. Source figures keep their attribution.
"""
from .common import *
from .policies import (PALIGEMMA, OPENVLA, RT2, PI0, PI05, PI05_BLOG, CS_IL,
                       mi, mn, mo, mt, sub, sup, par, norm2)

FAST = ('FAST §§V–VI', 'https://arxiv.org/html/2501.09747v1#S5')
GREEN = '#e5efec'
SAND = '#f4eadf'
ACTION = ['0.02', '0', '−0.01', '0', '0', '0.05', '1']
A = [['0.2','0.0'],['0.4','0.2'],['0.6','0.4']]
NOISE = [['−0.8','0.6'],['0.2','−0.6'],['0.8','0.0']]
BLEND = [['−0.3','0.3'],['0.3','−0.2'],['0.7','0.2']]
DIRECTION = [['1.0','−0.6'],['0.2','0.8'],['−0.2','0.4']]


def box(x,y,w,h,title,detail=(),fill=PALE,size=31):
    b=rect(x,y,w,h,fill)
    if isinstance(detail,str): detail=[detail]
    total=size+len(detail)*39
    yy=y+(h-total)/2+size*.8
    b+=text(x+w/2,yy,title,size,700,color=BLUE,anchor='middle')
    if detail:b+=lines(x+w/2,yy+43,detail,27,gap=39,anchor='middle')
    return b


def strip(x,y,values,cw=93,h=65,fill=PALE,size=28):
    return ''.join(rect(x+j*(cw+6),y,cw,h,fill)+text(x+j*(cw+6)+cw/2,y+h/2+size*.34,v,size,anchor='middle') for j,v in enumerate(values))


def numeric_matrix(x,y,values,cw=105,ch=68,fill=PALE,size=31):
    b=''
    for r,row in enumerate(values):
        for c,v in enumerate(row):
            xx,yy=x+c*cw,y+r*ch
            b+=rect(xx,yy,cw-4,ch-4,fill)+text(xx+(cw-4)/2,yy+ch*.66,v,size,anchor='middle')
    return b


def scene(x,y,w,h):
    return image(x,y,w,h,'pol-vlm-scene.png','Blue-cabinet mock kitchen from pi0.5 Figure 6')


def note(value,y=790,size=27,color=MUTED):
    return text(75,y,value,size,color=color)


def ce(target='y',context='I, Q'):
    return mo('−')+sub(mo('∑'),mi('j'))+mt('log ')+sub(mi('p'),mi('θ'))+par(sub(mi(target),mi('j'))+mo('|')+mt(context)+mo(',')+sub(mi(target),mo('<')+mi('j')))


def get_slides():
    out=[]
    def add(title,body,notes,sources=(),section='Learning from images and language'):
        out.append(slide(title,body,notes,sources,section))

    # 1: A fully specified input/output example, before architecture.
    b=scene(75,225,680,515)
    b+=text(825,250,'Question',29,700,color=BLUE)
    b+=lines(825,304,['“What color are','the cabinets?”'],35,gap=45)
    b+=arrow(1130,384,1130,455)+box(825,480,620,110,'Vision–language model (VLM)')
    b+=arrow(1130,590,1130,655)
    b+=text(825,710,'Answer',29,700,color=TEAL)
    b+=text(825,759,'“The cabinets are blue.”',34)
    b+=note('Illustrative question and answer, not a recorded model prediction.',size=24)
    add('A VLM turns an image and a question into an answer',b,
        'Start with the interface, not the model taxonomy. The photograph is the mock blue kitchen from original π0.5 Figure 6. The question and answer are authored teaching examples grounded in the visible cabinet color; no VLM was run for this slide. I denotes the image, Q the question, and y the answer tokens on later pages. PaliGemma is one representative generative VLM. Image-text training can also supervise captions or serialized boxes. The next slide opens the box.',[PALIGEMMA,PI05])

    # 2: Vectors, rather than vocabulary IDs, bridge the modalities.
    b=scene(75,235,300,230)
    b+=arrow(390,350,475,350)+box(475,285,300,130,'Image encoder',['patch features'])
    b+=arrow(775,350,860,350)+box(860,285,275,130,'Projection',['LM-sized vectors'])
    b+=arrow(1135,350,1215,350)+strip(1215,307,['v₁','v₂','…'],85,85,fill=GREEN,size=31)
    b+=text(1360,450,'Image prefix',28,anchor='middle')
    b+=text(75,565,'Question → text-token embeddings',31)
    b+=path('M 610 555 L 690 555 L 690 645',BLUE,2)+arrow(690,645,790,645)
    b+=path('M 1360 475 L 1360 645 L 1260 645',BLUE,2)+arrow(1260,645,1185,645)
    b+=box(790,575,395,145,'Language Transformer',['predict answer tokens'],fill=GREEN)
    b+=note('PaliGemma example: SigLIP image encoder + projection + Gemma language model.',size=27)
    add('Image patches become vectors the language model can read',b,
        'PaliGemma §3.1 and Figure 2: a SigLIP vision encoder produces a sequence of image features; a linear projection maps them to the Gemma embedding dimension. The decoder receives projected image vectors and text embeddings in one prefix. These visual vectors are continuous embeddings, not discrete words or quantized action IDs. Image and question tokens use full prefix attention; generated answer tokens use causal attention. At 224-pixel resolution PaliGemma has 256 visual tokens, but the diagram shows only three representative slots. The drawing is conceptual; it does not imply only three patches exist.',[PALIGEMMA])

    # 3: Training target and loss mask are concrete.
    b=text(75,205,'One training record: image I + question Q → answer y',31,700,color=BLUE)
    b+=scene(75,240,330,245)
    b+=lines(455,285,['Question Q','What color are the cabinets?'],30,gap=54)
    b+=text(455,435,'Target answer y',30,700,color=TEAL)
    b+=strip(455,472,['The','cabinets','are','blue','EOS'],cw=170,h=70,fill=GREEN,size=30)
    b+=rect(75,570,620,68,'#edf0f2','none')+text(385,614,'Image + question: no target loss',29,anchor='middle')
    b+=rect(720,570,805,68,GREEN,'none')+text(1122,614,'Answer positions: predict the next token',29,anchor='middle')
    b+=mathml(75,674,1450,100,ce(),41)
    b+=note('Use the correct preceding answer tokens during training.',size=27)
    add('VLM training predicts the answer, with the prompt loss masked',b,
        'This is an illustrative supervised record, not a claim that this photograph is in PaliGemma training. The objective sums negative log probabilities only over target answer positions, conditioned on image I, question Q, and preceding target tokens. Teacher forcing makes all target positions available in parallel during training while causal attention prevents looking at future answer tokens. Token labels are readable word-level stand-ins for subword tokens. EOS is end-of-sequence. PaliGemma §3.1 and §5.2 explain prefix attention and suffix-only supervision. A loss mask differs from an attention mask: the prefix contributes context even though it is not the prediction target.',[PALIGEMMA])

    # 4: Stay with the same scene, but expose the interface boundary.
    b=scene(75,235,640,485)
    b+=text(800,265,'At inference',32,700,color=BLUE)
    b+=strip(800,320,['The','cabinets','are','blue'],cw=170,h=75,fill=GREEN,size=28)
    b+=text(800,465,'This output describes the scene.',32)
    b+=line(800,520,1525,520)
    b+=text(800,587,'A controller needs numeric commands:',30,700,color=BLUE)
    b+=lines(800,649,['a coordinate frame, displacement,','rotation, and gripper setting.'],30,gap=48)
    b+=note('Generating a correct sentence does not supply a calibrated robot action.',size=28)
    add('A sentence answers the question; it does not move the robot',b,
        'At inference the answer suffix is not supplied: the VLM generates it autoregressively from the image, question, and its own earlier outputs. A semantic answer alone does not specify a robot command in the action interface defined in the preceding chapter. An application could translate language through separately engineered tools or skills; a VLA instead learns an action output from robot demonstrations. This page establishes the missing target supervision before introducing VLA architecture.',[PALIGEMMA,OPENVLA])

    # 5: Concrete VLA output in the chapter's declared action convention.
    b=scene(75,220,565,425)
    b+=text(720,240,'Instruction: “Close the cabinet.”',34,700,color=BLUE)
    b+=box(790,315,640,130,'Vision–language–action model',['predict a robot command'],fill=GREEN)
    b+=arrow(1110,445,1110,525)
    b+=strip(720,550,ACTION,cw=106,h=78,size=29)
    b+=text(884,690,'translation (m)',28,anchor='middle')
    b+=text(1220,690,'rotation (rad)',28,anchor='middle')
    b+=text(1450,690,'gripper',26,anchor='middle')
    b+=note('Toy single-arm command in the declared base frame.  g = 1 means open.',size=27)
    add('A VLA predicts an action in a defined robot interface',b,
        'The seven coordinates are [Δx, Δy, Δz, Δr_x, Δr_y, Δr_z, g]. Translation is expressed in the robot base frame in meters. The three rotation coordinates form an incremental rotation vector in radians, not additive Euler angles. The toy convention is g=1 open and g=0 closed. The example is [0.02,0,−0.01,0,0,0.05,1]; it illustrates an action interface and is not an inferred, calibrated, or demonstrated command for the pictured kitchen. The photograph shows a richer mobile platform; the seven-coordinate interface is an instructional single-arm example. Original OpenVLA predicts a seven-dimensional end-effector/gripper action; exact dataset rotation and gripper conventions must be respected rather than presumed universal.',[OPENVLA,PI05],'Learning robot actions')

    # 6: Preserve the liked action-token learning page, connect it to numbers.
    b=box(75,265,370,135,'Training input',['image I + instruction ℓ'])
    b+=arrow(445,332,520,332)+box(520,265,270,135,'Pretrained VLM',fill=GREEN)
    b+=arrow(790,332,855,332)+strip(855,300,['z₁','z₂','z₃','z₄','z₅','z₆','z₇'],cw=88,h=65,fill=GREEN)
    b+=text(1181,250,'Action-token targets',30,700,color=TEAL,anchor='middle')
    b+=arrow(1181,385,1181,455)+strip(855,475,ACTION,cw=88,h=65,size=26)
    b+=text(1181,589,'Decoded coordinate values',28,anchor='middle')
    b+=mathml(75,660,1450,100,ce('z','I, ℓ'),41)
    b+=note('Only the action-token targets contribute to this loss.',size=28)
    add('Train the VLM to predict the demonstrated action tokens',b,
        'OpenVLA §3.2 represents each action coordinate by one discrete token and applies next-token cross-entropy only to action-token positions. Here z_j denotes the token associated with coordinate j and ℓ is the instruction. The seven z labels stand for codebook entries; they are not literal strings of an actual tokenizer. The displayed numeric values are the same illustrative command as the previous page, rounded to readable precision. During dataset preparation a known demonstration action is encoded into target tokens. A dataset-specific decoder later maps these tokens to approximate coordinate values and restores units. The schematic does not pretend arbitrary numbers have universal token IDs. Original OpenVLA uses 256 bins per coordinate over the 1st–99th percentile interval, but the number-line construction is not needed to understand the learning objective.',[OPENVLA],'Learning robot actions')

    # 7: Now run the trained map without demonstration targets.
    b=box(75,275,365,140,'Current image',['+ instruction'])
    b+=arrow(440,345,515,345)+box(515,275,390,140,'Trained VLA',['generate action tokens'],fill=GREEN)
    b+=arrow(905,345,980,345)+box(980,275,545,140,'Action decoder',['tokens → seven numeric values'])
    b+=path('M 1252 415 L 1252 585 L 805 585',BLUE,2)+arrow(805,585,705,585)
    b+=box(75,515,630,140,'Robot controller',['track the commanded movement'])
    b+=path('M 280 515 L 280 462',TEAL,2)+arrow(280,462,280,415,TEAL)
    b+=text(800,710,'The next call sees the changed scene.',31,color=TEAL)
    b+=note('The demonstrated target is absent at inference. The model supplies the action.',size=28)
    add('Decode the tokens, execute the command, then observe again',b,
        'This closes the supervised-learning interface around a physical robot. The model predicts tokens without receiving the demonstration action. Decoding restores the action representation, normalization, and units expected by the robot controller. Execution changes the scene, so the policy must be called with new observations. Original OpenVLA predicts one action per call; later pages deliberately move from that single-step example to chunk generation. The controller and action adapter are essential parts of deployment, even when the policy is trained end to end.',[OPENVLA],'Learning robot actions')

    # 8: Compare concrete records, rather than model names.
    b=text(75,210,'The input can look similar. The supervision changes what the model learns to output.',31)
    b+=table(75,285,1450,['Training record','Input','Target'],[
        ['Image–text example','Image + cabinet-color question','“The cabinets are blue.”'],
        ['Robot demonstration','Image + “Close the cabinet.”','Seven action coordinates']
    ],ratios=[.28,.37,.35],row_h=130,size=29)
    b+=box(245,735,1110,68,'Robot experience supplies the action labels.',fill=GREEN,size=31)
    add('VLM and VLA training differ in their target records',b,
        'The two rows are deliberately authored examples over the same scene. Web image-text data can supply visual and semantic supervision. Robot demonstrations supply actions synchronized with observations and task instructions. A pretrained VLM transfers useful representations, but those representations do not replace the need for suitable action labels and an action interface. This table describes a distinction in record types rather than an exclusive partition: systems such as RT-2 and π0.5 co-train on both robot and non-robot records. Original OpenVLA trains the robot stage on demonstrations with an action-only loss.',[PALIGEMMA,OPENVLA,PI05],'Learning robot actions')

    # 9: The output object is now a numeric tensor.
    b=text(75,210,'Predict three future commands from the current observation.',33)
    b+=table(75,305,1450,['Time','Δx','Δy','Δz','','','','g'],[
        ['t','0.02','0','−0.01','0','0','0.05','1'],
        ['t + 1','0.01','0','−0.01','0','0','0.02','1'],
        ['t + 2','0','0','0','0','0','0','0']
    ],ratios=[.18]+[.82/7]*7,row_h=91,size=30)
    for j,axis in enumerate(['x','y','z']):
        xx=75+1450*.18+(3+j)*1450*.82/7
        b+=mathml(xx,323,130,55,mi('Δ')+sub(mi('r'),mi(axis)),30)
    b+=mathml(75,704,600,80,mi('A')+mo('∈')+sup(mi('ℝ'),mn(3)+mo('×')+mn(7)),42)
    b+=lines(740,726,['Rows: future commands. Columns: action coordinates.','General form: H × dₐ.'],28,gap=45)
    add('An action chunk is a matrix of future commands',b,
        'This three-by-seven matrix is a toy continuation of the single-command example, not a measured rollout. H is the number of future commands and d_a the action dimension. The storage convention throughout this chapter is H rows by d_a columns. The controller may execute only a prefix before the policy replans from new observations. A chunk describes physical command time; it does not specify how the neural network generates the matrix. That distinction motivates the next slide. Real π0/π0.5 chunks contain 50 commands; the original paper sometimes uses an inclusive time index with H=49, while this teaching deck consistently uses H to count commands.',[PI0,PI05],'Generating an action chunk')

    # 10: Architecture and generation choices are not conflated.
    b=box(75,375,335,145,'Context c',['scene + instruction','(+ robot state)'])
    b+=arrow(410,448,485,448)+box(485,335,440,225,'Transformer network',['shared architecture family'],fill=GREEN)
    b+=path('M 925 448 L 1000 448 L 1000 290',BLUE,2)+arrow(1000,290,1070,290)
    b+=box(1070,220,455,140,'Autoregressive output',['predict discrete tokens'])
    b+=path('M 1000 448 L 1000 620',BLUE,2)+arrow(1000,620,1070,620)
    b+=box(1070,550,455,140,'Flow output',['update a continuous chunk'])
    b+=note('Both routes can produce an action chunk. They use different targets and sampling procedures.',size=27)
    add('A Transformer can generate actions autoregressively or by flow',b,
        'Architecture, output representation, generation objective, and action chunking are distinct design choices. An autoregressive decoder predicts discrete action tokens with a causal output mask. A flow model can also use Transformer layers; π0’s continuous action block uses bidirectional attention among the action slots and predicts a vector field. In this diagram the box denotes an architectural family, not a claim that the two choices have identical weights or masks. Context can include robot state for models that use it. Chunking works with either route. Both can model multiple valid action choices, unlike a single deterministic mean-action predictor under squared error.',[PI0,FAST],'Generating an action chunk')

    # 11: Serial inference versus parallel teacher-forced training.
    b=text(75,210,'Encode the entire chunk as a sequence z₁, z₂, …, zₘ.',33)
    b+=strip(140,340,['context c','z₁','z₂','z₃','…','zₘ'],cw=200,h=90,size=33)
    for j in range(5):b+=arrow(240+j*206,310,446+j*206,310)
    b+=text(800,495,'At inference: each new token depends on the earlier generated tokens.',30,anchor='middle')
    b+=mathml(75,554,1450,95,ce('z','c'),42)
    b+=lines(75,720,['Training uses the known target prefix; positions can be computed in parallel.',
                      'One token per scalar: 50 commands × 14 coordinates = 700 action tokens.'],29,gap=51)
    add('Autoregressive chunk generation reuses next-token prediction',b,
        'c denotes the fixed conditioning context: current observation and instruction, plus state when used by the model. An action tokenizer encodes the full demonstrated matrix A into m tokens. During inference the model predicts tokens in sequence and then decodes them to the chunk. During teacher-forced training, causal masking still permits all target positions to be evaluated in parallel. The 700-token count is an illustrative 50×14 scalar-binning scheme, not a required length for every autoregressive VLA. FAST exists precisely because encoding every adjacent scalar separately can create long, highly redundant sequences. This comparison concerns serial dependencies, not an inference-time speed ratio.',[FAST,OPENVLA],'Generating an action chunk')

    # 12: Compression changes the target sequence; it is not a new controller.
    b=text(75,205,'FAST constructs a compact token sequence from the whole demonstrated chunk.',31)
    b+=numeric_matrix(95,315,A,cw=120,ch=80,size=33)
    b+=text(215,275,'Normalized chunk',29,700,color=BLUE,anchor='middle')
    b+=text(215,590,'Toy: 2 coordinates shown',23,color=MUTED,anchor='middle')
    b+=arrow(360,435,450,435)+box(450,345,325,180,'Frequency transform',['summarize time variation'])
    b+=arrow(775,435,860,435)+box(860,345,325,180,'Round + compress',['pack repeated patterns'])
    b+=arrow(1185,435,1260,435)+strip(1260,398,['z₁','…'],cw=115,h=75,fill=GREEN,size=31)
    b+=text(1380,585,'Fewer target tokens',27,anchor='middle')
    b+=arrow(1380,660,215,660,TEAL)
    b+=text(800,712,'Decode tokens → approximate chunk → restore physical units',31,color=TEAL,anchor='middle')
    b+=note('First scale each action coordinate to a comparable numeric range before encoding.',size=28)
    add('FAST compresses temporal redundancy before token prediction',b,
        'FAST §V and Figure 4 apply a discrete cosine transform independently along time for each normalized action coordinate, scale and round the coefficients, flatten them in frequency-first order, and use byte-pair encoding to pack frequent coefficient patterns. The transform itself is a change of basis; quantization introduces reconstruction error, while BPE compresses the quantized symbols losslessly. The schematic does not claim that this specific 3×2 matrix always becomes two tokens. Token count varies with the chunk, precision setting, and tokenizer. The VLA learns to predict the resulting tokens, and inverse decoding reconstructs an approximate action matrix. FAST §VI evaluates viable autoregressive policies; compressed token prediction is not intrinsically incapable of dexterous behavior.',[FAST],'Generating an action chunk')

    # 13: No noisy-action variable is used before being constructed.
    b=text(75,205,'Use a smaller toy chunk: three future commands, two normalized coordinates.',31)
    for x,title,values,fill in [(145,'Demonstrated chunk A',A,GREEN),(660,'Random noise ε',NOISE,SAND),(1170,'Noisy chunk at τ = 0.5',BLEND,PALE)]:
        b+=text(x+110,294,title,29,700,color=BLUE,anchor='middle')
        b+=numeric_matrix(x,335,values,cw=110,ch=80,fill=fill,size=33)
    b+=text(510,470,'+',42,700,anchor='middle')
    b+=arrow(965,455,1090,455)
    b+=text(1027,420,'÷ 2',29,700,color=BLUE,anchor='middle')
    b+=mathml(75,645,1450,105,sub(mi('X'),mi('τ'))+mo('=')+par(mn(1)+mo('−')+mi('τ'))+mi('ε')+mo('+')+mi('τ')+mi('A'),43)
    b+=text(800,735,'τ runs from 0 (noise) to 1 (demonstrated chunk).',27,anchor='middle')
    b+=note('Here τ = 0.5, so every entry is the average of the matching entries in A and ε.',size=28)
    add('A noisy chunk is just a numeric mixture of a chunk and noise',b,
        'All matrices are explicit toy values in normalized action space, with rows indexing future physical commands and columns indexing two displayed coordinates. Noise ε has the same shape as A; real training samples its entries from a standard Gaussian. The shown noise values are illustrative. For τ=0.5, Xτ=(A+ε)/2 gives exactly the displayed matrix. τ is generation time from zero to one; it is not physical robot time. No robot should execute Xτ during the generative procedure. A is available because this is construction of a training example. This removes the need to understand denoising terminology before seeing the arithmetic.',[PI0,CS_IL],'Learning a continuous chunk generator')

    # 14: Explain why flow learning is supervised, with the actual target.
    b=box(75,235,430,115,'Context c',['scene + instruction + robot state'])
    b+=text(75,410,'Training input: Xτ and τ',29,700,color=BLUE)
    b+=numeric_matrix(175,445,BLEND,cw=105,ch=67,size=30)
    b+=path('M 505 292 L 565 292 L 565 445',BLUE,2)+arrow(565,445,640,445)
    b+=arrow(400,545,565,545)+path('M 565 545 L 565 445',BLUE,2)
    b+=box(640,360,390,170,'Action Transformer',['predict vθ(Xτ, τ; c)'],fill=GREEN)
    b+=arrow(1030,445,1120,445)
    b+=text(1300,290,'Known target: A − ε',29,700,color=TEAL,anchor='middle')
    b+=numeric_matrix(1190,335,DIRECTION,cw=105,ch=67,fill=GREEN,size=30)
    loss=sub(mi('ℒ'),mt('flow'))+mo('=')+mi('𝔼')+norm2(sub(mi('v'),mi('θ'))+par(sub(mi('X'),mi('τ'))+mo(',')+mi('τ')+mo(';')+mi('c'))+mo('−')+par(mi('A')+mo('−')+mi('ε')))
    b+=mathml(75,650,1450,110,loss,39)
    b+=note('The demonstration A and sampled ε make the target direction known during training.',size=27)
    add('Flow training supervises a direction at the noisy input',b,
        'For the straight interpolation path Xτ=(1−τ)ε+τA, its derivative is A−ε. Subtracting the matrices on the previous page gives the exact target displayed here. The action expert predicts a vector of the same shape as the chunk. The squared loss compares predicted and known directions, averaged over demonstrations, noise, and sampled τ. No generated rollout or environment reward is needed to construct this supervised target. Statistically, the optimal regressor is the conditional expectation of velocity given noisy input and context, not a direct conditional mean of final actions. We use the noise-to-action sign in corrected π0 v4. The original π0.5 text has an inconsistent sign convention; this teaching derivation is internally consistent.',[PI0,CS_IL],'Learning a continuous chunk generator')

    # 15: Inference contains no demonstration target.
    b=text(75,205,'Keep the current context c fixed. The target demonstration A is now unavailable.',31)
    b+=numeric_matrix(95,335,NOISE,cw=100,ch=72,fill=SAND,size=30)
    b+=text(195,282,'Start: X₀ = ε',31,700,color=BLUE,anchor='middle')
    b+=arrow(320,440,435,440)+box(435,345,400,185,'Learned vector field',['read Xτ, τ, and context c'],fill=GREEN)
    b+=arrow(835,440,940,440)+box(940,345,285,185,'Update Xτ',['all entries together'])
    b+=path('M 1082 530 L 1082 600 L 635 600 L 635 555',TEAL,2)+arrow(635,555,635,530,TEAL)
    b+=arrow(1225,440,1315,440)+box(1315,345,210,185,'Final chunk',['Â = X₁'])
    update=sub(mi('X'),mi('τ')+mo('+')+mi('δ'))+mo('=')+sub(mi('X'),mi('τ'))+mo('+')+mi('δ')+sub(mi('v'),mi('θ'))+par(sub(mi('X'),mi('τ'))+mo(',')+mi('τ')+mo(';')+mi('c'))
    b+=mathml(75,668,1450,95,update,40)
    b+=note('Advance τ from 0 to 1.  δ is the update step size; π0 uses ten Euler steps.',size=28)
    add('Flow inference starts from noise and repeatedly updates the chunk',b,
        'Only the current context, sampled initial noise, and learned vector field are available at deployment. The network is not given A or A−ε. Starting at τ=0, a numerical integrator repeatedly evaluates the field and updates the entire action matrix until τ reaches one. π0 v4 §IV reports forward Euler with δ=0.1 for ten steps and caches the observation-prefix attention keys and values. Different samplers and step counts are possible. Intermediate matrices are internal generation states; the final matrix is decoded to the physical action convention and sent to the controller. This diagram is a computation graph, not a new trained policy or robot experiment.',[PI0],'Learning a continuous chunk generator')

    # 16: Explicitly separate the two time coordinates.
    b=text(75,205,'The generator refines a whole candidate chunk before the controller executes it.',31)
    for r,(tau,fill) in enumerate([('τ = 0',SAND),('τ = 0.5',PALE),('τ = 1',GREEN)]):
        yy=300+r*150
        b+=text(315,yy+53,tau,31,700,color=BLUE,anchor='end')
        for j in range(3):
            b+=box(390+j*245,yy,210,95,['noise','candidate','command'][r],fill=fill,size=27)
    b+=arrow(185,310,185,692)+text(75,750,'Generation time τ',29,color=BLUE)
    b+=arrow(410,740,1095,740,TEAL)+text(752,795,'Physical command time: t → t + 1 → t + 2',28,color=TEAL,anchor='middle')
    b+=lines(1190,400,['One generation step','updates all slots.'],28,gap=42)
    b+=lines(1190,600,['The controller executes','commands in order.'],28,gap=42)
    add('Generation time and robot time are different axes',b,
        'Each row represents one full action chunk at a different generation-time value τ. Each column is a future robot command time. Advancing τ changes every entry of the candidate chunk; it does not move the robot through those candidate values. Once generation finishes, the controller executes commands along the physical-time axis, usually only a prefix before observing and replanning. The τ=0.5 row is schematic and does not claim that inference follows the straight path to a particular known demonstration. This distinction applies to iterative diffusion and flow chunk generators.',[PI0,CS_IL],'Learning a continuous chunk generator')

    # 17: Summarize the choice without a false multimodality dichotomy.
    b=mathml(75,190,1450,95,mt('Both aim to model ')+mi('p')+par(mi('A')+mo('|')+mi('c')),43)
    b+=table(75,320,1450,['Choice','Autoregressive tokens','Continuous flow'],[
        ['Training target','Next action token','Local direction'],
        ['Inference','Tokens generated in sequence','Whole chunk updated repeatedly'],
        ['Representation','Tokenizer + numeric decoder','Continuous action coordinates'],
        ['Important cost','Token count and backbone size','Refinement count and expert size']
    ],ratios=[.22,.38,.40],row_h=86,size=28)
    b+=note('Both can represent multiple valid chunks. Actual speed and quality require measurement.',size=28)
    add('Compare the training and inference procedures',b,
        'These are alternative ways of representing a conditional distribution, not an expressive versus inexpressive distinction. An autoregressive categorical model can represent multimodal action sequences. A flow field can transform noise into a multimodal continuous distribution. The averaging limitation of direct squared-error action prediction does not apply to autoregressive likelihood training. FAST §VI-E reports comparable performance for FAST and flow on some evaluated tasks, with faster training for FAST in some settings and faster inference for the smaller flow expert in that implementation. Such results depend on tokenizer, backbone, model size, sampler, task, and hardware. Do not read a serial token count divided by a flow step count as a wall-clock speedup.',[FAST,PI0],'Learning a continuous chunk generator')

    # 18: Start pi0.5 with a household objective.
    b=text(75,205,'Task: “Put the dishes in the sink.”',35,700,color=BLUE)
    b+=image(75,285,1450,320,'pol-pi05-kitchen.png','Published pi0.5 household demonstration sequence, Figure 2')
    labels=['Choose a dish','Pick it up','Place it in the sink','Observe and repeat']
    for j,l in enumerate(labels):
        x=75+j*375
        b+=box(x,665,325,87,l,fill=GREEN,size=27)
        if j<3:b+=arrow(x+325,708,x+365,708)
    b+=note('Published household photographs; the short task trace is illustrative.',size=26)
    add('π0.5 combines household decisions with learned robot motion',b,
        'The photographs are from original π0.5 Figure 2; they depict a broader kitchen-cleaning demonstration. The four-box trace is our illustrative decomposition of the concrete dishes-in-sink task, not a frame-by-frame annotation or a measured rollout. This objective requires both choosing a useful next subtask and physically carrying it out in a new arrangement. The coming pages explain the records, training stages, and runtime that produce these two capabilities. The focus is the original research system, not a claim that a public checkpoint alone reproduces its full deployment.',[PI05],'π0.5 as a worked example')

    # 19: Actual input/target record roles with three concrete examples.
    b=text(75,205,'Illustrative records show what each data source teaches the model.',32)
    b+=text(75,275,'Record',29,700,color=BLUE)+text(405,275,'Input',29,700,color=BLUE)+text(1090,275,'Target',29,700,color=BLUE)
    for y in [300,465,630,795]:b+=line(75,y,1525,y)
    b+=lines(75,347,['Home / lab robots','action record'],28,gap=42,weight=700)+lines(405,347,['Images + state','“Put the cup in the sink.”'],28,gap=42)
    b+=lines(1090,347,['Normalized future','action chunk A'],28,gap=42)
    b+=text(75,530,'High-level subtask',29,700)+lines(405,512,['Images + state','“Put the dishes in the sink.”'],28,gap=42)
    b+=text(1090,550,'“Pick up the cup.”',30,color=TEAL)
    b+=text(75,695,'Web example',29,700)+lines(405,677,['Image + question','“What color are the cabinets?”'],28,gap=42)
    b+=text(1090,715,'“Blue.”',31,color=TEAL)
    add('Different records supervise actions, subtasks, and visual answers',b,
        'These are illustrative input/target records matching the original paper’s data roles, not reproductions of particular training entries. π0.5 §IV-C combines mobile household robot demonstrations, non-mobile robots in diverse homes, laboratory robots and Open X-Embodiment data, manually labeled high-level subtasks and boxes, and web visual tasks. The target mobile household subset is about 400 hours in about 100 home environments; it is not the entire dataset. Robot observations include images and proprioceptive state. Subtask labels condition low-level action training; high-level training predicts those labels from the overall task and current observation. Web records include captions, visual answers, and localization targets. The full numeric data mixing ratios are not disclosed.',[PI05],'π0.5 as a worked example')

    # 20: Discrete stage from the already established CE objective.
    b=text(75,205,'Stage 1: turn every target into a token sequence.',33,700,color=BLUE)
    for y,title,detail in [(310,'Robot record','context → FAST tokens'),(485,'Subtask record','context → text / boxes'),(660,'Web record','image + question → answer')]:
        b+=box(75,y,430,110,title,[detail],fill='white',size=29)
        b+=path(f'M 505 {y+55} L 605 {y+55} L 605 540',BLUE,2)
    b+=arrow(605,540,715,540)+box(715,420,480,240,'Web-pretrained VLM',['target-token cross-entropy','update the model weights'],fill=GREEN,size=34)
    b+=arrow(1195,540,1280,540)+text(1395,529,'Updated',32,700,color=BLUE,anchor='middle')+text(1395,574,'VLA',32,700,color=BLUE,anchor='middle')
    add('π0.5 first learns all record types with next-token prediction',b,
        'Original π0.5 §IV-C first trains a web-pretrained VLM autoregressively on discrete targets, including FAST-encoded action chunks. Textual records naturally supply text tokens; localization supplies serialized location tokens. The loss is target-token cross-entropy, not flow matching at this phase. This reuses the VLM learning setup rather than introducing a separate architecture for each supervision type. The paper reports 280k gradient steps. Data include home and laboratory robot actions, high-level labels, and web tasks. Control-mode markers distinguish joint versus end-effector pose targets; actions are normalized with dataset-specific quantiles and padded for differing embodiments. Those implementation details belong in the training records, not in an assumed universal action convention.',[PI05,FAST],'π0.5 as a worked example')

    # 21: Distinct post-training objectives without suggesting CE disappears.
    b=text(75,205,'Stage 2: keep token supervision and add the continuous action expert.',32)
    b+=box(75,305,420,200,'Pretrained VLM',['images + task + state'])
    b+=lines(542,350,['Context','features'],24,gap=29,color=BLUE,anchor='middle')
    b+=arrow(495,405,590,405)+box(590,305,470,200,'New action expert',['noisy chunk + τ','predict a flow direction'],fill=GREEN)
    b+=lines(1150,355,['Joint training','of both outputs'],32,gap=50,weight=700,color=BLUE)
    b+=text(285,565,'Token targets: text + FAST',28,color=TEAL,anchor='middle')
    b+=text(825,565,'Flow target: A − ε',28,color=TEAL,anchor='middle')
    b+=mathml(75,645,1450,95,mi('ℒ')+mo('=')+sub(mi('ℒ'),mt('token CE'))+mo('+')+mn(10)+sub(mi('ℒ'),mt('flow')),43)
    b+=note('The flow expert cannot attend to FAST action targets.',size=30,color=BLUE)
    add('Post-training adds flow while keeping token supervision',b,
        'Original π0.5 §IV-B/D jointly optimizes target-token cross-entropy, including FAST action tokens, and flow matching with weight 10 for 80k additional steps. The action expert starts from random weights; the model retains its text-prediction capability. The attention mask prevents discrete and continuous action representations from attending to each other, so the flow head cannot copy the FAST action target. Post-training uses successful, length-filtered mobile and other-home robot episodes, corresponding high-level labels, web records, and new verbal-instruction examples. In those examples an expert directs the learned low-level policy with subtask commands, supplying high-level choices. Laboratory cross-embodiment data do not remain in this phase. This is supervised post-training, not reinforcement learning.',[PI05],'π0.5 as a worked example')

    # 22: Full runtime with observations entering both decisions.
    b=text(75,205,'Overall instruction: “Put the dishes in the sink.”',33)
    b+=box(75,365,360,145,'Observation',['images + robot state'])
    b+=arrow(435,437,520,437)+box(520,365,435,145,'Predict subtask text',['“Pick up the cup.”'])
    b+=arrow(955,437,1040,437)+box(1040,365,485,145,'Generate action chunk',['condition on that subtask'],fill=GREEN)
    b+=path('M 255 365 L 255 280 L 1282 280',BLUE,2)+arrow(1282,280,1282,365)
    b+=text(800,263,'The action generator also sees the current observation.',27,color=BLUE,anchor='middle')
    b+=path('M 1282 510 L 1282 655 L 255 655 L 255 540',TEAL,2)+arrow(255,540,255,510,TEAL)
    b+=text(800,706,'Execute a command prefix, then observe the changed scene.',30,color=TEAL,anchor='middle')
    b+=note('Reuse the subtask across low-level calls; refresh it less frequently.',size=29)
    add('At runtime, predict a subtask and then generate its motion',b,
        'Original π0.5 §IV-A/B separates p(s|o,ℓ), a high-level subtask distribution, from p(A|o,s), the low-level action-chunk distribution. The low-level factor uses inferred subtask s instead of the overall instruction ℓ. Both are represented by the same model, used with different inference roles. High-level inference uses autoregressive text decoding; low-level inference uses ten flow integration steps. The model predicts 50-command chunks for the full 18–19 dimensional mobile platform, not the seven-coordinate teaching interface. Low-level actions use joint/gripper and torso targets plus base velocity. The diagram omits scheduling rates: choosing a new semantic subtask need not occur once per individual control tick.',[PI05],'π0.5 as a worked example')

    # 23: Only now assemble all named inputs into the full model.
    b=box(75,250,370,120,'Images',['encoder → patch vectors'])
    b+=arrow(445,310,535,310)
    b+=box(75,490,370,150,'State + task text',['state is discretized','into text tokens'])
    b+=arrow(445,565,535,565)
    b+=box(535,245,465,395,'PaliGemma backbone',['image / language context'],fill=GREEN,size=35)
    b+=arrow(1000,315,1090,315)+box(1090,245,435,140,'Text output',['next subtask tokens'])
    b+=arrow(1000,555,1090,555)+box(1090,470,435,170,'Action expert',['separate Transformer weights','output: chunk vector field'],size=31)
    b+=box(535,710,465,87,'Noisy chunk Xτ + time τ',fill=SAND,size=29)
    b+=path('M 1000 753 L 1307 753',BLUE,2)+arrow(1307,753,1307,640)
    b+=text(75,753,'One model; two output roles.',29,700,color=BLUE)
    add('The π0.5 architecture now has a role for every input',b,
        'Original π0.5 §IV-A and Appendix E use PaliGemma for image-language processing and a separate 300M-parameter action expert. Proprioceptive state is discretized and supplied as text tokens in π0.5, unlike the continuous state projection used by original π0. Image features and task/state text form contextual tokens. The action expert receives the candidate noisy action chunk and generation-time embedding and attends to that context. Within the continuous action block, tokens use bidirectional attention. Text generation remains causal. The two output roles are invoked with the conditioning appropriate to high-level or low-level inference; the diagram is not claiming that one unchanged prompt simultaneously produces both outputs on every call. Training also retains a FAST discrete-action output, as the preceding page showed.',[PI05],'π0.5 as a worked example')

    # 24: One real evidence figure with the metric and trial unit visible.
    b=image(75,195,1025,605,'pi05-fig7.png','Original pi0.5 Figure 7b average task progress by task and environment')
    b+=text(1160,280,'3 held-out homes',31,700,color=BLUE)
    b+=lines(1160,338,['3 kitchens','3 bedrooms'],29,gap=45)
    b+=text(1160,470,'Task progress',31,700,color=BLUE)
    b+=lines(1160,525,['10 trials per','task / environment'],29,gap=43)
    b+=lines(1160,655,['Partial credit differs','from full completion.'],27,gap=39,color=MUTED)
    b+=note('Next: where this learned policy sits inside a complete robot system.',y=812,size=25,color=BLUE)
    add('The paper measures task progress in unseen homes',b,
        'Original π0.5 §V-A and Figure 7 evaluate three held-out real homes, each with kitchen and bedroom tasks: three kitchens and three bedrooms, not six independent homes. Panel b reports average task progress over ten trials per task/environment. The plotted task families are items in drawer, dishes in sink, and laundry in basket; not every task is present in every home. Progress rubrics award partial completion, so these bars are not binary full-task success rates. Mock environments are also plotted. Quantitative episodes last approximately two to five minutes; longer published demonstrations do not establish an equivalent measured success rate. The figure supports transfer of learned household skills to these test environments. It does not establish unrestricted household autonomy or performance on arbitrary homes and tasks.',[PI05],'π0.5 as a worked example')

    assert len(out)==24
    return out
