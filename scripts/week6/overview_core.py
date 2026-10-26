"""Figure-led foundations and synthesis for the 30-slide robot learning overview."""
from .common import *
from .common_visual import robot_demo_body
from .policies import PALIGEMMA, RT2, OPENVLA, PI0, PI05, PI05_BLOG, CS_IL
from .teaching_foundations import get_slides as foundations

CLIP=('CLIP', 'https://arxiv.org/abs/2103.00020')
MIT=('MIT: pick and place','https://manipulation.mit.edu/pick.html')
ACT=('ACT / ALOHA','https://arxiv.org/abs/2304.13705')
DATA=('ALOHA dataset','https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human')
CAP=('Code as Policies','https://code-as-policies.github.io/')
UNITAS=('UNITAS §3','https://arxiv.org/html/2610.12099v1#S3')
RAX=('RAX','https://arxiv.org/abs/2610.22496')
RATS=('RATs','https://playful-rats.github.io/')
RSI=('Wang: 123 rounds','https://arxiv.org/abs/2609.31760')
GREEN='#e5efec'
SAND='#f4eadf'

def box(x,y,w,h,heading,detail=(),fill=PALE,color=BLUE,size=30):
    if isinstance(detail,str): detail=[detail]
    out=rect(x,y,w,h,fill)
    yy=y+(h-(size+len(detail)*40))/2+size*.8
    out+=text(x+w/2,yy,heading,size,700,color,anchor='middle')
    if detail: out+=lines(x+w/2,yy+44,detail,27,gap=40,anchor='middle')
    return out

def foot(value,y=795,size=27):
    return text(75,y,value,size,color=MUTED)

def photo(x,y,w,h,name,alt=''):
    return image(x,y,w,h,name,alt)

def core_slides():
    out=[]
    def add(title,body,notes,sources=(),section='Foundations'):
        out.append(slide(title,body,notes,sources,section))

    # 1. Simple title + robot photograph, as in the supplied reading-group decks.
    b=lines(75,245,['From visual understanding','to robot behavior'],56,gap=75,weight=700,color=BLUE)
    b+=lines(75,452,['VLM · VLA · world-action models','Code as Policies · self-improvement'],33,gap=56)
    b+=photo(900,200,625,450,'mit-pick.png','Robot pick-and-place scene from MIT Robotic Manipulation')
    b+=line(75,685,1525,685)+text(75,741,'A tutorial for researchers familiar with LLMs',30)
    b+=foot('LLM Systems Study Group  ·  26 October 2026',size=24)
    add('Robot learning and code-based agents',b,
        'This 30-slide tutorial starts with observable interfaces, then introduces neural action policies, world-action models, and program-based control. Recent RAX, RATs, COAP and the 123-round RSI report are presented as different systems with different evaluation protocols. They are not a single leaderboard. Teaching diagrams are original. The robot image is from Russ Tedrake’s MIT Robotic Manipulation materials. The layout follows the supplied reading-group references: white background, clear title rule, one large diagram or evidence item per page, and short annotations. The existing 52-slide Week 6 deck is retained separately.',[MIT], 'Introduction')

    # 2. Return to the same map at the end: outputs, rather than an exclusive taxonomy.
    b=text(75,212,'Start with what the system produces.',35)
    rows=[('VLM','Image + text','Meaning, answers, visual features',BLUE),
          ('VLA','Image + instruction + robot state','Robot action or action chunk',TEAL),
          ('WAM','Observation + task / action context','Actions and a model of future changes',WARM),
          ('Code as Policy','Instruction + API + observations','An executable control program',BLUE)]
    for i,(name,inp,output,c) in enumerate(rows):
        yy=265+i*116
        b+=text(75,yy+43,name,32,700,c)+text(375,yy+22,inp,27,color=MUTED)
        b+=arrow(375,yy+55,475,yy+55,c)+text(505,yy+65,output,31)
        b+=line(75,yy+96,1525,yy+96)
    b+=foot('These interfaces can be combined in one robot system.',size=29)
    add('Four interfaces to keep separate',b,
        'These are useful interfaces rather than mutually exclusive categories. VLM is broad: image–text alignment models and generative multimodal models both qualify. VLA specifies a learned vision/language-conditioned action policy. WAM is an evolving family that couples action learning with a representation of future world evolution; models differ in whether future prediction is required at inference. Code as Policy describes an executable program as the policy representation and may call a VLM, VLA, planner, or classical controller. RSI describes a process for modifying a system, not a fifth action representation.',[RT2,UNITAS,CAP], 'Introduction')

    # 3. Use the existing verified dataset video, not a new claimed rollout.
    add('Watch one observation–action loop',robot_demo_body(),
        'This is a recorded ALOHA simulated insertion demonstration from the LeRobot dataset. It is not an autonomous rollout of one of the models discussed later. Two robot arms must align a peg and socket. The stored episode has 500 records at nominal 50 Hz; the embedded playback is sampled at 10 Hz. A high-level policy selects desired targets, a lower-level controller tracks them, and new images and robot measurements close the loop. A target is different from the realized next state. Watch for the need to observe alignment before insertion.',[DATA,ACT])

    # 4. Retain the complete, verified action record with its coordinate meanings.
    s=foundations()[4].copy()
    s['title']='What does a robot action actually contain?'
    out.append(s)

    # 5. The shape of training data and the source of supervision.
    b=text(75,208,'A demonstration is an ordered episode, not a list of independent pictures.',30)
    b+=photo(75,275,460,345,'aloha-frame100.png','Frame 100 of the ALOHA simulated insertion episode')
    b+=text(75,663,'An observed scene at time t',29,color=BLUE)
    b+=code(610,260,915,315,'episode_id: 0\nframe_index: 100\nimage: uint8[480, 640, 3]\nstate: float32[14]   # joints + grippers\naction: float32[14]  # desired targets',27)
    b+=text(610,635,'Training input',28,700,BLUE)+text(610,678,'image + state (+ task instruction)',29)
    b+=arrow(610,710,745,710)+text(775,721,'next H recorded actions',30,700,TEAL)
    b+=foot('Behavior cloning learns this mapping from demonstrations; H is the prediction horizon.',size=27)
    add('Robot learning starts with aligned observations and actions',b,
        'The field shapes are from the actual ALOHA simulation dataset; the code block is a readable schema, not a literal file serialization. Each action is a desired joint/gripper configuration and state records the measured configuration. A training window pairs the current observation with H consecutive command vectors. A language-conditioned model may also receive the task instruction, but original ACT is not language conditioned. Demonstrations can be recorded from teleoperation, where a person moves leader arms to supply follower-arm commands. Behavior cloning optimizes an action prediction objective on these aligned records. At deployment the recorded target is unavailable. Split complete episodes and scenes for evaluation; splitting adjacent frames can leak nearly identical observations.',[DATA,ACT,CS_IL])

    # 6. Include the discriminative and generative families visible in reference deck 2.
    b=text(75,210,'Vision–language models connect visual observations to language.',32)
    b+=text(75,290,'Match image and text',34,700,BLUE)
    b+=photo(75,325,365,255,'pol-vlm-scene.png','Kitchen from pi0.5')
    b+=box(500,330,235,95,'Image encoder',size=26)+box(500,490,235,95,'Text encoder',size=26)
    b+=arrow(445,440,490,380)+text(75,642,'“a blue kitchen”',29)
    b+=path('M 350 632 L 460 632 L 460 538',BLUE,2)+arrow(460,538,490,538)
    b+=path('M 735 377 L 760 377 L 760 625 L 730 625',TEAL,2)+arrow(730,625,720,625,TEAL)
    b+=arrow(617,585,617,615,TEAL)
    b+=text(525,650,'similarity score',28,color=TEAL)+text(75,721,'CLIP: paired items should align.',29)
    b+=line(790,275,790,740)
    b+=text(855,290,'Generate an answer',34,700,BLUE)
    b+=box(855,350,670,95,'Image + “What color are the cabinets?”',size=28)
    b+=arrow(1190,445,1190,485)+box(980,495,420,95,'Generative VLM',fill=GREEN)
    b+=arrow(1190,590,1190,630)+text(1190,685,'“The cabinets are blue.”',32,anchor='middle')
    b+=foot('A VLM can supply perception or reasoning inside a larger robot controller.',size=29)
    add('VLMs can match, describe, and answer',b,
        'CLIP learns image and text encoders from paired data, with a contrastive objective that favors matching pairs over alternatives. A generative VLM such as PaliGemma predicts text conditioned on visual embeddings and a prompt. The question and answer are an illustrative teaching example, not a measured prediction. These two interfaces can support retrieval, open-vocabulary perception, or task reasoning. The architecture on the following slide follows a generative VLM, which is one common starting point for a VLA; it does not describe every vision-language model.',[CLIP,PALIGEMMA,PI05], 'VLM → VLA')

    # 7. Show exactly where robot adaptation changes an image-text model.
    b=photo(75,245,300,220,'aloha-frame0.png','Robot observation')
    b+=arrow(390,355,455,355)+box(465,275,300,150,'Vision encoder',['image features'])
    b+=arrow(765,350,840,350)+box(850,255,380,205,'Language backbone',['visual + text context'])
    b+=text(75,555,'“Insert the peg”',32)+text(75,610,'Measured robot state',30)
    b+=path('M 410 540 L 810 540 L 810 425',BLUE,2)+arrow(810,425,840,425)
    b+=arrow(1230,355,1320,355)+text(1340,340,'Action',32,700,TEAL)+text(1340,389,'decoder',32,700,TEAL)
    b+=path('M 1410 430 L 1410 550 L 1135 550',TEAL,2)+arrow(1135,550,1135,590,TEAL)
    b+=box(745,590,780,110,'Numeric commands → controller → motion',fill=GREEN,size=29)
    b+=text(75,722,'Robot demonstrations train the output to match the robot’s action interface.',30)
    b+=foot('The action decoder may emit discrete tokens or continuous vectors.',size=29)
    add('Vision–language–action models learn robot commands',b,
        'Vision–language–action models condition a learned action policy on images and language, often also robot proprioception. The diagram is generic: visual features are connected to an LLM-derived backbone, and an action output is trained using robot data. RT-2 represents robot actions as discrete tokens; OpenVLA adapts a pretrained VLM to action prediction. A flow-based VLA adds a continuous action expert. Robot supervision supplies the physical grounding that image–text pretraining alone does not establish: which coordinates, scale, and timing lead to successful motion. The actual robot controller still tracks the generated targets.',[RT2,OPENVLA,PI0], 'VLM → VLA')

    # 8. Both routes can use Transformers. Introduce noise before the word flow.
    b=text(75,214,'Both approaches can use Transformers. They differ in how an action is represented.',30)
    b+=text(75,295,'Autoregressive action tokens',34,700,BLUE)
    for i,l in enumerate(['117','42','208','…']):
        b+=box(75+i*165,350,135,90,l,size=32)
        if i<3:b+=arrow(215+i*165,395,235+i*165,395)
    b+=lines(75,515,['Quantize / encode the action.','Predict tokens, then decode numbers.'],29,gap=46)
    b+=text(75,651,'RT-2 / OpenVLA; FAST encodes chunks.',27,color=MUTED)
    b+=line(775,265,775,710)
    b+=text(835,295,'Continuous action generation',34,700,TEAL)
    for i,(label,v,fill) in enumerate([('Random array',['−0.8  0.6',' 0.2 −0.6',' 0.8  0.0'],PALE),('Refine',['−0.3  0.3',' 0.3 −0.2',' 0.7  0.2'],GREEN),('Action chunk',[' 0.2  0.0',' 0.4  0.2',' 0.6  0.4'],GREEN)]):
        x=835+i*239
        b+=text(x+100,346,label,25,700,anchor='middle')
        b+=rect(x,370,200,150,fill)+lines(x+100,411,v,26,gap=39,anchor='middle')
        if i<2:b+=arrow(x+204,445,x+230,445,TEAL)
    b+=lines(835,590,['Condition every refinement on the scene.','Only the final array becomes commands.'],28,gap=46)
    b+=text(75,740,'Flow training: mix a demonstrated chunk with noise; learn how to move back toward the data.',27)
    b+=foot('Toy normalized arrays: rows = future control steps; columns = action coordinates.',size=27)
    add('Actions can be generated as tokens or continuous arrays',b,
        'The token IDs and continuous arrays are illustrative, not model outputs. Autoregression is a valid robot-policy design: discrete actions are trained with a token prediction objective. FAST compresses action chunks before tokenization. In flow matching, training pairs a demonstrated normalized action array A with noise ε. An interpolation Xτ=(1−τ)ε+τA provides the input, and A−ε is a supervised direction. At inference a Transformer predicts successive refinements starting from noise, conditioned on the observation. The refinement index τ is not robot time: each array already contains a sequence of future commands. Noise and intermediate arrays are not sent to motors. Flow can represent multiple possible actions, while simple squared-error regression may average them. Autoregressive models can also represent multimodality; flow is not the only valid choice.',[PI0,('FAST','https://arxiv.org/abs/2501.09747')], 'VLM → VLA')

    # 9. A verified, interactive execution schedule from the detailed tutorial.
    s=foundations()[12].copy()
    s['title']='Predict a chunk, execute a prefix, observe again'
    out.append(s)

    # 10. pi0.5 task decomposition and continuous output in one figure.
    b=text(75,213,'Instruction: “Clean the bedroom.”',34)
    b+=photo(75,270,600,375,'pol-pi05-real-rooms.png','Real homes from the pi0.5 paper')
    b+=arrow(685,440,755,440)
    b+=box(775,270,695,115,'VLM: predict the next subtask',size=31)
    b+=text(1120,447,'“Pick up the pillow”',36,700,BLUE,anchor='middle')
    b+=arrow(1120,474,1120,515)
    b+=box(775,530,695,120,'Flow-based action expert',['observations + state + subtask'],fill=GREEN)
    b+=arrow(1120,650,1120,690)+text(1120,735,'50-step robot action chunk',31,700,TEAL,anchor='middle')
    b+=foot('π0.5 uses the same overall model for high-level text and low-level actions.',size=28)
    add('π0.5 connects a semantic subtask to a motor action chunk',b,
        'π0.5 extends the π0 family with heterogeneous co-training and hierarchical inference. Given observations and a broad instruction, it produces a subtask in language, then conditions its continuous action expert on that subtask and robot observations/state. The published system outputs 50-step low-level chunks, corresponding to one second at 50 Hz. For the mobile platforms these include arm, gripper and torso targets plus base velocities, not just joint positions. The full model supports both discrete text generation and flow matching. Pretraining uses FAST-encoded action tokens; post-training adds the continuous action expert. Its main motivation is generalization to previously unseen homes; it does not imply that arbitrary new physical skills appear without robot data. The diagram omits detailed attention masks and action-expert internals.',[PI05,PI05_BLOG], 'VLM → VLA')

    # 11. Three visually distinct sources of supervision.
    b=text(75,208,'Different data teach different parts of the same behavior.',32)
    items=[(75,'Robot trajectories','pol-pi05-mobile.png','How to move','images + state → actions'),
           (575,'Subtask annotations','pol-pi05-hl.png','What to do next','scene + goal → subtask'),
           (1075,'Web image–text data','pol-pi05-web.png','What things mean','image + question → answer')]
    for x,title_,img,role,example in items:
        b+=text(x,295,title_,30,700,BLUE)
        b+=photo(x,335,440,235,img,title_+' from pi0.5 Figure 4')
        b+=text(x,625,role,33,700,TEAL)+text(x,681,example,25)
    b+=line(75,735,1525,735)+foot('Co-training combines semantic supervision with physical demonstrations.',size=29)
    add('π0.5 learns from more than robot action labels',b,
        'The paper mixes data from different robots/environments with high-level task annotations and web vision-language tasks. These are distinct supervision sources, not evidence that unlabelled internet video alone supplies exact joint targets. The diagram groups the mixture into three teaching categories; the original recipe has finer categories, including verbal-instruction data. Images are credited crops of π0.5 Figure 4. Joint training lets shared representations learn semantics and physical behaviors, but the embodiment-specific action interface must still be supplied.',[PI05], 'VLM → VLA')

    # 12. WAM is introduced through two questions about the same scene.
    b=text(75,213,'A world-action model couples behavior with predictions of its consequences.',31)
    b+=photo(75,290,420,315,'mit-pick-sketch.png','Pick-and-place diagram from MIT')
    b+=text(75,652,'Current scene + task',30,700,BLUE)
    b+=arrow(505,445,595,445)+box(615,340,360,190,'Shared model',['visual / physical','representation'])
    b+=arrow(975,385,1055,305,TEAL)+box(1075,240,450,135,'Action',['how the robot should move'],fill=GREEN,size=32)
    b+=arrow(975,490,1055,605,WARM)+box(1075,550,450,135,'Future world',['what should change'],fill=SAND,size=32)
    b+=foot('Future = video, visual features, or 3D motion; the representation depends on the model.',size=27)
    add('WAM: learn actions together with future world changes',b,
        'World-action model is a family name rather than a single standardized architecture. A policy learns to choose actions; a world model learns how observations or states evolve, often conditioned on an action. A WAM couples these capabilities in training or generation. This picture shows a conceptual shared model and is not a claim that every WAM factorizes the same way. Future prediction can supply a physical learning signal or support planning, but not every deployment samples a future or searches action candidates. Some models retain action-only inference. A visually plausible future is not by itself evidence of accurate contact dynamics.',[UNITAS,MIT], 'World-action models')

    # 13. Clip to the central paper panel while retaining the original image asset.
    b=text(75,211,'UNITAS represents future gripper motion and scene motion in a common 3D frame.',30)
    b+='<svg x="75" y="250" width="820" height="550" viewBox="345 0 720 750" preserveAspectRatio="xMidYMid meet"><defs><clipPath id="unitas-center"><rect x="345" y="0" width="720" height="750"/></clipPath></defs><g clip-path="url(#unitas-center)">'
    b+=photo(0,0,1436,750,'overview-unitas.png','Central panel of UNITAS Figure 1, cropped to its 3D interface')+'</g></svg>'
    b+=text(955,300,'Policy mode',32,700,BLUE)
    b+=lines(955,353,['Observation + instruction','→ executable action chunk'],29,gap=44)
    b+=line(955,439,1525,439)
    b+=text(955,493,'World-model mode',32,700,TEAL)
    b+=lines(955,546,['Given a gripper trajectory,','predict scene-point motion.'],29,gap=44)
    b+=lines(955,680,['Action-only execution can skip','future-motion generation.'],28,gap=43,color=MUTED)
    add('A concrete WAM: predict motion in 3D',b,
        'UNITAS couples a point-dynamics expert and an action expert. The point-dynamics branch represents gripper and scene displacement trajectories in a metric frame; the action branch outputs executable end-effector commands. Scene prediction conditions on a supplied or generated gripper trajectory. Policy-only inference can omit both future point-flow branches. Therefore this model should not be drawn as necessarily doing imagined-rollout search before each robot action. The central panel of Figure 1 is displayed with the side panels clipped, without modifying its content. Credit: Wang et al., UNITAS, 2026, CC BY 4.0. The project offers an interactive 3D viewer; its link is in the footer.',[UNITAS,('Interactive 3D demo','https://unitas-wam.github.io/#explore')], 'World-action models')

    # 14. Explain timescales without assigning arbitrary universal frequencies.
    b=text(75,212,'A useful decomposition separates task decisions from continuous feedback.',31)
    b+=box(75,300,570,165,'Slow reasoning / planning',['choose a subtask or program'],fill=PALE)
    b+=arrow(645,380,880,380)+text(760,337,'goal / intent',26,anchor='middle')
    b+=box(900,300,625,165,'Fast motor control',['produce and track targets'],fill=GREEN)
    b+=path('M 1210 465 L 1210 620 L 365 620 L 365 475',TEAL,3)+arrow(365,475,365,470,TEAL)
    b+=text(785,672,'observations · progress · failure signals',30,anchor='middle')
    b+=text(75,748,'Often called System 2 and System 1. A/B labels depend on the paper.',28)
    b+=foot('The fast component may be a VLA, a classical controller, or a program using both.',size=28)
    add('Reasoning and motor control run at different timescales',b,
        'System 1/2 and A/B are informal architecture labels whose exact mapping varies. Here slow means less frequent semantic or planning updates, while fast means robot action generation and control feedback. A VLM may choose a goal for a VLA; generated code may instead select geometric targets for a conventional controller. The measured robot state closes the fast loop, and task progress can close a slower loop. One model can support both roles, as π0.5 illustrates. A system can also use separate models or machines. The slide intentionally avoids universal frequency claims.',[PI05,CAP], 'Control architectures')

    # 15. Code generation vs execution is the key boundary for the rest of the deck.
    b=text(75,211,'“Put every block into the bowl with the matching color.”',34)
    b+=photo(75,275,660,365,'recent-cap-block-bowl.png','Blocks-and-bowls example from Code as Policies')
    b+=box(850,265,675,100,'Coding model + robot API documentation',size=29)
    b+=arrow(1185,365,1185,400)
    b+=code(850,410,675,235,'for block in detect_blocks():\n    bowl = find_bowl(block.color)\n    pick_and_place(block, bowl)',26)
    b+=arrow(1185,645,1185,685)+text(1185,730,'Python executes the policy',31,700,TEAL,anchor='middle')
    b+=foot('The APIs may call perception models, geometry, motion planners, and feedback controllers.',size=26)
    add('Code as Policies makes a program the policy',b,
        'The photo is from the original Code as Policies project; the short program is authored pseudocode, not a claim about its exact API. The model writes code that composes provided perception/control operations, arithmetic, conditions and loops. A program can re-observe, branch and recover rather than simply replay commands. What the model generates depends strongly on the available API: pick_and_place hides more robotics than move_pose. CaP systems can use learned perception and pre-existing manipulation primitives. Separate the code-writing call from execution: some systems call the model repeatedly online, while others freeze and run a completed program.',[CAP], 'Code as Policy')

    # 16. Explain the source of numerical action vectors through arithmetic.
    b=text(75,214,'Code computes coordinates from measurements and an explicit action contract.',31)
    b+=code(75,280,890,305,'p = np.array(object_position())  # metres\np_above = p + [0.0, 0.0, 0.10]\nq_goal = inverse_kinematics(\n    position=p_above, orientation=R\n)\nrobot.move_joints(q_goal)',28)
    b+=box(1035,295,490,100,'Measured object position',size=28)
    b+=arrow(1280,395,1280,440)+box(1035,450,490,100,'Desired tool pose',size=30,fill=GREEN)
    b+=arrow(1280,550,1280,595)+box(1035,605,490,100,'Numeric joint targets',size=29,fill=GREEN)
    b+=lines(75,665,['IK = inverse kinematics: find joint angles','that place the tool at a requested pose.'],29,gap=43)
    b+=foot('Pseudocode; frame = robot base. Orientation R and controller semantics must be defined.',size=26)
    add('A program can produce precise numeric actions',b,
        'The LLM does not need to memorize all joint coordinates. It can write a computation that obtains a measured object position, applies a geometric offset, solves inverse kinematics and passes the result to a robot API. The toy code uses metres in the robot-base frame and assumes a chosen orientation R. Actual perception may use RGB-D, calibration, known geometry or a learned model. IK may have multiple solutions or no feasible solution, so practical execution adds collision, joint-limit and reachability checks. The low-level controller converts desired positions into actuator behavior. This is an explanatory program, not runnable robot code.',[MIT,CAP], 'Code as Policy')

    # 17. A concrete reusable skill, not just a name in a box.
    b=text(75,210,'A skill packages a behavior, its inputs, and evidence that it worked.',31)
    b+=code(75,270,830,355,'def grasp_and_lift(object_id):\n    pose = localize(object_id)\n    approach(pose)\n    close_gripper()\n    lift(distance=0.05)\n    if not object_is_held():\n        return Failure("grasp slipped")\n    return Success()',26)
    b+=text(990,300,'Inputs',30,700,BLUE)+text(990,350,'object identity + live sensing',28)
    b+=text(990,433,'Preconditions',30,700,BLUE)+text(990,483,'reachable object; empty gripper',28)
    b+=text(990,566,'Postcondition',30,700,TEAL)+text(990,616,'object remains held after lifting',28)
    b+=foot('Later attempts can reuse the function, its operating conditions, and its failure record.',size=28)
    add('What is stored in a skill library?',b,
        'The function is illustrative and deliberately exposes a feedback check after grasping. A stored skill may contain source code, parameters, preconditions, effects, dependencies, demonstrations, run logs and reliability estimates. Different papers use skill for different artifacts: a neural policy, a callable program, or a prompt/instruction document. This deck labels the artifact explicitly. RAX and RATs can retain program knowledge across attempts, but reusable source may still need adaptation to a new robot API or scene. A single successful execution does not establish a universal precondition.',[RAX,RATS], 'Code as Policy')

    # 18. Persistent state and the two loops are unambiguous.
    b=text(75,212,'Recursive self-improvement (RSI): what persists after a failed episode?',31)
    nodes=[(75,'Execute','policy + robot'),(475,'Diagnose','traces + outcome'),(875,'Revise','code / memory')]
    for x,h,d in nodes:b+=box(x,315,330,150,h,[d],fill=GREEN if h=='Execute' else PALE)
    b+=arrow(405,390,465,390)+arrow(805,390,865,390)
    b+=path('M 1205 390 L 1405 390 L 1405 610 L 240 610 L 240 475',BLUE,3)+arrow(240,475,240,470)
    b+=text(800,658,'Test the revision, then retain it for the next episode',31,700,BLUE,anchor='middle')
    b+=lines(75,747,['Frozen model weights can coexist with changing programs, skill libraries, and memory.'],29)
    b+=foot('Ask whether the policy, the tool library, the evaluator, or the improvement procedure changes.',size=26)
    add('RSI claims need an explicit improvement loop',b,
        'In recent robotics work, self-improvement often means repeated program synthesis, execution, feedback and retention of useful artifacts while the coding model stays fixed. Calling this recursive does not establish that the foundation model improves its own weights or that its own learning procedure is modified. Distinguish within-episode feedback control from between-episode persistent adaptation. Also distinguish improvement of a task program from changing the harness that judges and accepts it. The next case studies locate these boundaries and examine evidence rather than assuming a shared meaning of RSI.',[RAX,RATS,RSI], 'Code and self-improvement')
    assert len(out)==18
    return out


def closing_slides():
    out=[]
    b=text(75,214,'Proposed test: keep the execution system fixed and vary only the retained library.',31)
    b+=box(75,320,460,135,'Same base agent',['same tools and test budget'])
    b+=arrow(535,355,675,295)+arrow(535,420,675,545)
    b+=box(695,235,380,125,'Without learned skills',fill=PALE,size=28)
    b+=box(695,485,380,125,'With learned skills',fill=GREEN,size=28)
    b+=arrow(1075,295,1140,295)+arrow(1075,545,1140,545)
    b+=box(1160,235,365,125,'Fresh test episodes',size=27)
    b+=box(1160,485,365,125,'Same test protocol',size=27)
    b+=lines(75,711,['Measure task success, intervention, execution time, and the cost of acquiring the skills.'],28)
    b+=foot('Frozen artifacts and held-out layouts help separate learning from extra test-time attempts.',size=27)
    out.append(slide('How would we know that self-improvement helped?',b,
        'This is a proposed experimental design, not an additional result. Use paired or otherwise matched test episodes and retain all relevant budgets, tools and controller settings. To attribute improvement to the library, hold the execution agent fixed. A second experiment can vary the test-time agent. Use an untouched final evaluation when model/program selection repeatedly consults validation results. Report exploration cost, human interventions, inference overhead and physical task time separately. Record failures as well as successes and show whether new code is actually exercised. RATs provides a useful library/execution ablation; the RSI report illustrates how local acceptance and end-to-end performance can diverge.',[RATS,RSI], 'Synthesis'))

    b=table(75,250,1450,['System','Output at execution','Persistent learned artifact'],[
        ['VLM','features / scores / text','model weights'],
        ['VLA','action / action chunk','action-policy weights'],
        ['WAM','actions; optional future prediction','dynamics / policy weights'],
        ['Code-based agent','targets / commands via APIs','programs, skills, memory'],
    ],ratios=[.19,.39,.42],row_h=93,size=29)
    b+=lines(75,783,['One robot may combine all four; locate the control and learning boundaries.'],31,weight=700,color=BLUE)
    out.append(slide('Compare interfaces and what gets updated',b,
        'This table describes common cases, not hard category boundaries. Neural weights are usually updated during training and frozen for a deployment rollout. Code systems may call these trained models and may or may not revise code online. WAM future generation can be optional at inference. A hybrid might use a VLM for target selection, a WAM for prediction, a VLA for manipulation, and code to coordinate them. RSI is an update process applied to a chosen part of that system. Compare components only after specifying observations, action interface, training data and evaluation budget.',[RT2,UNITAS,CAP], 'Synthesis'))

    b=text(75,210,'Design exercise: a robot must clear a table in a new kitchen.',35)
    b+=photo(75,280,600,390,'pol-vlm-scene.png','Kitchen teaching scene from pi0.5')
    b+=lines(760,315,['1. What observations and action API exist?',
                        '2. Which behavior needs robot data?',
                        '3. What would a world model predict?',
                        '4. What can code improve after failure?'],30,gap=80)
    b+=line(75,720,1525,720)
    b+=text(75,769,'Read next:',27,700,BLUE)
    b+=text(235,769,'π0.5 → UNITAS → RAX / RATs → the 123-round RSI report',29)
    b+=foot('For each paper, draw: observations → computation → commands → feedback → retained change.',size=25)
    out.append(slide('Apply the framework to one robot task',b,
        'Suggested discussion: first specify camera/depth and robot state, then joint or tool-pose commands and the controller. Decide whether the central challenge is semantic grounding, dexterous behavior, consequence prediction, or task-level coordination. A VLA needs appropriate physical action supervision; a WAM needs future-state or trajectory targets; a code agent needs usable sensing and control interfaces. Propose a failure record and a held-out evaluation before choosing an RSI loop. Core readings are linked on their corresponding slides and in these notes. Style references supplied by the user were the Flex-π reading-group deck and How to Train and Fine-tune VLMs. Their organization inspired the figure-led teaching approach; no institutional branding is copied.',[PI05,UNITAS,RAX,RATS,RSI], 'Discussion'))
    return out
