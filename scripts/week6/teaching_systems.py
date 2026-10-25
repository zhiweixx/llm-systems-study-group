"""Fourteen teaching slides on programmatic and hierarchical robot control.

Toy programs, coordinates, and scenes are authored explanations, not robot runs.
Published architectural claims retain explicit system/version citations.
"""
from .common import *

CAP = ('Code as Policies', 'https://code-as-policies.github.io/')
CAP_PAPER = ('CaP · methods + APIs', 'https://arxiv.org/html/2209.07753v4')
HELIX = ('Helix · 20 Feb 2025', 'https://www.figure.ai/news/helix')
GROOT = ('GR00T N1 · §2.1', 'https://arxiv.org/html/2503.14734v2#S2.SS1')
PI05 = ('π0.5 · original paper', 'https://www.physicalintelligence.company/download/pi05.pdf')
OPENVLA = ('OpenVLA · §3', 'https://arxiv.org/html/2406.09246v3#S3')
DIRECT = ('Astra direct interface · App. A', 'https://arxiv.org/html/2609.24170v1')
HYBRID = ('Astra hybrid · RoboDojo', 'https://arxiv.org/html/2609.38537v1')
SECTION = 'Programs, hierarchy, and robot execution'


def box(x,y,w,h,title,sub=None,fill=PALE,color=BLUE,size=31):
    b=rect(x,y,w,h,fill,LINE)
    if sub:
        b+=text(x+w/2,y+h*.42,title,size,700,color,anchor='middle')
        b+=text(x+w/2,y+h*.76,sub,26,color=MUTED,anchor='middle')
    else:b+=text(x+w/2,y+h/2+size*.34,title,size,700,color,anchor='middle')
    return b


def small_code(x,y,w,h,content,size=29):
    return code(x,y,w,h,content,size)


def bowl(x,y,r=65):
    return f'<ellipse cx="{x}" cy="{y}" rx="{r}" ry="{r*.53}" fill="#e4f0eb" stroke="{TEAL}" stroke-width="3"/>' + path(f'M {x-r} {y} Q {x-r*.7} {y+r*.85} {x} {y+r*.86} Q {x+r*.7} {y+r*.85} {x+r} {y}',TEAL,3,'#d3e6df')


def cube(x,y,size=44,color=BLUE):
    return rect(x-size/2,y-size/2,size,size,color,'white')


def gripper(x,y,closed=False):
    gap=25 if closed else 46
    b=line(x,y-90,x,y-36,BLUE,8)+line(x-gap,y-36,x+gap,y-36,BLUE,6)
    b+=path(f'M {x-gap} {y-36} V {y+20} M {x+gap} {y-36} V {y+20}',BLUE,6)
    return b


def scene(x,y,w,h,stage=0):
    """Side-view toy scene. Each stage is a manually authored snapshot."""
    b=rect(x,y,w,h,'#fafcfd',LINE)
    floor=y+h-65; bx=x+w*.24; by=floor-1; tx=x+w*.75
    b+=line(x+25,floor+21,x+w-25,floor+21,LINE,3)+bowl(tx,floor-36,65)
    if stage==0:
        b+=cube(bx,by)+gripper(bx,floor-135)
    elif stage==1:
        b+=cube(bx,by)+gripper(bx,by,True)
    elif stage==2:
        b+=cube(tx,floor-140)+gripper(tx,floor-140,True)
    else:
        b+=cube(tx,floor-19,37)+gripper(tx,floor-140)
    return b


def sample_chunk(x,y,labels,highlight=0,w=108):
    b=''
    for i,label in enumerate(labels):
        b+=box(x+i*(w+10),y,w,70,label,fill='#e4f0eb' if i<highlight else PALE,size=28)
    return b


def get_slides():
    out=[]
    def add(title,b,notes,sources=()):
        out.append(slide(title,b,notes,sources,SECTION))

    # 1. Concrete physical referents, explicit instruction, and actual code logic.
    b=text(75,207,'“Put the blue block into the green bowl.”',34,700,color=BLUE)
    b+=image(75,280,660,280,'recent-cap-block-bowl.png','Physical block-and-bowl manipulation images from Code as Policies Figure 2a')
    b+=lines(75,620,['Project photographs: Code as Policies.',
                     'The program is an authored toy example.'],27,gap=44,color=MUTED)
    b+=text(835,247,'LLM output: a program',31,700,color=BLUE)
    program='block = detect("blue block")\nbowl = detect("green bowl")\np = get_position(block)\nmove_to(p + [0, 0, 0.12])\nmove_to(p)\ngripper("close")\nmove_to(p + [0, 0, 0.12])\nmove_to(get_position(bowl)\n        + [0, 0, 0.12])\ngripper("open")'
    b+=small_code(835,274,690,455,program,27)
    b+=text(835,785,'The supplied APIs give these names physical meaning.',26,color=BLUE)
    add('Code as Policies: write a program for the task',b,
        'Start by identifying the blue block and green bowl in the actual project photographs. These images are from Figure 2a; the instruction and program on this slide are authored teaching material, not a reproduced model output or an executed robot test. The toy uses vector addition and a fixed vertical gripper orientation. Its object anchors and 12-cm clearance are assumed suitable for this scene. The program chooses a sequence of perception and motion calls. Following slides define the external functions and show their execution. Do not describe this particular program as a general pick-and-place implementation: grasp geometry, calibration, motion planning and grasp-status checking belong to its provided interface.',[CAP])

    # 2. The API contract says what is already supplied to the generated code.
    b=text(75,207,'Provided functions: the program calls them; their implementations already exist.',31)
    b+=text(75,290,'Read the scene',34,700,color=BLUE)
    b+=text(855,290,'Execute a command',34,700,color=TEAL)
    api_left=[('detect("blue block")','→ object identifier'),
              ('get_position(block)','→ NumPy [x, y, z] vector; metres'),
              ('grasped(block)','→ True or False')]
    api_right=[('move_to([x, y, z])','→ completion status'),
               ('gripper("close")','→ completion status')]
    for i,(call,result) in enumerate(api_left):
        yy=355+i*125
        b+=text(75,yy,call,31,700,color=BLUE)+text(75,yy+49,result,28)
    for i,(call,result) in enumerate(api_right):
        yy=355+i*125
        b+=text(855,yy,call,31,700,color=TEAL)+text(855,yy+49,result,28)
    b+=line(785,270,785,695)
    b+=lines(855,650,['Perception, calibration, planning,','and servo control live below the API.'],27,gap=43,color=MUTED)
    b+=text(75,802,'Toy contract: one matching object; fixed gripper orientation; blocking motion calls.',25,color=MUTED)
    add('Define the API before reading the generated program',b,
        'These names and return types define an authored toy interface. detect returns the identifier of one uniquely matching object; get_position returns a NumPy three-vector for its calibrated anchor in metres in the robot base frame. Therefore adding a three-element list in the previous program performs numerical vector addition, not Python list concatenation. grasped is a supplied status predicate. move_to tracks a Cartesian target at fixed gripper orientation and returns when its motion operation completes; this does not itself verify task success. The gripper call opens or closes the fingers. The LLM composes these functions rather than implementing the detector, calibration, kinematics, or servo here. The broader Code as Policies method can also generate control logic and helper functions.',[CAP_PAPER])

    # 3. The runtime trace makes intermediate values observable.
    frames=[]
    calls=[('Read','p = get_position(block)','p = [0.40, 0.10, 0.03] m'),
           ('Grasp','move_to(p); gripper("close")','Target → joint controller → fingers close'),
           ('Transfer','move_to(get_position(bowl)\n        + [0, 0, 0.12])','The held block moves above the bowl'),
           ('Release','gripper("open")','A new image shows the resulting scene')]
    for i,(name,call,result) in enumerate(calls):
        s=text(75,208,'Toy execution trace · illustrative coordinates, not recorded robot data',28,color=MUTED)
        s+=text(75,304,f'{i+1}. {name}',39,700,color=BLUE)
        s+=small_code(75,349,705,125,call,28)
        s+=lines(75,552,[result],27,color=BLUE)
        s+=scene(855,282,670,355,i)
        s+=text(1190,686,'Scene after this operation',28,anchor='middle')
        frames.append(s)
    b=steps('systems-program-trace',frames,y=743)
    add('Run the program one operation at a time',b,
        'Advance through four authored snapshots. The first demonstrates a perception return value. The second requests the grasp position and closure. The third summarizes the lift-and-transfer part of the program; the full program includes a lift before lateral transfer. The fourth releases and receives a new image. The values are illustrative and use the toy base-frame convention, not the different world-frame convention of the later RoboDojo interface. The runtime, perception stack, planner, controller and physical world produce these intermediate values and effects. The diagram is a teaching animation rather than a physics simulation or execution of the displayed Python. A returned motion status and an observed task outcome are different checks.',[CAP])

    # 4. Generated code can repeatedly read sensors without a new LLM query.
    b=text(75,208,'Generate the loop once. Its runtime can read new observations repeatedly.',32)
    b+=small_code(75,290,700,390,'for attempt in range(3):\n    gripper("open")\n    p = get_position(block)\n    move_to(p)\n    gripper("close")\n    if grasped(block):\n        break',30)
    b+=box(875,280,550,100,'LLM writes this program',size=31)
    b+=arrow(1150,380,1150,435)
    b+=box(875,450,550,100,'Program runtime',size=33)
    b+=path('M 1425 500 H 1495 V 665 H 910 V 580',TEAL,3)
    b+=arrow(910,580,910,550,TEAL)
    b+=text(1200,627,'read → command → check',27,700,color=TEAL,anchor='middle')
    b+=text(75,777,'Sensor feedback can occur inside the generated program.',32,700,color=BLUE)
    add('A feedback loop does not require an LLM call at every step',b,
        'This authored bounded loop retries a grasp using newly queried positions and a supplied grasp-status check. It illustrates a runtime distinction: the LLM produces source code, and ordinary program execution repeats its body. A system may separately ask an LLM to revise code after failure, but that is an additional loop and is not required for each sensor query. The loop uses the previous slide’s toy API, including blocking moves and a fixed orientation. It omits recovery after exhausting attempts because the teaching point is where repeated feedback occurs. Code as Policies supports such perception-conditioned logic, but the exact program here is not a quoted experimental implementation.',[CAP_PAPER])

    # 5. Hold the physical task fixed while changing what the model emits.
    b=text(75,208,'Same task: move the blue block into the green bowl.',34,700,color=BLUE)
    b+=text(75,321,'VLA',36,700,color=BLUE)
    b+=box(285,270,420,115,'Images + instruction')
    b+=arrow(705,327,795,327)+box(810,270,330,115,'Numeric action',sub='e.g., pose increment')
    b+=arrow(1140,327,1230,327)+box(1245,270,280,115,'Controller',size=29)
    b+=text(285,437,'Robot demonstrations train the action mapping.',29,color=BLUE)
    b+=text(75,575,'Code policy',32,700,color=TEAL)
    b+=box(285,525,420,115,'Instruction + API examples')
    b+=arrow(705,582,795,582)+box(810,525,330,115,'Program',sub='calls + loops')
    b+=arrow(1140,582,1230,582)+box(1245,525,280,115,'API runtime',size=29)
    b+=text(285,692,'Motor competence comes from the supplied execution stack.',28,color=TEAL)
    b+=text(75,802,'The API implementation may itself contain learned policies.',27,color=MUTED)
    add('Change the model output, not the physical task',b,
        'The upper row uses original OpenVLA as a concrete numeric-action example: visual and language inputs are mapped to action tokens, decoded into a robot command, and executed by a controller. The lower row describes the authored program example. Its language model uses code knowledge and API examples to generate logic; its execution stack supplies physical capabilities. This does not mean a code-based robot needs no robot experience. Its detector, calibration, learned skills or controller may have been built or trained using embodied data. Conversely, a VLA may also be wrapped in tools or code. The rows distinguish the emitted object and the location of learned motor behavior, not two mutually exclusive species of robot.',[OPENVLA,CAP_PAPER])

    # 6. Motivation for hierarchy: multiple semantic jobs in one task.
    b=text(75,208,'Overall task: “Clean the table.”',36,700,color=BLUE)
    b+=box(540,275,520,105,'Choose the next subtask',size=34)
    b+=path('M 800 380 V 435 H 350 V 470',BLUE,3)+arrow(350,450,350,485)
    b+=path('M 800 435 H 1230 V 470',BLUE,3)+arrow(1230,450,1230,485)
    b+=box(75,500,550,110,'Put the cup in the tray',size=32)
    b+=box(955,500,570,110,'Wipe the spill',size=32)
    b+=arrow(350,610,350,655)+arrow(1230,610,1230,655)
    b+=sample_chunk(125,680,['a₀','a₁','a₂','…'],w=100)
    b+=sample_chunk(1010,680,['a₀','a₁','a₂','…'],w=100)
    b+=text(800,809,'A subtask can require many sensor-dependent motor commands.',29,700,color=BLUE,anchor='middle')
    add('A task contains decisions at different scales',b,
        'This authored clean-table example separates choosing a useful behavior from producing its motor commands. Putting a cup in a tray and wiping a spill require different motion patterns, and each behavior can involve many feedback updates. The diagram depicts possible subtasks, not an instruction to execute both branches simultaneously. Their order should depend on the actual scene and task. Hierarchical designs can reuse a motor behavior under several broader goals and let local feedback proceed between semantic updates. A hierarchy does not guarantee success: an incorrect subtask, a poor motion policy, or an unsuitable execution interface can each prevent completion.',[PI05])

    # 7. Operational definitions; neither label implies conscious reasoning.
    b=text(75,208,'“System 2” and “System 1” name roles in some robot architectures.',32)
    b+=box(75,320,395,145,'Scene + instruction',sub='What is the current task?')
    b+=arrow(470,392,570,392)
    b+=box(585,320,390,145,'System 2',sub='interpret task + scene')
    b+=arrow(975,392,1075,392)
    b+=box(1090,320,435,145,'System 1',sub='generate motor commands',fill='#e4f0eb',color=TEAL)
    b+=box(585,595,390,105,'Fresh local sensors',fill='white',size=31)
    b+=path('M 975 647 H 1307 V 500',TEAL,3)+arrow(1307,500,1307,465,TEAL)
    b+=text(75,593,'Semantic updates can be slower.',30,color=BLUE)
    b+=text(75,650,'Local control can update faster.',30,color=TEAL)
    b+=text(75,799,'A/B labels depend on the paper. System 2 need not produce a verbal reasoning trace.',27,color=MUTED)
    add('Separate semantic context from local action generation',b,
        'Use these labels operationally. Some designs separate visual-language interpretation from a motor-action network, allowing different update schedules. Semantic context can be a latent feature representation, not an explicit verbal plan or chain of thought. The motor module still uses a robot controller beneath its action output. There is no universal model size, frequency, message format, or training objective implied by the names. System A/B labels are paper-specific and are not treated here as standard aliases. GR00T N1, for example, connects VLM token features to a flow-based action module and trains the modules together. A modular learned policy and a separate-model deployment are different design decisions.',[GROOT])

    # 8. Two interfaces, explicitly alternatives rather than consecutive stages.
    b=text(75,208,'What information crosses the high-level / low-level boundary?',34)
    b+=text(75,315,'Text interface',32,700,color=BLUE)
    b+=box(375,270,300,110,'Semantic model',size=30)
    b+=arrow(675,325,775,325)
    b+=box(790,270,370,110,'“Put cup in tray”',fill='white',size=31)
    b+=arrow(1160,325,1240,325)
    b+=box(1255,270,270,110,'Motor policy',size=29)
    b+=text(375,435,'An explicit subtask conditions a learned behavior.',29,color=BLUE)
    b+=text(75,575,'Feature interface',32,700,color=TEAL)
    b+=box(375,530,300,110,'VLM features',size=30)
    b+=arrow(675,585,775,585)
    for j,v in enumerate([.35,.8,.5,.65,.2,.7]):
        b+=rect(795+j*55,550,40,70,PALE,LINE)+rect(795+j*55,620-v*70,40,v*70,TEAL,'none')
    b+=arrow(1160,585,1240,585)
    b+=box(1255,530,270,110,'Action network',fill='#e4f0eb',size=27)
    b+=text(375,695,'A learned representation need not decode into a sentence.',28,color=TEAL)
    b+=text(75,802,'Alternative interfaces—not two required stages of every robot system.',27,color=MUTED)
    add('A message can be a subtask or learned features',b,
        'The top row illustrates an explicit semantic-subtask interface, as used in the original π0.5 model factorization. The lower row illustrates feature conditioning; the colored bars are symbolic dimensions, not measured neural activations. GR00T N1 uses a sequence of VLM features with cross-attention, while original Helix uses one continuous latent vector. Those are related feature interfaces but not identical implementations. Neither learned features nor a System 2 label establish that a model produced a natural-language reasoning trace. An interface must be specified before deciding how it can be inspected, cached, updated, or reused.',[PI05,GROOT])

    # 9. The original Helix version provides a concrete feature interface.
    b=text(75,205,'Original Helix · February 2025 · jointly trained visuomotor system',31,700,color=BLUE)
    b+=box(75,300,355,130,'Image + robot state',sub='+ language instruction',fill='white')
    b+=arrow(430,365,540,365)
    b+=box(555,300,400,130,'S2: 7B VLM',sub='reported 7–9 Hz')
    b+=arrow(955,365,1050,365)
    b+=box(1065,300,460,130,'One latent vector',sub='continuous task conditioning')
    b+=path('M 1295 430 V 520 H 955',BLUE,3)+arrow(985,520,955,520)
    b+=box(555,480,400,145,'S1: 80M policy',sub='reported 200 Hz',fill='#e4f0eb',color=TEAL)
    b+=box(75,495,355,130,'Latest image + state',fill='white',size=31)
    b+=arrow(430,560,540,560,TEAL)
    b+=arrow(755,625,755,685,TEAL)
    b+=text(755,737,'Wrist, finger, head, and torso targets',31,700,color=TEAL,anchor='middle')
    b+=text(75,804,'Training: ~500 h teleoperation + hindsight language labels.',28,color=MUTED)
    add('Helix sends latent intent to a fast visuomotor policy',b,
        'This is the original February 2025 Helix, not a later Helix version. Figure reports a 7B VLM feeding one continuous latent vector to an 80M visuomotor Transformer. The action module also sees fresh images and robot state. Both are trained jointly with regression; inference uses separate onboard processes and a shared latent. The reported 7–9 Hz and 200 Hz rates describe this implementation, not universal definitions. Approximately 500 hours of teleoperation receive hindsight language labels. The outputs are upper-body targets, not direct evidence of torque-level LLM control. No explicit textual reasoning trace is required by this architecture.',[HELIX])

    # 10. Place the already-taught pi0.5 architecture on the same interface map.
    b=text(75,208,'Overall instruction ℓ: “Put the dishes away.”',34,700,color=BLUE)
    b+=rect(340,280,1185,355,'#fafcfd',LINE)
    b+=text(370,327,'π0.5 model: two output roles',31,700,color=BLUE)
    b+=box(75,405,230,130,'Observation',sub='images + state',fill='white',size=29)
    b+=arrow(305,470,390,470)
    b+=box(405,405,370,130,'Semantic prediction',sub='cₜ: “pick up the cup”',size=29)
    b+=arrow(775,470,885,470)
    b+=box(900,405,420,130,'Continuous action expert',sub='Aₜ: 50-step chunk',fill='#e4f0eb',size=28)
    b+=arrow(1320,470,1430,470)
    b+=text(1450,478,'Aₜ',39,700,color=TEAL)
    b+=path('M 190 405 V 365 H 1110 V 385',BLUE,2)+arrow(1110,385,1110,405)
    b+=text(795,702,'The inferred subtask and current scene condition motion.',31,700,color=BLUE,anchor='middle')
    b+=text(75,804,'A model’s internal modules do not determine how many deployment processes it needs.',26,color=MUTED)
    add('π0.5 uses an explicit semantic subtask',b,
        'Map the original π0.5 factorization onto the preceding interface distinction: p(A,c|o,ℓ)=p(c|o,ℓ)p(A|o,c). The low-level factor conditions on the inferred subtask and observation rather than directly on the overall task prompt. The semantic and continuous outputs belong to the paper’s unified model with different expert weights. This drawing does not assert a particular GPU allocation, number of OS processes, or asynchronous schedule. High-level labels supervise semantic decisions, while action demonstrations ground motor generation. The outer model boundary distinguishes internal computation roles from a system assembled by independently deploying two unrelated pretrained models.',[PI05])

    # 11. Simple logical schedule: new local input while a message persists.
    b=text(75,208,'Illustrative schedule: one intent persists through several fresh observations.',31)
    b+=box(75,285,1450,95,'Intent: put the cup in the tray',size=35)
    xs=[75,455,835,1215]
    labels=['Approach','Align','Carry','Release']
    for i,x in enumerate(xs):
        b+=path(f'M {x+290} 380 V 590 H {x+155}',BLUE,2)
        b+=box(x,460,240,95,'Fresh observations',fill='white',size=25)
        b+=path(f'M {x+120} 555 V 590 H {x+155}',TEAL,2)
        b+=arrow(x+155,590,x+155,620,TEAL)
        b+=box(x,620,310,95,labels[i],sub='several motor updates',fill='#e4f0eb',color=TEAL,size=30)
        if i<3:b+=arrow(x+310,665,x+365,665,TEAL)
    b+=text(75,798,'The action network can react before the next semantic message arrives.',30,700,color=BLUE)
    add('Keep the intent; refresh the local feedback',b,
        'This is an authored logical schedule, not a measured timing trace, fixed frequency ratio, or description of every hierarchical robot. The semantic message stays unchanged while the motor policy repeatedly receives fresh observations. Each displayed action box summarizes a behavior stage for readability; a real stage usually contains many numerical updates. The distinction is between refreshing sensor-conditioned local control and recomputing semantic context. A system could implement this with separate asynchronous processes or another scheduler. A reused message can also become stale, so the design needs an update rule appropriate to its task. These generic possibilities are not a claim that the original π0.5 deployment uses the exact Helix schedule.',[])

    # 12. Show a real schema with clearly authored numerical values.
    b=text(75,207,'A different interface: the model chooses the next end-effector target.',32)
    b+=small_code(75,285,690,285,'move_eef({\n  "targets": {"right_z": 1.05},\n  "note": "raise right gripper"\n})',30)
    b+=text(75,620,'z: 1.00 → 1.05 m (world frame)',30,700,color=BLUE)
    b+=text(75,670,'Other dimensions keep their observed values.',27)
    b+=arrow(765,327,880,327)
    b+=box(895,275,630,105,'Validate → IK / joint path',size=31)
    b+=arrow(1210,380,1210,430)
    b+=box(895,445,630,105,'Robot controller executes',size=32)
    b+=arrow(1210,550,1210,600,TEAL)
    b+=box(895,615,630,105,'New images + measured state',fill='#e4f0eb',size=30)
    b+=text(75,797,'Published tool fields; illustrative target value. The next model call uses feedback.',26,color=MUTED)
    add('A tool call can request a numeric robot target',b,
        'The call uses the move_eef schema from An Unexpected Robot Policy, Appendix A. The target value and note are authored illustrations. The published interface uses absolute world-frame grasp-point coordinates, permits subsets of dimensions, and retains observed values for unspecified dimensions. Non-learned software validates a request and attempts joint-path planning; accepted paths execute before a new observation returns. IK means inverse kinematics. A target request is different from generating a reusable Python program and from selecting a learned policy proposal. The model-facing motion tool does not supply semantic pick/place primitives. This is a simulation interface example, not a reported successful motion.',[DIRECT])

    # 13. The branch is exclusive; arrays are explicit pedagogical values.
    b=text(75,208,'RoboDojo hybrid: task-finetuned π0.5 proposes; Astra reviews.',32,700,color=BLUE)
    b+=box(75,285,430,110,'Learned policy proposal',sub='50 joint-command vectors')
    b+=sample_chunk(75,430,['a₁','a₂','a₃','…'],highlight=3,w=96)
    b+=text(75,549,'One coordinate: 0.10, 0.12, 0.14, … rad',25,color=MUTED)
    b+=arrow(505,340,600,340)
    b+=box(615,285,330,110,'Astra review',sub='images + proposal')
    b+=path('M 945 340 H 995 V 315 H 1030',TEAL,3)+arrow(1020,315,1050,315,TEAL)
    b+=path('M 995 340 V 570 H 1030',WARM,3)+arrow(1020,570,1050,570,WARM)
    b+=box(1065,255,460,120,'Accept first n = 3',sub='joint-command prefix',fill='#e4f0eb',color=TEAL)
    b+=text(1295,458,'OR',35,700,color=MUTED,anchor='middle')
    b+=box(1065,510,460,120,'Replace with targets',sub='e.g., z = [1.05, 1.08] m',fill='#f7efe7',color=WARM)
    b+=text(75,701,'Only one branch executes, then a new observation arrives.',31,700,color=BLUE)
    b+=text(75,801,'Array values are illustrative. Published limits: prefix 1–15; correction 1–5 steps.',26,color=MUTED)
    add('A general model can supervise a learned policy',b,
        'This follows the RoboDojo hybrid in Systematically Exploring. A released task-finetuned π0.5 policy proposes 50 joint-action steps. Astra either accepts a prefix of 1–15 steps or replaces it with 1–5 end-effector-target steps. The branches are mutually exclusive. The highlighted n=3, joint-coordinate values and two z targets are authored examples, not values extracted from a trial. Joint commands and Cartesian corrections use different action representations and their associated executors. The general model supplies review and correction while the robot policy supplies learned motion proposals. Frozen general-model weights do not remove the policy’s robot-specific training. No benchmark ranking or new robot test is asserted.',[HYBRID])

    # 14. A compact interface map, not a catalog of competing brand names.
    b=text(75,207,'Ask what comes out of the model—and what turns it into physical motion.',32)
    cols=[75,315,720,1135]
    heads=['Interface','Input','Output','Robot grounding']
    for x,h in zip(cols,heads):b+=text(x,292,h,31,700,color=BLUE)
    b+=line(75,323,1525,323)
    rows=[
        ('VLM',['image + question'],['text / features'],['needs an action interface']),
        ('VLA',['images + instruction'],['numeric command / chunk'],['learned from robot data']),
        ('Code policy',['instruction + API context'],['executable program'],['provided APIs + executor']),
        ('Dual system',['task + scene + local state'],['context → motor actions'],['trained motor module']),
    ]
    for i,(name,inputs,outputs,ground) in enumerate(rows):
        yy=389+i*110
        b+=text(cols[0],yy,name,30,700,color=BLUE)
        b+=lines(cols[1],yy,inputs,27,gap=39)
        b+=lines(cols[2],yy,outputs,27,gap=39)
        b+=lines(cols[3],yy,ground,26,gap=39)
        b+=line(75,yy+48,1525,yy+48,'#dce4e9')
    b+=text(75,805,'A dual-system VLA can also expose tools. These are composable interface choices.',27,color=MUTED)
    add('Which component actually supplies robot actions?',b,
        'Use this table to answer four concrete questions, rather than to rank model families. A general VLM output alone is not a calibrated executable robot command. A VLA learns an action mapping from robot supervision and still needs a controller. A generated program relies on the meaning and implementation of its APIs, which may contain learned robot skills. A dual-system architecture divides semantic context and motor generation; it may itself be a VLA and can also expose tools. The row labels overlap. The robot-grounding column identifies where physical action competence must be supplied, not a universal claim that each component was trained from scratch or that programs cannot contain learned control.',[OPENVLA,CAP_PAPER,GROOT])

    assert len(out)==14
    return out
