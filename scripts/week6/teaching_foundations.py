"""Fourteen concrete teaching steps: robot records, action interfaces, and learning.

All new diagrams are editable SVG. Photographs and the episode video reuse the
primary-source assets documented by the existing Week 6 modules.
"""
from .common import *
from .common_visual import robot_demo_body
from .foundations import MIT_PICK, CONTROL, CS224R, ACT, UMI, META, ROWS, CONVERT
from .visual_foundations import arm, crop, camera

GREEN = '#e4f0eb'
GRAY = '#e9edf0'
SIM = ('ALOHA · Action interface', 'https://github.com/tonyzhaozh/act/blob/main/sim_env.py')
CONSTANTS = ('ALOHA · Joint order', 'https://github.com/tonyzhaozh/act/blob/main/constants.py')


def _bottom(value, y=792, size=30):
    return text(800, y, value, size, 600, BLUE, 'middle')


def _cells(x, y, values, widths, fills=None, size=30):
    b = ''
    for i, (value, w) in enumerate(zip(values, widths)):
        fill = fills[i] if fills else PALE
        b += rect(x, y, w-8, 78, fill, LINE)
        b += text(x+(w-8)/2, y+50, value, size, 600, BLUE, 'middle')
        x += w
    return b


def get_slides():
    out = []

    def add(title, b, notes, sources=(), section='Robot data and actions'):
        out.append(slide(title, b, notes, sources, section))

    # 1. An explicit learning objective, grounded in the actual data example.
    b = image(75,220,790,545,'aloha-frame100.png','Recorded ALOHA simulation, episode 0 at nominal time 2 seconds')
    b += text(940,250,'Learn the interface',43,700,BLUE)
    b += lines(940,327,['A camera gives an image.','The robot needs numbers.'],32,gap=49)
    for yy, num, head, detail in [(460,'1','Read an episode','images, measurements, commands'),
                                  (585,'2','Learn an action policy','input → demonstrated target'),
                                  (710,'3','Add vision and language','connect understanding to motion')]:
        b += circle(965,yy-10,24,PALE,BLUE)+text(965,yy,num,29,700,BLUE,'middle')
        b += text(1010,yy,head,31,700)+text(1010,yy+43,detail,25,color=MUTED)
    b += text(75,799,'Running data example: two simulated arms inserting a peg into a socket.',27,color=MUTED)
    add('From images and language to robot actions',b,
        'The goal is to understand the interfaces, not to memorize a list of methods. We will inspect an actual stored episode, decode the robot action, create supervised training pairs, and follow a learned prediction into execution. Later sections connect this interface to visual-language representations, generative action models, code-based control, and hierarchical systems. The cover is a frame from the simulated ALOHA insertion dataset, not a physical-robot rollout or a result from a model introduced in this lesson.',[META,CS224R],'Opening')

    # 2. Explicit input and output at every boundary in one feedback loop.
    b = text(75,222,'Observation',33,700,BLUE)
    b += image(75,250,330,248,'aloha-frame0.png','An image observed by the simulated ALOHA robot')
    b += text(75,539,'image + measured joints',27)
    b += arrow(420,368,485,368)
    b += rect(500,292,245,152,PALE,BLUE)+text(622,354,'Policy',37,700,BLUE,'middle')
    b += text(622,400,'chooses a target',26,anchor='middle')
    b += arrow(760,368,825,368)
    b += text(965,282,'Command',33,700,BLUE,'middle')
    b += text(965,364,'[0.009, −0.934, …]',28,600,BLUE,'middle')
    b += text(965,414,'14 desired coordinates',26,anchor='middle')
    b += arrow(1100,368,1165,368)
    b += rect(1180,292,345,152,GREEN,TEAL)+text(1352,354,'Controller',37,700,TEAL,'middle')
    b += text(1352,400,'tracks the target',27,anchor='middle')
    b += path('M 1352 445 L 1352 620 L 1225 620',TEAL,3)+arrow(1290,620,1225,620,TEAL)
    b += arm(930,495,.65,color=TEAL)+text(1080,751,'robot + world',28,700,TEAL,'middle')
    b += path('M 945 697 L 240 697 L 240 558',BLUE,3)+arrow(240,620,240,558)
    b += text(585,670,'new image + new measurements',30,600,BLUE,'middle')
    b += text(75,802,'A target is a request. Sensor feedback tells us what the robot actually did.',29,color=BLUE)
    add('The policy commands; the controller tracks',b,
        'This is the joint-position interface used in our running example. The policy takes observations and selects desired joint/gripper coordinates. A lower-level feedback controller tracks those targets through motor actuation. The robot and environment then produce the next observations. Proprioception is measurement of the robot itself, such as joint position or velocity; it does not include the entire scene. A controller uses measured errors, so the commanded target need not equal the next measured position. Some robotics literature uses policy and controller interchangeably, and some learned policies directly output torques; the distinction here describes this concrete position-control stack. The displayed command begins with actual episode-zero action values rounded for readability.',[CONTROL,SIM,ROWS])

    # 3. Preserve the actual, locally embedded episode demonstration.
    add('Watch one recorded demonstration',robot_demo_body(),
        'This is episode 0 of lerobot/aloha_sim_insertion_human, a recorded simulation. The original dataset has a nominal 50-Hz record rate. The local playback samples every fifth source frame, giving ten displayed frames per second and covering nominal times 0 through 9.9 seconds. The encoded container lasts about 10.24 seconds, so playback time is not a sensor timestamp. Pause while the arms approach, grasp, and align. Ask what must be inferred from the image and which robot measurements are also useful. The original task is two-arm peg insertion; this is not evidence that a language model controls this rollout.',[META,ROWS],'Recorded demonstration')

    # 4. Preserve the liked one-record page; expose the actual field names.
    b = image(75,250,670,502,'aloha-frame0.png','ALOHA simulation episode 0, frame 0')
    b += text(870,235,'One stored record',37,700,BLUE)
    for yy,head,field,detail in [(320,'RGB image','observation.images.top','480 × 640 × 3'),
                               (485,'Measured configuration','observation.state','float32[14]'),
                               (650,'Target action','action','float32[14]')]:
        b += text(870,yy,head,31,700)+text(870,yy+40,field,24,color=MUTED)
        b += text(870,yy+85,detail,32,color=BLUE)
    b += text(75,803,'14 = (6 arm coordinates + 1 gripper) × 2 arms',29,color=MUTED)
    add('One frame is more than an image',b,
        'These are the actual dataset fields and shapes. The image shown is episode 0 frame 0. RGB storage uses height 480, width 640, and three channels; a model loader may transpose it to channel-first tensors. Both observation.state and action contain fourteen float32 numbers, but one records measured configuration and the other a desired configuration. The joint and gripper ordering is unpacked on the next page. LeRobot organizes storage, timestamps, and access; it does not make all datasets share the same action meaning.',[META,ROWS,SIM])

    # 5. Every coordinate in the real action is visible, including both grippers.
    b = text(75,217,'Actual action · episode 0, frame 0 · values rounded to four decimals',27,color=MUTED)
    headers = ['waist','shoulder','elbow','forearm','wrist','wrist','gripper']
    details = ['','','','roll','angle','rotate','opening']
    xs = [340+i*168 for i in range(7)]
    for xx,head,detail in zip(xs,headers,details):
        b += text(xx+75,285,head,26,700,anchor='middle')
        if detail:b += text(xx+75,321,detail,25,anchor='middle')
    vals = [['0.0092','−0.9342','1.1735','−0.0046','−0.3114','−0.0061','0.1436'],
            ['−0.0061','−0.9587','1.1582','0.0138','−0.2991','0.0337','−0.0312']]
    for yy,name,row in [(355,'Left arm',vals[0]),(495,'Right arm',vals[1])]:
        b += text(75,yy+50,name,33,700,BLUE)
        b += _cells(340,yy,row,[168]*7,[PALE]*6+[GREEN],27)
    b += line(340,615,1338,615,BLUE,3)+text(839,661,'6 absolute joint targets · radians',30,700,BLUE,'middle')
    b += text(1423,654,'gripper',27,700,TEAL,'middle')+text(1423,693,'unitless',25,color=TEAL,anchor='middle')
    b += lines(75,741,['The two rows are concatenated into one 14D action vector.',
                       'Gripper scale: approximately 0 = closed, 1 = open.'],28,gap=48)
    add('Decode the complete 14D ALOHA command',b,
        'The first seven coordinates are the six left-arm revolute-joint targets and the left normalized gripper opening. The next seven repeat that ordering for the right arm. sim_env.py documents these as absolute joint position commands, not joint increments, tool coordinates, or torques. constants.py supplies the named joint order. Revolute joints use radians in the simulation. Gripper normalization is an affine mapping relative to calibration endpoints, and it is not clipped by the shown normalization function: the actual first right-gripper target is approximately −0.03124. Thus do not label every coordinate in the array as radians or promise a strict [0,1] bound. The stored target is associated with nominal frame time, not verified wall-clock time of a human action.',[ROWS,SIM,CONSTANTS])

    # 6. One fully declared alternative action interface, not a second dataset row.
    b = text(75,220,'Illustrative single-arm interface · one tool-motion command',29,color=MUTED)
    vals = ['0.02','0','−0.01','0','0','0.05','1']
    names = ['Δx','Δy','Δz','x','y','z','g']
    for i,(val,name) in enumerate(zip(vals,names)):
        x=200+i*175
        if 3 <= i <= 5:
            b+=f'<text x="{x+80}" y="295" font-size="34" font-weight="700" fill="{BLUE}" text-anchor="middle">Δr<tspan baseline-shift="sub" font-size="24">{name}</tspan></text>'
        else:
            b+=text(x+80,295,name,34,700,TEAL if i==6 else BLUE,'middle')
        b+=_cells(x,325,[val],[175],[GREEN if i==6 else PALE],38)
    b += line(200,447,716,447,BLUE,3)+text(458,495,'translation · meters',31,700,BLUE,'middle')
    b += line(725,447,1241,447,BLUE,3)+text(983,495,'rotation vector · radians',29,700,BLUE,'middle')
    b += text(1330,495,'gripper',29,700,TEAL,'middle')
    b += lines(115,600,['Move +2 cm along base x,','−1 cm along base z.'],32,gap=48)
    b += lines(685,600,['Rotate 0.05 rad about base z','(about 2.9°).'],32,gap=48)
    b += lines(1295,600,['Open.','1 = open','0 = closed'],29,gap=44,color=TEAL,anchor='middle')
    b += text(75,765,'All motion increments are expressed in the robot base frame.',31,700,BLUE)
    b += text(75,810,'Seven output coordinates describe this interface; they do not tell us how many arm joints exist.',26,color=MUTED)
    add('A different robot interface can use a 7D tool command',b,
        'This vector is a declared teaching example, not an ALOHA record or a measured DROID sample. The interface is [Δx, Δy, Δz, Δr_x, Δr_y, Δr_z, g]. Translation is in meters in the robot base frame. The three rotation coordinates form an incremental rotation vector expressed in that same frame; its direction is the rotation axis and its norm is the angle in radians. It is not an elementwise addition to Euler angles. Here only the z coordinate is nonzero, so the requested orientation increment is 0.05 radians about base z. The gripper convention is explicitly 1=open, 0=closed. Action dimension depends on the interface: a seven-joint arm with a gripper might instead receive eight joint/gripper targets. Other conventions use velocities, quaternions, absolute poses, or normalized commands.',[MIT_PICK,UMI])

    # 7. Follow the target into an actual control stack, without a geometry detour.
    b = text(75,219,'Continue the illustrative 7D command',28,color=MUTED)
    b += text(75,290,'Current tool position',31,700,BLUE)+text(75,346,'[0.40, 0.10, 0.20] m',33)
    b += text(75,416,'+ [0.02, 0, −0.01] m',33,color=TEAL)
    b += line(75,442,475,442)+text(75,499,'[0.42, 0.10, 0.19] m',34,700,BLUE)
    b += text(75,548,'new position target',28,color=MUTED)
    b += arrow(500,470,560,470)
    b += rect(575,330,370,220,PALE,BLUE)
    b += text(760,381,'Robot motion interface',31,700,BLUE,'middle')
    b += lines(760,434,['pose target + current joints','↓','feasible joint targets'],29,gap=44,anchor='middle')
    b += arrow(960,470,1020,470)
    b += rect(1035,330,480,220,GREEN,TEAL)
    b += text(1275,381,'Joint feedback controller',31,700,TEAL,'middle')
    b += lines(1275,434,['compare target with measurement','↓','motor actuation'],29,gap=44,anchor='middle')
    b += text(75,657,'Orientation is composed with the rotation increment; the gripper receives its own target.',28)
    b += text(75,717,'An inverse-kinematics solver finds joint positions; planning checks the route between them.',28)
    b += text(75,795,'A pose target can be unreachable or blocked. The execution interface must handle that case.',28,600,BLUE)
    add('How does a tool target become motor motion?',b,
        'The arithmetic shown is only the translation part of the previous declared interface. Position coordinates are all expressed in the base frame. For the rotation-vector convention, orientation is updated by composing the current rotation with the incremental rotation, on the side specified by the spatial/base-frame convention. We do not fabricate numerical joint targets without a robot model. In this illustrative stack, inverse kinematics searches for joint configurations satisfying the target tool pose, and motion planning considers a feasible path and limits. An endpoint IK solution alone is not a collision-free trajectory. The joint controller then tracks the requested joint targets with measurement feedback. Some robot interfaces combine these steps in Cartesian impedance or velocity control; this page presents one concrete route, not a required universal architecture.',[MIT_PICK,CONTROL])

    # 8. The complete temporal record, explicitly joined by an index.
    b = text(75,214,'Episode 0 · 500 ordered records · nominal 50 Hz (one record every 0.02 s)',29,color=BLUE)
    cols=[75,300,680,1030]
    for xx,label in zip(cols,['frame / time','camera image','measured state qₜ','target action aₜ']):
        b+=text(xx,284,label,30,700,BLUE)
    b+=line(75,308,1525,308)
    b+=text(75,384,'0 / 0.00 s',31)+image(300,327,240,180,'aloha-frame0.png','Actual episode-zero frame-zero image')
    b+=text(680,398,'float32[14]',31)+text(1030,398,'float32[14]',31)
    b+=text(680,448,'shoulder −0.9600',26,color=MUTED)+text(1030,448,'shoulder −0.9342',26,color=MUTED)
    b+=line(75,526,1525,526)
    b+=text(75,596,'100 / 2.00 s',31)+image(300,545,240,180,'aloha-frame100.png','Actual episode-zero frame-one-hundred image')
    b+=text(680,617,'float32[14]',31)+text(1030,617,'float32[14]',31)
    b+=text(680,667,'measured at this index',26,color=MUTED)+text(1030,667,'target at this index',26,color=MUTED)
    b+=text(1475,515,'⋮',35,color=MUTED,anchor='middle')+text(1475,740,'⋮',35,color=MUTED,anchor='middle')
    b+=text(75,797,'Each row pairs what the robot observed with the command stored at that frame.',30,600,BLUE)
    add('An episode is an aligned sequence of records',b,
        'The actual episode contains records 0 through 499. This page shows two real images, frame 0 and frame 100, with omitted intervening records indicated by ellipses. The first-row shoulder values are actual left-shoulder measurements and targets; frame 100 only shows verified field shapes, not invented values. The converter copies source qpos, images, and actions using the same index, then creates nominal timestamps as index/FPS. The simulation training loader begins its action target at that index. This establishes a dataset alignment convention, not a claim about exact hardware capture or command-issue timestamps. The episode lasts 500 command slots at 50 Hz; its final nominal sample is 9.98 seconds. Episode identity must be retained when records are concatenated. The converter marks the final record done; that flag alone is not a task-success label.',[META,ROWS,CONVERT])

    # 9. Human input is translated into precisely the labels learned below.
    b=text(385,224,'ALOHA: move leader arms',33,700,BLUE,'middle')
    b+=text(1170,224,'UMI: move a handheld gripper',31,700,BLUE,'middle')
    b+=crop(75,260,650,350,'foundations-aloha-teleop.jpg','60 90 1115 810',3000,2250,'ACT primary photograph of ALOHA leader-arm teleoperation')
    b+=crop(960,260,420,350,'foundations-umi.png','5 55 325 480',2041,604,'UMI primary photograph of handheld gripper and wrist camera')
    b+=arrow(385,615,385,652)+arrow(1170,615,1170,652)
    b+=lines(385,694,['leader joint positions','→ joint / gripper target labels'],30,gap=44,anchor='middle')
    b+=lines(1170,694,['tracked tool motion + opening','→ tool / gripper trajectory labels'],29,gap=44,anchor='middle')
    b+=text(75,805,'The demonstration records both what the learner can observe and what the person commands.',28,600,BLUE)
    add('Where do the action labels come from?',b,
        'ALOHA teleoperation maps human-moved leader-arm joint positions to follower-arm targets and records robot observations alongside those commands. The displayed photograph is a physical ALOHA setup from the ACT paper; it is not a photograph of the simulated episode. UMI records human manipulation through a portable camera-equipped gripper, reconstructs the gripper trajectory with visual-inertial SLAM, and measures gripper width. Its data and deployment interfaces align observations and future actions, and account for timing and coordinate conventions. The two systems do not produce interchangeable arrays without conversion. We use these images to show where supervision comes from; no learner-generated action labels or per-frame language annotations are implied.',[ACT,UMI,SIM])

    # 10. Preserve the liked trajectory-to-training-window diagram.
    b=text(75,220,'Input now',34,700,TEAL)+text(75,430,'Future target',34,700,BLUE)
    for i in range(5):
        x=395+i*190;b+=text(x+70,220,['t','t+1','t+2','t+3','t+4'][i],27,color=MUTED,anchor='middle')
        b+=circle(x+70,290,36,GREEN if i==0 else 'white',TEAL if i==0 else LINE)
        b+=text(x+70,301,'o',34,700,TEAL if i==0 else MUTED,'middle')
    for i,lab in enumerate(['aₜ','aₜ₊₁','aₜ₊₂','aₜ₊₃','aₜ₊₄']):
        b+=label_box(395+i*190,390,180,76,lab,PALE if i<4 else 'white',31)
    b+=line(395,505,1145,505,BLUE,4)+text(770,557,'target chunk Aₜ: four consecutive commands',31,700,BLUE,'middle')
    b+=arrow(465,326,465,384,TEAL)
    b+=line(1405,205,1405,615,WARM,3,'10 7')+text(1405,657,'episode end',26,color=WARM,anchor='middle')
    b+=text(75,673,'oₜ = image + measured configuration',30,700,TEAL)
    b+=text(75,727,'aₜ = one 14D target; Aₜ = [aₜ, aₜ₊₁, aₜ₊₂, aₜ₊₃]',30,700,BLUE)
    b+=_bottom('Slide the window to create more pairs; never cross an episode boundary.',803,29)
    add('Turn an episode into input–target training pairs',b,
        'The current observation is paired with the next H stored commands, beginning at the same index under this simulation alignment convention. Here H=4 is illustrative, and every action cell is a complete fourteen-dimensional command for both arms. The observation includes the image and measured joint/gripper configuration. Moving t generates more training examples from the same episode. Other tasks may use image history or language as additional inputs. The input does not include the future observations drawn for temporal context. At an episode boundary, either drop incomplete windows or use padding with a validity mask rather than spill into another episode. Evaluation splits should use whole episodes and suitable scene/task separation to avoid near-duplicate neighboring frames leaking across splits.',[ACT,CONVERT,CS224R])

    # 11. Distinguish command dimension from the prediction horizon numerically.
    b=text(75,220,'One command controls both arms at one execution step.',31)
    b+=text(75,295,'aₜ',49,700,BLUE)+text(185,291,'14 numbers',35,700,BLUE)
    b+=_cells(500,245,['6 left joints','gᴸ','6 right joints','gᴿ'],[340,130,340,130],[PALE,GREEN,PALE,GREEN],28)
    b+=line(75,365,1525,365)
    b+=text(75,437,'A chunk predicts',34,700,BLUE)+text(75,485,'H commands at once.',34,700,BLUE)
    b+=lines(75,578,['Example: H = 50, d = 14','shape = 50 × 14','700 output scalars'],34,gap=58)
    for row,label in [(0,'aₜ'),(1,'aₜ₊₁'),(2,'⋮'),(3,'aₜ₊₄₉')]:
        yy=410+row*79;b+=text(615,yy+48,label,29,700,BLUE,'end')
        for col in range(14):b+=rect(650+col*57,yy,50,61,GREEN if col in [6,13] else PALE,LINE)
    b+=text(1050,758,'14 coordinates per row',28,color=BLUE,anchor='middle')
    b+=text(75,807,'At 50 Hz, 50 command slots cover 1 second. The robot still has the same joints.',28,600,BLUE)
    add('Action dimension and chunk length are different axes',b,
        'A single action is a vector with d coordinates; an action chunk is an H-by-d matrix. With this robot interface, d=14. Choosing H=50 gives 700 scalar outputs, not a robot with 700 joints or degrees of freedom. At 50-Hz execution, fifty slots cover one second of playback; their nominal start times span 0 through 0.98 seconds relative to the first slot. This H is a teaching example and should not be read as a universal hyperparameter for ACT, diffusion policies, or any other method. Predicting a chunk does not require committing to all its rows, which motivates the prefix-execution example next.',[ACT,META])

    # 12. Training is supervised prediction; deployment does not have target labels.
    b=text(75,218,'Training: recorded inputs and demonstrated targets',33,700,BLUE)
    b+=image(75,265,260,195,'aloha-frame0.png','Recorded observation used as training input')
    b+=text(75,500,'image + measured qₜ',27)
    b+=arrow(355,360,435,360)
    b+=rect(450,287,290,145,PALE,BLUE)+text(595,350,'Policy πθ',37,700,BLUE,'middle')
    b+=text(595,397,'trainable weights θ',25,anchor='middle')
    b+=arrow(755,360,835,360)
    b+=text(855,327,'Predicted chunk Âₜ',30,700,BLUE)+text(855,380,'shape H × 14',29)
    b+=text(1160,245,'Recorded target Aₜ',30,700,TEAL)
    b+=arrow(1305,268,1305,312,TEAL)
    b+=rect(1160,325,350,100,GREEN,TEAL)+text(1335,385,'compare predictions',28,600,TEAL,'middle')
    b+=arrow(1110,361,1150,361)
    b+=path('M 1335 435 L 1335 525 L 595 525 L 595 445',TEAL,3)+arrow(595,485,595,445,TEAL)
    b+=text(955,508,'loss → update θ',29,700,TEAL,'middle')
    b+=line(75,568,1525,568)
    b+=text(75,629,'Deployment',33,700,BLUE)
    b+=text(75,703,'new image + qₜ',30)+arrow(345,692,435,692)
    b+=label_box(450,646,290,90,'trained policy',PALE,30)+arrow(755,692,835,692)
    b+=text(855,703,'predicted commands',29)+arrow(1165,692,1225,692)
    b+=text(1250,703,'controller',31,700,TEAL)
    b+=text(75,805,'Behavior cloning learns from action labels. At deployment, no human target is supplied.',29,600,BLUE)
    add('Behavior cloning learns the recorded input–action mapping',b,
        'Behavior cloning uses demonstrations as supervised training data. The input is the observation at the start of a window; the target is the demonstrated action or action chunk. A simple regression policy could minimize an L1 or squared error between predicted and target actions; a probabilistic or generative policy uses an objective appropriate to its output model. Numerical normalization is usually performed per action coordinate and undone before execution. The essential point is the training/deployment distinction: the recorded target is available to compute training loss, but it is not an input the deployed model can consult. The trained weights produce commands from fresh observations, and the controller executes them. Original ACT does not take a language instruction; adding language later is an extension of this input interface.',[CS224R,ACT])

    # 13. Show what each cell means and how a fresh prediction replaces the suffix.
    scenes=[]
    for stage in range(4):
        s=text(75,212,'Toy schedule: predict 6 commands; execute 2; observe again.',32,700,BLUE)
        s+=text(75,257,'Every cell is one complete 14D command. Columns are robot execution steps.',28,color=MUTED)
        for j in range(10):s+=text(450+j*103+45,314,str(j),27,color=MUTED,anchor='middle')
        s+=text(75,314,'step index',27,color=MUTED)
        for row,start in enumerate([0,2,4]):
            if row==1 and stage<2 or row==2 and stage<3:continue
            yy=350+row*111
            s+=text(75,yy+28,f'Observe at step {start}',28,700,BLUE)
            s+=text(75,yy+67,'→ predict a fresh chunk',25,color=MUTED)
            completed = (row==0 and stage>=1) or (row==1 and stage>=3)
            replaced = (row==0 and stage>=2) or (row==1 and stage>=3)
            for k in range(6):
                xx=450+(start+k)*103;done=completed and k<2
                fill=GREEN if done else (GRAY if replaced else PALE)
                col=TEAL if done else (MUTED if replaced else BLUE)
                s+=rect(xx,yy,94,76,fill,LINE)+text(xx+47,yy+48,'14D',27,700,col,'middle')
                if replaced and k>=2:s+=line(xx+12,yy+61,xx+82,yy+15,MUTED,2)
        captions=['Predict all six commands from the first observation.',
                  'Execute the green prefix: two 20 ms command slots.',
                  'At step 2, replace unused predictions using a new observation.',
                  'At step 4, repeat: observe, predict, and execute the next prefix.']
        s+=text(75,726,captions[stage],29,700,BLUE)
        scenes.append(s)
    b=steps('teaching-prefix',scenes,y=750)
    add('Execute a short prefix, then predict again',b,
        'This interaction is a toy receding-horizon schedule using the verified fourteen-dimensional robot interface. The model predicts H=6 commands, then the runtime executes h=2 commands at 50 Hz, covering forty milliseconds of playback. Every cell is an entire fourteen-dimensional target, not a single scalar or a model token. Rows distinguish predictions made from different observations; columns share the same robot execution index. Crossed-out gray cells are old predictions that are discarded and replaced by a new row. Green means executed. A controller can still use rapid feedback while following each target. The diagram omits inference and communication latency; real deployments must schedule targets with those delays in mind. H and h here are pedagogical choices, not the original ACT execution algorithm or its temporal-ensembling rule.',[ACT,UMI])

    # 14. Bridge to the next section by adding a concrete input, not a taxonomy.
    b=text(75,215,'Illustrative language-conditioned policy',29,color=MUTED)
    b+=image(75,258,360,270,'aloha-frame100.png','Example image input to a robot action policy')
    b+=text(75,569,'image + measured configuration',27,700,BLUE)
    b+=rect(75,628,505,99,GREEN,TEAL)+text(98,669,'Optional task instruction',26,700,TEAL)
    b+=text(98,708,'“Insert the peg into the socket.”',27)
    b+=arrow(455,410,645,410)+arrow(590,676,645,520,TEAL)
    b+=rect(660,320,350,240,PALE,BLUE)
    b+=text(835,380,'Learned policy',37,700,BLUE,'middle')
    b+=lines(835,438,['interpret the observation','select the next motion'],29,gap=49,anchor='middle')
    b+=arrow(1025,440,1120,440)
    b+=text(1320,309,'Robot interface',33,700,BLUE,'middle')
    for yy,lab in [(360,'command 1: 14 values'),(425,'command 2: 14 values'),(490,'…')]:
        b+=label_box(1135,yy,380,56,lab,PALE,27)
    b+=text(1325,606,'numeric targets',31,700,BLUE,'middle')
    b+=text(1325,651,'→ controller → motion',28,anchor='middle')
    b+=text(75,791,'Next: what can a vision–language model contribute to this learned mapping?',31,700,BLUE)
    add('Add language while keeping an executable action interface',b,
        'The next section adds language and pretrained visual representations to the interface we have established. The instruction on this page is the dataset task metadata string. It is not a per-frame spoken annotation, and it was not an input to the original ACT policy. This diagram illustrates a language-conditioned extension: images, robot measurements, and a task instruction condition a policy whose output still must be decoded into the declared robot commands. A familiar visual-language model may produce textual descriptions or answers; producing useful physical actions requires an action interface and suitable training or control machinery. Later pages distinguish direct vision-language-action policies from systems that generate code or call existing skills.',[META,ACT,CS224R],'Bridge to vision and language')

    assert len(out)==14
    return out
