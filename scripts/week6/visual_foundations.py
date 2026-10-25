"""Twenty-four visual teaching steps for the expanded Week 6 deck."""
from .common import *
from .common_visual import robot_demo_body
from .foundations import MIT_PICK, CONTROL, CS224R, ACT, DP, UMI, DAGGER, META, ROWS, CONVERT
from math import sin, cos, pi

GREEN = '#e4f0eb'
GRAY = '#e9edf0'


def caption(value, y=785, size=31, color=BLUE):
    return text(800, y, value, size, 600, color, 'middle')


def camera(x, y, scale=1):
    return (f'<g transform="translate({x},{y}) scale({scale})">'
            + rect(0, 0, 100, 65, PALE, BLUE) + circle(55, 33, 19, 'white', BLUE)
            + rect(12, -11, 30, 11, PALE, BLUE) + '</g>')


def arm(x, y, scale=1, labels=False, color=BLUE):
    s = f'<g transform="translate({x},{y}) scale({scale})">'
    s += line(0, 310, 440, 310, MUTED, 2) + rect(30, 275, 90, 35, PALE, color)
    pts = [(75,275),(135,120),(290,70),(370,140)]
    for a, b in zip(pts, pts[1:]):
        s += line(*a, *b, LINE, 20) + line(*a, *b, color, 3)
    for j, (xx, yy) in enumerate(pts[:-1]):
        s += circle(xx, yy, 14, 'white', color)
        if labels: s += text(xx-30, yy-27, ['q₁','q₂','q₃'][j], 28, 700, color)
    s += path('M 370 140 L 400 140 M 400 119 L 425 119 M 400 161 L 425 161 M 400 119 L 400 161', color, 5)
    return s + '</g>'


def crop(x, y, w, h, filename, viewbox, source_w, source_h, alt):
    vx,vy,vw,vh=viewbox.split()
    clip_id='foundations-crop-'+filename.replace('.','-')+'-'+str(x)
    return (f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewBox="{viewbox}" '
            'preserveAspectRatio="xMidYMid meet" overflow="hidden">'
            + f'<defs><clipPath id="{clip_id}"><rect x="{vx}" y="{vy}" width="{vw}" height="{vh}"/></clipPath></defs>'
            + f'<g clip-path="url(#{clip_id})">'+image(0, 0, source_w, source_h, filename, alt)+'</g></svg>')


def chunk(x, y, count=6, first=0, origin=None, executed=0, old=False, cell=120):
    s = ''
    sub = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    sup = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    for j in range(count):
        fill = GREEN if j < executed else (GRAY if old else PALE)
        col = TEAL if j < executed else (MUTED if old else BLUE)
        xx = x + j*cell
        val = 'a'+str(first+j).translate(sub)
        if origin is not None: val += '⁽'+str(origin).translate(sup)+'⁾'
        s += rect(xx,y,cell-10,76,fill,LINE)+text(xx+(cell-10)/2,y+49,val,31,600,col,'middle')
        if old and j >= executed: s += line(xx+13,y+61,xx+cell-23,y+16,MUTED,1.5)
    return s


def route_scene(mean=False, x=0, y=0, scale=1):
    s = f'<g transform="translate({x},{y}) scale({scale})">'
    s += circle(180, 440, 15, 'white', BLUE)+circle(1280,440,15,'white',BLUE)
    s += rect(615,330,225,220,GRAY,MUTED)+text(727,370,'obstacle',31,color=MUTED,anchor='middle')
    s += path('M 195 440 C 400 440 400 240 610 240 L 845 240 C 1050 240 1060 440 1265 440',TEAL,7)
    s += path('M 195 440 C 400 440 400 640 610 640 L 845 640 C 1050 640 1060 440 1265 440',BLUE,7)
    s += text(727,211,'route A',32,700,TEAL,'middle')+text(727,690,'route B',32,700,BLUE,'middle')
    if mean:
        s += line(195,440,1265,440,WARM,4,'13 8')
        s += text(470,480,'mean',30,700,WARM,'middle')
    return s+'</g>'


def density(x, y, w, h, multimodal=False, color=BLUE):
    s=line(x,y+h,x+w,y+h,MUTED,2)
    if multimodal:
        d=f'M {x} {y+h} C {x+w*.18} {y+h} {x+w*.13} {y+h*.1} {x+w*.28} {y+h*.1} C {x+w*.43} {y+h*.1} {x+w*.38} {y+h} {x+w*.5} {y+h} C {x+w*.63} {y+h} {x+w*.6} {y+h*.3} {x+w*.73} {y+h*.3} C {x+w*.86} {y+h*.3} {x+w*.84} {y+h} {x+w} {y+h}'
    else:
        d=f'M {x} {y+h} C {x+w*.36} {y+h} {x+w*.32} {y} {x+w*.5} {y} C {x+w*.68} {y} {x+w*.64} {y+h} {x+w} {y+h}'
    return s+path(d,color,6)


def drift(x=0,y=0,scale=1,correction=False):
    s=f'<g transform="translate({x},{y}) scale({scale})">'
    s+=path('M 130 400 C 440 315 750 310 1250 340','#e8f0f5',100)
    s+=path('M 130 400 C 440 315 750 310 1250 340',BLUE,4)
    s+=path('M 130 400 L 470 408 L 835 515 L 1245 635',WARM,4)
    for xx,yy in [(130,400),(470,340),(835,321),(1245,340)]:s+=circle(xx,yy,14,'white',BLUE)
    for xx,yy in [(470,408),(835,515),(1245,635)]:s+=circle(xx,yy,14,'white',WARM)
    if correction:
        s+=arrow(835,515,925,382,TEAL)+text(973,466,'expert label',31,700,TEAL)
    return s+'</g>'


def get_slides():
    out=[]
    def add(title,b,notes,sources=(),section='Foundations'):
        out.append(slide(title,b,notes,sources,section))

    # 1. A single real teaching figure establishes the physical problem.
    b=image(75,210,900,550,'mit-pick.png','MIT manipulation: a robot reaching for a brick')
    b+=lines(1030,290,['Observe.','Choose.','Move.'],53,gap=105,weight=700,color=BLUE)
    b+=lines(1030,660,['What is an action?','How is it learned?','What does language add?'],29,gap=48)
    add('From images and language to robot motion',b,
        'The source illustration is the pick-and-place setup from MIT’s manipulation course. Start with the concrete task of moving an object to a desired pose. The central question is what numerical or executable output actually moves the machine. The lesson first follows sensing, geometry, data, and imitation learning, then adds pretrained vision/language models and LLM-based control interfaces. This figure is a teaching setup, not an experimental result from the ALOHA demonstration used later.',[MIT_PICK],'Opening')

    # 2. The feedback loop, without every definition on the same page.
    b=circle(265,450,115,PALE,BLUE)+text(265,460,'Policy',43,700,BLUE,'middle')
    b+=circle(775,450,115,PALE,TEAL)+text(775,460,'Controller',36,700,TEAL,'middle')
    b+=arm(1150,300,.78)
    b+=arrow(380,450,660,450)+text(520,416,'command',31,color=BLUE,anchor='middle')
    b+=arrow(890,450,1160,450,TEAL)+text(1020,416,'actuation',31,color=TEAL,anchor='middle')
    b+=path('M 1370 340 L 1370 240 L 265 240 L 265 335',BLUE,3)
    b+=arrow(265,285,265,335)+text(800,221,'images + robot measurements',31,color=BLUE,anchor='middle')
    b+=caption('An action changes the world—and the next observation.',760,35)
    add('A robot acts inside a feedback loop',b,
        'The policy chooses a command, such as a desired joint configuration. A lower-level controller uses measurements to track that command through actuation. The robot and its environment produce new observations. Proprioception means measurements of the robot itself, including joint positions and velocities. A policy may require observation history because measurements reveal only part of the physical state. “Controller” and “policy” overlap in some robotics usage; the separation here is a useful position-command example, not a universal taxonomy. Learned modules, classical planning, and feedback control can coexist in one stack.',[CONTROL,MIT_PICK])

    # 3. Task-space geometry before architectures.
    b=image(110,210,950,570,'mit-pick-sketch.png','MIT pick-and-place sketch showing intermediate gripper poses')
    b+=lines(1130,280,['Where is the object?','Where should it end up?'],31,gap=70,weight=700,color=BLUE)
    b+=lines(1130,535,['Approach','Grasp','Move','Release'],33,gap=58)
    add('A task goal becomes a sequence of geometric goals',b,
        'MIT’s pick-and-place chapter deliberately begins with ideal object-pose measurement to isolate geometry. The four gripper poses show why even a seemingly simple task needs an approach pose, a grasp, transfer, and release. Real perception must estimate object and tool poses from measurements, and contact/friction may remain uncertain. The idealized chapter assumes the object stays fixed in the world while ungrasped and fixed relative to the gripper while held. These are teaching assumptions, not general laws of manipulation. The picture introduces the need for target poses before discussing learned policies.',[MIT_PICK])

    # 4. The configuration and the tool pose are two descriptions.
    b=arm(120,270,1.24,True)
    b+=text(200,740,'Joint configuration q',35,700,BLUE)
    b+=text(1085,272,'Tool pose',37,700,BLUE,anchor='middle')
    b+=lines(1085,335,['position + orientation','in a stated frame'],32,gap=48,anchor='middle')
    b+=text(825,479,'q',40,700,BLUE,'middle')+text(1405,479,'pose',34,700,BLUE,'middle')
    b+=arrow(885,465,1310,465)+text(1095,435,'forward kinematics',30,color=BLUE,anchor='middle')
    b+=text(825,604,'q',40,700,TEAL,'middle')+text(1405,604,'pose',34,700,TEAL,'middle')
    b+=arrow(1310,590,885,590,TEAL)+text(1095,645,'inverse kinematics (IK)',30,color=TEAL,anchor='middle')
    b+=text(1085,740,'Joint target or tool target?',34,700,BLUE,anchor='middle')
    add('Joint angles and tool pose describe different things',b,
        'A configuration q is the set of the robot’s joint coordinates. The tool pose is its position and orientation relative to a reference frame. Forward kinematics computes pose from configuration; inverse kinematics seeks a configuration achieving a target pose. IK may have no solution or multiple solutions, and a valid endpoint does not establish a collision-free trajectory. The planar arm is an authored schematic, not the ALOHA robot. Revolute joint coordinates commonly use radians; prismatic coordinates use length. A tool-position target and a vector of desired joint angles are different action interfaces.',[MIT_PICK])

    # 5. Coordinate frames deserve a separate, visible example.
    b=image(80,195,600,550,'mit-frames.png','MIT robot coordinate frames attached to robot bodies')
    ox,oy=1020,520
    b+=arrow(ox,oy,ox+250,oy,BLUE)+arrow(ox,oy,ox,oy-235,BLUE)
    b+=text(1290,530,'x_W',29,color=BLUE)+text(1003,267,'z_W',29,color=BLUE)
    b+=arrow(ox,oy,ox+180,oy-180,TEAL)+arrow(ox,oy,ox-175,oy-175,TEAL)
    b+=text(1210,330,'x_tool',29,color=TEAL)+text(800,322,'z_tool',29,color=TEAL)
    b+=circle(ox,oy,10,'white',BLUE)
    b+=text(1080,615,'“Move +x” depends on the frame.',31,700,BLUE,'middle')
    b+=caption('Numbers need units, axes, and an origin.',790,34)
    add('A vector needs a coordinate frame',b,
        'The MIT figure shows frames attached to bodies. The right-hand illustration is a two-dimensional slice with world and tool axes sharing an origin for clarity; in general origins differ as well. A displacement expressed in one frame changes coordinates through rotation, while a point-position conversion generally includes translation. A numeric target must identify its reference frame, orientation convention, units, and whether it is absolute or relative. Scaling data by a mean and standard deviation is numerical normalization, not geometric frame conversion. Do not infer those conventions from the length of an array.',[MIT_PICK,('Modern Robotics · Transforms','https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/')])

    # 6. Three execution layers; the camera sample is an input event.
    b=text(75,204,'Illustrative timing · physical time continues during inference',28,color=MUTED)
    x0,scale=430,5
    for t in [0,50,100,150,200]:
        x=x0+t*scale;b+=line(x,290,x,710,GRAY,2,'5 6')+text(x,272,f'{t} ms',25,color=MUTED,anchor='middle')
    b+=camera(325,300,.65)+arrow(390,330,430,365)
    b+=text(75,385,'Model',34,700)+rect(430,338,400,75,PALE)+text(630,387,'80 ms computation',28,600,anchor='middle')
    b+=rect(930,338,400,75,PALE)+text(1130,387,'next model call',28,600,anchor='middle')
    b+=text(75,530,'Commands',34,700)+text(75,572,'50 Hz playback',26,color=MUTED)
    b+=arrow(830,413,830,475)
    for j in range(6):b+=rect(830+j*100,475,94,75,GREEN,TEAL)
    b+=text(630,520,'previous target / hold',26,color=MUTED,anchor='middle')
    b+=text(75,677,'Joint servo',34,700)+text(75,720,'1 kHz feedback',26,color=MUTED)
    b+=rect(430,635,1000,65,'#f3f7f4','none')
    for j in range(201):b+=line(430+j*5,642,430+j*5,693,TEAL,1)
    b+=caption('Fast playback does not imply fast model decisions.',795,32)
    add('One slow decision can produce many fast commands',b,
        'All rates and latencies on this page are invented for teaching. The camera icon marks an image available at the beginning of the first model call. Its result arrives 80 milliseconds later; during that interval the system executes an earlier target or a hold behavior. The result is then played as commands at 50 Hz. The joint servo continues to use rapid sensor feedback, even while model computation is in progress. Servo ticks are one millisecond apart. Actual systems must measure sensor, transport, computation, scheduling, and actuation delays. UMI explicitly aligns captured observations and future execution timestamps. Some simulators pause during inference, which changes the physical-time interpretation.',[UMI,CONTROL])

    # 7. Root-owned embedded recording, with no duplicate captions.
    add('Watch one recorded demonstration',robot_demo_body(),
        'Play the actual simulated ALOHA insertion episode before interpreting the data. This is episode 0 of lerobot/aloha_sim_insertion_human, not a rollout generated by our lesson or a physical-robot video. The original records are nominally 50 Hz. The embedded offline video samples every fifth source frame, giving ten displayed frames per second and covering nominal episode times 0 through 9.9 seconds. The encoded container lasts about 10.24 seconds; do not infer sensor timestamps from playback time. Ask which observations reveal approach, alignment, grasp, and insertion. The native controls allow pause and replay, while print uses a poster image.',[META,ROWS],'Robot demonstration')

    # 8. Disentangle schema from the command/outcome lesson on the next page.
    b=image(75,250,670,502,'aloha-frame0.png','ALOHA simulated insertion, episode 0 frame 0')
    b+=text(885,260,'One record',38,700,BLUE)
    for y,head,detail in [(345,'RGB image','480 × 640 × 3'),(485,'Measured configuration','14 numbers'),(625,'Target action','14 numbers')]:
        b+=text(885,y,head,31,700)+text(885,y+49,detail,34,color=BLUE)
    b+=text(75,803,'14 = (6 arm coordinates + 1 gripper) × 2 arms',29,color=MUTED)
    add('One frame is more than an image',b,
        'The image is actual episode 0 frame 0. The LeRobot record contains observation.state and action as float32 arrays of length fourteen: six left-arm coordinates, left gripper, six right-arm coordinates, right gripper. RGB storage is height 480 by width 640 by three channels; a loader may return channel-first tensors. Episode 0 has 500 records at nominal 50 Hz. The current task metadata string is “Insert the peg into the socket.” This is a dataset label, not per-frame speech or evidence that original ACT consumes language. A state field here describes measured robot variables, not the entire physical world. LeRobot is storage/access machinery rather than a universal action schema.',[META,ROWS,('LeRobot · Format','https://huggingface.co/docs/lerobot/lerobot-dataset-v3')],'Robot data')

    # 9. Actual measurements, two samples only.
    b=text(75,215,'Left shoulder · episode 0 · radians · nominal frame time',29,color=MUTED)
    x0,x1=480,1260;y=lambda q:680-(q+.965)/.04*390
    b+=line(350,260,350,710,MUTED,2)+line(350,710,1390,710,MUTED,2)
    for q in [-.960,-.950,-.940,-.930]:b+=text(325,y(q)+9,f'{q:.3f}',26,color=MUTED,anchor='end')+line(350,y(q),1390,y(q),GRAY,1.5)
    target=-.9341943264
    b+=line(x0,y(target),x1,y(target),WARM,4)
    for x in [x0,x1]:b+=circle(x,y(target),11,'white',WARM)
    b+=text(880,y(target)-35,'stored target  −0.9342',32,700,WARM,'middle')
    q0,q1=-.9599999785,-.9566894174
    b+=line(x0,y(q0),x1,y(q1),BLUE,4)+circle(x0,y(q0),12,'white',BLUE)+circle(x1,y(q1),12,'white',BLUE)
    b+=text(x0,y(q0)-35,'measured −0.9600',29,600,BLUE,'middle')+text(x1,y(q1)-35,'−0.9567',29,600,BLUE,'middle')
    b+=text(x0,753,'0.00 s',29,anchor='middle')+text(x1,753,'0.02 s',29,anchor='middle')
    b+=caption('A desired position is different from a measured position.',805,32)
    add('The command is not the outcome',b,
        'These are rounded values from the first two stored rows, not synthetic dynamics. The connected segments only join sampled values; they do not establish the continuous response between samples. The target is an absolute desired revolute-joint coordinate in radians. The converter copies image, qpos, and action at the same index and constructs nominal time as frame index divided by FPS. The original ACT simulation loader starts the target chunk at that index. Therefore call it a stored target at nominal frame time, not a documented human wall-clock command. The grippers use a separate normalized opening convention, nominally zero closed and one open. A command does not guarantee the next measured position.',[ROWS,CONVERT,('ACT · Simulation API','https://github.com/tonyzhaozh/act/blob/main/sim_env.py')],'Robot data')

    # 10. Dataset windows with a visible episode boundary.
    b=text(75,220,'Input now',34,700,TEAL)+text(75,430,'Future target',34,700,BLUE)
    for i in range(5):
        x=395+i*190;b+=text(x+70,220,['t','t+1','t+2','t+3','t+4'][i],27,color=MUTED,anchor='middle')
        b+=circle(x+70,290,36,GREEN if i==0 else 'white',TEAL if i==0 else LINE)
        b+=text(x+70,301,'o',34,700,TEAL if i==0 else MUTED,'middle')
    for i,lab in enumerate(['aₜ','aₜ₊₁','aₜ₊₂','aₜ₊₃','aₜ₊₄']):
        b+=label_box(395+i*190,390,180,76,lab,PALE if i<4 else 'white',31)
    b+=line(395,505,1145,505,BLUE,4)+text(770,557,'one target chunk Aₜ',34,700,BLUE,'middle')
    b+=arrow(465,326,465,384,TEAL)
    b+=line(1405,205,1405,615,WARM,3,'10 7')+text(1405,657,'episode end',26,color=WARM,anchor='middle')
    b+=mathml(325,650,950,100,'<mrow><msub><mi>A</mi><mi>t</mi></msub><mo>=</mo><mo>[</mo><msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>a</mi><mrow><mi>t</mi><mo>+</mo><mi>H</mi><mo>−</mo><mn>1</mn></mrow></msub><mo>]</mo></mrow>',43)
    b+=caption('Keep each training window inside its episode.',800,32)
    add('Turn a trajectory into training pairs',b,
        'The green observation at t is paired with the next H stored commands, beginning at t under the verified simulation alignment convention. H=4 is illustrative. The action dimension d_a is independent of H; it is fourteen for the ALOHA example. Input history or optional language can be added when the selected method uses them. Original ACT uses images and joint observations, without language. A different acquisition pipeline may need a different action/observation shift. Split evaluation data by whole episodes and relevant scenes or tasks to avoid neighboring-frame leakage. At episode ends, either drop an incomplete window or pad it with a validity mask; do not silently continue into the next episode. The final-record next.done flag is not itself a success label.',[ACT,CONVERT],'Robot data')

    # 11. Real source photographs; native SVG crop keeps source assets intact.
    b=text(385,225,'Move leader robot arms',33,700,BLUE,'middle')+text(1170,225,'Move a handheld gripper',33,700,BLUE,'middle')
    b+=crop(75,260,650,440,'foundations-aloha-teleop.jpg','60 90 1115 810',3000,2250,'ACT source photograph: a person teleoperates ALOHA leader arms')
    b+=crop(960,260,420,440,'foundations-umi.png','5 55 325 480',2041,604,'UMI source photograph: handheld gripper with wrist camera')
    b+=text(385,745,'ALOHA teleoperation',31,700,BLUE,'middle')+text(1170,745,'UMI portable demonstrations',31,700,BLUE,'middle')
    b+=caption('Collection interfaces determine which observations and actions are recorded.',805,28)
    add('How does a person provide a demonstration?',b,
        'The ALOHA photograph is a crop of the original ACT setup figure. The person backdrives leader arms; follower arms mirror their joint configuration, and leader positions provide command labels while follower positions provide measurements. The UMI photograph is cropped from its hardware figure and shows a handheld gripper with camera/IMU rather than a leader robot. UMI estimates gripper trajectories and handles latency/alignment when transferring them to a robot. Neither raw human video nor a shared file format automatically supplies executable robot labels. DROID is another collection route, using a common real robot across diverse environments; Open X-Embodiment aggregates many datasets and embodiments. Their action units, frames, absolute/relative conventions, and timing must remain explicit. Original source assets are preserved; only the displayed viewports are cropped.',[ACT,UMI,('Open X-Embodiment','https://arxiv.org/html/2310.08864v6')],'Demonstrations')

    # 12. One objective, drawn as the familiar statistical problem.
    b=''
    for j in range(4):
        yy=300+j*77;b+=circle(160,yy,23,PALE,BLUE)+text(160,yy+10,'o',28,600,BLUE,'middle')
        b+=arrow(200,yy,320,yy)+text(360,yy+10,'a*',34,700,TEAL)
    b+=text(260,240,'demonstration pairs',31,700,BLUE,'middle')
    b+=arrow(460,425,630,425)+circle(810,425,130,PALE,BLUE)+text(810,438,'πθ',53,700,BLUE,'middle')
    b+=arrow(940,425,1090,425)+text(1280,423,'action distribution',32,700,BLUE,'middle')
    b+=density(1110,470,340,125,True)
    b+=mathml(180,675,1240,100,'<mrow><munder><mo>min</mo><mi>θ</mi></munder><mspace width="0.6em"/><msub><mi>𝔼</mi><mrow><mo>(</mo><mi>o</mi><mo>,</mo><mi>a</mi><mo>)</mo><mo>∼</mo><mi>𝒟</mi></mrow></msub><mo>[</mo><mo>−</mo><mi>log</mi><msub><mi>π</mi><mi>θ</mi></msub><mo>(</mo><mi>a</mi><mo>|</mo><mi>o</mi><mo>)</mo><mo>]</mo></mrow>',43)
    add('Behavioral cloning is supervised learning of actions',b,
        'Behavioral cloning treats demonstrated observation/action pairs as supervised data. The policy can represent a conditional action distribution rather than only a point prediction. Negative log likelihood is a common explicit-density formulation; under a fixed-variance Gaussian assumption it reduces to squared-error regression up to constants and scale. The same idea applies to an action chunk A rather than a single action a. Diffusion and flow-matching policies learn conditional distributions with their respective denoising or vector-field objectives; this equation is not a claim that every policy directly evaluates tractable likelihood. No reward function is needed for this imitation objective. Reinforcement learning will later change the learning signal to cumulative reward.',[CS224R],'Imitation learning')

    # 13. Ask first; reveal the mean on a second frame.
    assumption=text(75,200,'Same observation; two equally common tool-target sequences.',27,color=MUTED)
    scenes=[assumption+route_scene(False,70,35,.94)+caption('Both routes are demonstrated. Where does squared-error regression aim?',790,30),
            assumption+route_scene(True,70,35,.94)+caption('A conditional mean can lie between valid actions.',790,33)]
    add('Two valid actions can have an invalid average',steps('visual-multimodal',scenes,y=705),
        'Pause on the first frame and ask for the squared-error optimum. The second frame draws a pointwise mean between two equally weighted geometric command routes. That average intersects the obstacle although each demonstrated route avoids it. This is an authored illustrative command-space example, not a measured robot rollout. It motivates representing more than a conditional mean. It does not establish that all real datasets are strongly multimodal or that multimodality explains every empirical gain of generative policies. Temporal correlation, optimization, robustness, and action representations also matter. A larger deterministic network alone does not change the distribution family.',[CS224R,DP],'Action distributions')

    # 14. Distribution transformation before any denoising equations.
    b=density(100,340,410,240,False,TEAL)+density(1010,340,450,240,True,BLUE)
    b+=text(305,286,'Easy to sample',37,700,TEAL,'middle')+text(1235,286,'Useful actions',37,700,BLUE,'middle')
    b+=arrow(560,460,950,460)+text(755,426,'learn a transformation',30,700,BLUE,'middle')
    b+=camera(705,195,.9)+arrow(750,280,750,360)
    b+=text(1020,242,'condition on observation',27,color=MUTED,anchor='middle')
    b+=text(305,633,'Gaussian noise',31,color=TEAL,anchor='middle')+text(1235,633,'conditional distribution',31,color=BLUE,anchor='middle')
    b+=caption('Generate a sample, rather than only the mean.',780,35)
    add('Learn to turn noise into an action',b,
        'This is the broad generative-policy intuition used by the Stanford lecture: start from a distribution that is easy to sample and learn a conditioned transformation toward demonstrated actions. The drawn curves are illustrative one-dimensional marginals, not the full high-dimensional chunk distribution. The observation is held fixed while generating a candidate action. Different noise samples can lead to different valid candidates. This page intentionally precedes the particular diffusion algorithm. Diffusion and flow matching differ in their training objectives and sampling formulations; neither is merely training a deterministic regression from arbitrary independent noise directly to an arbitrary demonstration.',[CS224R,DP],'Action distributions')

    # 15. One large trajectory-space figure.
    b=drift(0,-30,1.12)
    b+=text(650,255,'Expert demonstrations',35,700,BLUE,'middle')
    b+=text(1180,725,'Learner rollout',35,700,WARM,'middle')
    b+=text(200,790,'A small error can lead to an unfamiliar next input.',35,700,BLUE)
    add('The learner changes its own input distribution',b,
        'The pale region represents observations or states encountered by demonstrations; this is not a physical tool path or a quantitative density contour. The learned policy can leave that region after an imperfect action. Its next input is therefore sampled from the distribution induced by the learner, which may differ from the expert distribution. Errors can compound. The key contrast with ordinary held-out prediction is that the policy’s outputs affect its future inputs. Do not quote a worst-case horizon-squared bound without its assumptions, or claim that chunking removes distribution shift. Offline action loss is useful, but task completion and recovery require rollouts.',[DAGGER,CS224R],'Closed-loop learning')

    # 16. Corrective labels, with iteration shown spatially.
    b=drift(10,45,.70,True)
    b+=text(460,235,'Visit learner states',35,700,WARM,'middle')
    b+=arrow(940,410,1120,410,TEAL)
    b+=text(1295,338,'Add expert labels',34,700,TEAL,'middle')
    for j in range(3):b+=rect(1160+j*12,380+j*15,230,120,'white',LINE)
    b+=text(1295,460,'𝒟 ← 𝒟 ∪ 𝒟new',32,700,TEAL,'middle')
    b+=text(1295,585,'Retrain',38,700,BLUE,'middle')
    b+=path('M 1295 620 L 1295 700 L 450 700 L 450 615',BLUE,3)+arrow(450,670,450,615)
    b+=caption('DAgger collects labels where the learner actually goes.',805,34)
    add('Ask the expert about the learner’s mistakes',b,
        'DAgger iterates between running the current policy, querying an expert for actions at the visited states, aggregating those labels with prior data, and retraining. The green correction arrow is an expert action label at a learner-visited state; it is not a guarantee that the current learner already executed the correction. The original algorithm can mix expert and learner control during collection. Human-gated intervention is a related practical variant in which a person takes over and supplies corrective demonstrations. Labeling and takeover are different interfaces. Expert availability and the cost of collecting new trajectories remain part of the method’s assumptions.',[DAGGER,CS224R],'Closed-loop learning')

    # 17. Make the array dimensions visible.
    b=camera(125,350,1.5)+text(200,510,'oₜ',39,700,BLUE,'middle')
    b+=arrow(315,415,475,415)+circle(610,415,105,PALE,BLUE)+text(610,430,'Policy',36,700,BLUE,'middle')
    b+=arrow(715,415,870,415)
    for j in range(6):
        xx=900+j*95;b+=rect(xx,285,78,300,PALE,LINE)
        for k in range(14):b+=circle(xx+39,304+k*20,4,TEAL if k in [6,13] else BLUE,TEAL if k in [6,13] else BLUE)
        b+=text(xx+39,635,f'{j}',27,color=MUTED,anchor='middle')
    b+=text(1180,240,'H future commands',34,700,BLUE,'middle')
    b+=text(1180,693,'each has dₐ coordinates',31,color=MUTED,anchor='middle')
    b+=mathml(110,690,650,100,'<mrow><msub><mi>A</mi><mi>t</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>H</mi><mo>×</mo><msub><mi>d</mi><mi>a</mi></msub></mrow></msup></mrow>',48)
    add('Predict a sequence of commands in one call',b,
        'H is the number of future commands and d_a is the dimension of one command. The illustrated output has six commands and fourteen coordinates each; those numbers are pedagogical and use the ALOHA action dimension. Green dots mark gripper coordinates to reinforce that coordinates can have different units and meanings. A chunk is a structured command array, not six model conversations and not six predicted physical states. The observation can include one or more cameras, proprioception, and sometimes a history or language instruction. Predicting a chunk does not yet determine how much of it will be executed before observing again.',[ACT,DP],'Action chunking')

    # 18. The prefix interaction has room to breathe.
    scenes=[]
    for stage in range(4):
        s=text(75,215,'Predict H = 6. Execute h = 2. Observe again.',36,700,BLUE)
        for j in range(10):s+=text(420+j*108+48,292,str(j),26,color=MUTED,anchor='middle')
        s+=text(75,292,'Execution step',29,700)
        for row,origin in enumerate([0,2,4]):
            yy=330+row*120;s+=text(75,yy+48,f'From o{str(origin).translate(str.maketrans("024","₀₂₄"))}',31,700,INK if stage>=row+1 or row==0 else LINE)
            if row==0 or stage>=row+1:
                done=2 if (row==0 and stage>=1) or (row==1 and stage>=3) else 0
                s+=chunk(420+origin*108,yy,6,origin,origin,done,stage>row+1,108)
        s+=text(75,720,['Predict from o₀','Execute two commands','Replace the unused suffix','Repeat with observation o₄'][stage],32,700,BLUE)
        scenes.append(s)
    add('Execute a prefix, then update the plan',steps('visual-prefix',scenes,y=750),
        'The first frame predicts six commands. The second executes only two, which turn green. A new observation at step two supports a fresh chunk; unused predictions from the first chunk are crossed out. At step four the process repeats. Subscripts identify execution steps and parenthesized superscripts identify prediction/observation steps, so two proposals for the same execution step can have different values. The drawing is logical timing and omits inference latency. H is the prediction horizon and h the executed prefix, with h at most H. This receding-horizon pattern is used in Diffusion Policy. Larger h can reduce model calls but delay visual correction. Rapid joint feedback may remain active inside the prefix.',[DP],'Action chunking')

    # 19. ACT temporal ensembling is a distinct execution policy.
    b=text(75,210,'Different observations can predict the same future execution step.',32)
    for row,origin in enumerate([0,1,2]):
        yy=300+row*120;b+=text(75,yy+48,f'From o{str(origin).translate(str.maketrans("012","₀₁₂"))}',31,700)
        b+=chunk(420+origin*140,yy,4,origin,origin,cell=140)
    b+=rect(697,282,136,352,'none',TEAL,attrs='stroke-width="4"')
    b+=arrow(765,638,765,714,TEAL)+text(955,685,'combine predictions',31,700,TEAL)
    b+=text(765,770,'one command for step 2',34,700,TEAL,'middle')
    add('Overlapping chunks can be combined',b,
        'This is ACT’s temporal-ensembling idea, distinct from simply executing a prefix and discarding the remainder. Querying at successive observation steps creates overlapping chunks. At execution step two, the highlighted column contains predictions made from observations zero, one, and two. ACT combines predictions aligned to that same execution time using a weighted average. It does not average adjacent execution times as an ordinary temporal smoothing filter would. This adds inference-time computation and controls how quickly new observations influence the command. The schematic uses short chunks for readability, not the original reported horizon. It makes no claim that model inference is instantaneous.',[ACT],'Action chunking')

    # 20. Original ACT, with language deliberately absent.
    for j in range(4):b0=camera(95+j*32,300-j*22,.95) if j==0 else b0+camera(95+j*32,300-j*22,.95)
    b=b0+text(230,455,'4 RGB views',32,700,BLUE,'middle')
    b+=text(230,575,'joint positions qₜ',31,700,BLUE,'middle')
    b+=arrow(390,350,630,400)+arrow(390,565,630,480)
    b+=rect(630,330,370,240,PALE,BLUE)+text(815,425,'ACT',47,700,BLUE,'middle')+text(815,490,'Transformer policy',30,anchor='middle')
    b+=arrow(1000,450,1130,450)
    for j,lab in enumerate(['aₜ','aₜ₊₁','aₜ₊₂']):b+=label_box(1130+j*125,409,115,76,lab,PALE,31)
    b+=text(1310,565,'joint-target chunk',31,700,BLUE,'middle')
    b+=text(815,240,'latent z',31,700,TEAL,'middle')+arrow(815,260,815,327,TEAL)
    b+=text(75,735,'Training: conditional VAE',31,700,BLUE)+text(920,735,'Inference: z = 0',31,700,TEAL)
    b+=text(75,800,'Original ACT conditions on images and robot measurements.',28,color=MUTED)
    add('ACT: Action Chunking with Transformers',b,
        'Original Action Chunking with Transformers learns from ALOHA demonstrations. In the real system its observation includes four RGB camera views and fourteen follower joint/gripper measurements. The output is a chunk of desired joint positions. ACT trains a conditional VAE: an encoder sees demonstrated actions and joint observations to infer a latent style variable, while the transformer decoder predicts the action sequence. Reconstruction and KL terms train the model. At inference the encoder is removed and z is set to the prior mean, zero. The displayed architecture is a simplified inference path; it deliberately does not draw the full training encoder. Original ACT has no language input. The simulated example earlier can use fewer camera views, so do not infer its schema from this real-system architecture.',[ACT],'Action policies')

    # 21. The generation target is explicit before opening its denoiser.
    b=camera(110,300,1.35)+text(180,460,'observations',31,700,BLUE,'middle')
    b+=arrow(315,355,550,355)
    b+=circle(765,355,145,PALE,BLUE)+lines(765,344,['Diffusion','Policy'],35,gap=46,weight=700,color=BLUE,anchor='middle')
    b+=arrow(910,355,1090,355)
    for j,lab in enumerate(['aₜ','aₜ₊₁','aₜ₊₂']):b+=label_box(1090+j*140,315,130,76,lab,PALE,31)
    b+=text(1295,485,'action chunk',35,700,BLUE,'middle')
    b+=path('M 1300 540 L 1300 675 L 180 675 L 180 510',TEAL,3)+arrow(180,560,180,510,TEAL)
    b+=text(750,653,'execute a prefix → observe again',34,700,TEAL,'middle')
    b+=text(75,800,'The output is commands, not a forecast of future images or measured states.',29,color=MUTED)
    add('Diffusion Policy generates an action chunk',b,
        'Diffusion Policy models a conditional distribution of action sequences given recent observations. Observations are conditioning inputs, not part of a jointly generated future image/state sequence. After predicting a horizon, the system executes a prefix and replans from new observations. The horizon sizes in this diagram are illustrative. The method can use visual observations or suitable low-dimensional observations, and its action coordinates depend on the robot interface. It need not contain an explicit world model. The next page opens the repeated denoising computation inside the circle.',[DP],'Action policies')

    # 22. The x-axis is command offset; panels are generative computation.
    b=text(75,215,'The observation stays fixed during one generation.',32,color=BLUE)
    b+=text(75,252,'One action coordinate · schematic',25,color=MUTED)
    xs=[115,650,1185]
    values=[[40,240,70,220,80,255,120],[220,185,205,115,155,85,65],[245,220,180,130,90,65,55]]
    for xx,vals,lab in zip(xs,values,['Noise','Refine','Command sequence']):
        b+=text(xx+150,295,lab,34,700,BLUE,'middle')+line(xx,640,xx+300,640,MUTED,2)+line(xx,330,xx,640,MUTED,2)
        pts=[(xx+j*48,350+v) for j,v in enumerate(vals)]
        b+=path('M '+' L '.join(f'{x} {y}' for x,y in pts),BLUE,5)
        for x,y in pts:b+=circle(x,y,5,BLUE,BLUE)
        b+=text(xx+150,691,'future command index',27,color=MUTED,anchor='middle')
    b+=arrow(455,490,600,490)+arrow(990,490,1135,490)
    b+=caption('Denoising iterations are computation, not robot execution time.',790,33)
    add('Refine a candidate command sequence',b,
        'Each panel shows one coordinate across a future action chunk, not the whole H-by-d_a array. The panels are successive generative refinements conditioned on the same observation. Their horizontal axes are future command offsets; moving from panel to panel is a model-computation iteration, not the robot advancing through those commands. The plotted values are authored illustrations, not samples from a trained policy. During diffusion training, demonstrated chunks are corrupted at sampled noise levels and a conditioned network learns to predict the added noise. During sampling, a schedule of learned denoising updates transforms noise toward an action sequence. Exact samplers may include stochastic terms and differ in parameterization. Flow matching, introduced later, uses a related but distinct vector-field objective.',[DP,CS224R],'Action generation')

    # 23. Servo feedback is a physical execution layer.
    b=text(110,362,'target q*',36,700,BLUE)+arrow(290,350,455,350)
    b+=circle(500,350,44,'white',BLUE)+text(500,362,'−',43,700,BLUE,'middle')
    b+=arrow(545,350,690,350)+text(617,315,'q* − q',29,color=BLUE,anchor='middle')
    b+=rect(690,290,290,120,PALE,TEAL)+text(835,362,'Controller',35,700,TEAL,'middle')
    b+=arrow(980,350,1130,350,TEAL)+text(1055,305,'motor effort',27,color=TEAL,anchor='middle')
    b+=arm(1150,260,.78)
    b+=path('M 1390 530 L 1390 655 L 500 655 L 500 395',TEAL,3)+arrow(500,455,500,395,TEAL)
    b+=text(930,633,'measured joints q',34,700,TEAL,'middle')
    b+=caption('A visual policy can pause while joint feedback continues.',795,34)
    add('A generated target still needs feedback control',b,
        'The diagram isolates a joint-position servo: compare a desired configuration q* with the measured q, use the error in a control law, and apply motor effort. The minus sign represents subtracting measured position from desired position; real controllers can also use velocity, integral terms, feedforward dynamics, torque limits, and other signals. We do not claim every robot has the same PID implementation. This lower-level feedback can remain active while a learned policy executes an open-loop visual prefix or computes its next chunk. Physical motion depends on both control and dynamics, including contact and disturbance, so generated command arrays are not guaranteed outcomes. A tool-pose command may additionally require IK or another Cartesian controller.',[CONTROL,ACT],'Execution')

    # 24. A concise interface check prepares the VLM/VLA section.
    b=label_box(100,270,360,120,'Policy output',size=38)+arrow(460,330,640,330)
    b+=text(1020,345,'[ 0.2, −0.1, 0.8, … ]',51,700,BLUE,'middle')
    b+=line(200,460,1400,460,LINE,2)
    checks=[('What moves?','joints or tool'),('In what coordinates?','units + frame'),('When is it used?','timing + controller')]
    for i,(a,c) in enumerate(checks):
        xx=100+i*505;b+=text(xx,550,a,33,700,BLUE)+text(xx,613,c,30)
    b+=caption('An action interface connects statistical predictions to physical behavior.',785,34)
    add('What would you need to execute this array?',b,
        'Use this as a short audience check before adding vision-language models. An array dimension is insufficient. Ask for coordinate meanings and ordering, units, reference frame, absolute versus relative convention, gripper semantics, intended execution times, limits, and the controller or skill executor consuming it. The numbers are invented and do not correspond to an actual robot command. A VLA can learn the mapping into this interface, while an LLM might instead select a skill or produce an executable program. None of those labels eliminate the need to define sensing, feedback, and the action contract. The lesson now turns to what pretrained visual and language representations add.',[MIT_PICK,CS224R],'Transition to VLAs')
    assert len(out)==24
    return out
