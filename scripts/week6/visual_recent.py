"""Fourteen visual teaching slides: language interfaces and recent evidence.

Only plots explicitly labeled as reported results contain measured values.
Diagrams and the final exercise are authored teaching illustrations.
"""
from .common import *

SAYCAN = ('SayCan · 2022', 'https://say-can.github.io/')
CAP = ('Code as Policies · 2022', 'https://code-as-policies.github.io/')
VOX = ('VoxPoser · 2023', 'https://voxposer.github.io/')
UNEXPECTED = ('RoboDojo study §3–4', 'https://arxiv.org/html/2609.24170v1')
ROBO_RELEASE = ('Released run notes', 'https://github.com/RoboProbe/RoboProbe/blob/3c0bc68383a75bf3fa2d7bfc5ab1d1241c7e171e/results/l3_inspect_eef_official_2100/README.md')
ASTRA = ('Astranomy §2–4', 'https://research.nvidia.com/labs/srl/projects/astranomy/index.html')
HYBRID = ('Astra policy study §3', 'https://arxiv.org/html/2609.38537v1')
EFE = ('Embodied First Exam protocol', 'https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md')
SECTION = 'Language models and robot control'


def box(x, y, w, h, title, sub=None, fill=PALE, color=BLUE, size=31):
    b = rect(x, y, w, h, fill, LINE)
    if sub:
        return b + text(x+w/2,y+h*.43,title,size,700,color,anchor='middle') + text(x+w/2,y+h*.75,sub,26,color=MUTED,anchor='middle')
    return b + text(x+w/2,y+h/2+size*.34,title,size,700,color,anchor='middle')


def arm(x, y, scale=1):
    b = rect(-40,0,80,18,PALE,BLUE)
    b += path('M 0 0 L 20 -105 L 135 -165 L 230 -105',BLUE,13)
    for xx,yy in [(0,0),(20,-105),(135,-165),(230,-105)]:
        b += circle(xx,yy,13,'white',BLUE)
    b += path('M 230 -105 V -75 M 210 -75 H 250 M 210 -75 V -45 M 250 -75 V -45',BLUE,5)
    return f'<g transform="translate({x} {y}) scale({scale})">{b}</g>'


def cup(x, y, color=TEAL):
    return path(f'M {x} {y} l 6 47 h 44 l 6 -47 Z',color,3,'#e4f0ec') + path(f'M {x+56} {y+10} C {x+84} {y+1} {x+83} {y+40} {x+52} {y+36}',color,3)


def chips(x, y, labels, widths, gap=15):
    b = ''
    for value,w in zip(labels,widths):
        b += box(x,y,w,66,value,size=27)
        x += w+gap
    return b


def get_slides():
    out = []

    # 1. Separate language selection from the learned motor skill.
    b = text(75,211,'Instruction: help clean a spill',34,700)
    b += box(75,295,450,115,'Language model','Which skill helps?')
    b += box(75,515,450,115,'Learned value function','Which skill can succeed here?')
    b += path('M 525 352 H 630 V 456 H 690',BLUE,3)
    b += path('M 525 572 H 630 V 456',BLUE,3)
    b += arrow(675,456,710,456)
    b += circle(770,456,52,PALE,BLUE) + text(770,475,'×',58,700,BLUE,anchor='middle')
    b += arrow(822,456,915,456)
    b += box(930,400,290,112,'Select a skill','e.g., pick up sponge')
    b += arrow(1220,456,1300,456)
    b += arm(1338,505,.65)
    b += text(1380,574,'Learned skill',29,700,BLUE,anchor='middle')
    b += text(1380,614,'executes motion',27,anchor='middle')
    b += text(75,742,'The language model chooses from an existing robot skill repertoire.',32,color=BLUE)
    out.append(slide('SayCan: choose a useful, feasible skill',b,
        'SayCan combines language-model usefulness with learned value-function feasibility and selects from a predefined repertoire of language-conditioned robot skills. The diagram is conceptual, without invented numerical scores. The selected skill has learned low-level control. Scene grounding enters through the value functions; do not imply the original language prompt receives a rich visual feedback stream. The spill instruction and candidate skill paraphrase the paper’s motivating example. The original study includes physical mobile manipulation, unlike the simulation results later in this section.',[SAYCAN],SECTION))

    # 2. A primary robot view alongside large, explicitly illustrative code.
    b = text(75,211,'The output can be a program with perception calls and feedback loops.',32)
    b += image(75,298,680,285,'recent-cap-block-bowl.png','Code as Policies Figure 2a: physical tabletop blocks and bowls before and after manipulation')
    b += text(75,627,'Physical tabletop examples',28,700,BLUE)
    b += text(75,666,'Project figure crop: Code as Policies',24,color=MUTED)
    b += text(845,282,'Illustrative generated program',29,700,BLUE)
    b += code(845,313,680,335,'while objects_remain():\n    scene = observe()\n    target = choose_target(scene)\n    move_gripper(target)\n    update_from_feedback()',30)
    b += arrow(1100,662,1100,706)
    b += box(845,718,680,70,'Perception APIs + control APIs',size=30)
    out.append(slide('Code as Policies: let code close the loop',b,
        'The code shown here is authored pseudocode, not a verbatim model output or a promise that a specific API exists. Code as Policies uses language-model-generated programs to combine perception and control APIs; these programs can contain loops, geometry computations, and feedback. Running a generated program is distinct from querying an LLM for every servo command. The physical tabletop image is a crop of Figure 2a from the paper, rendered from PDF page 2. It illustrates the original system’s setting and is not a test of the pseudocode on this slide. The full project provides demonstrations and prompts.',[CAP],SECTION))

    # 3. Primary maps and robot path, with the dense code panel omitted.
    b = text(75,211,'Language can specify where motion should go—and what it should avoid.',32)
    b += box(75,300,380,98,'Instruction + RGB-D',size=30)
    b += arrow(265,398,265,439)
    b += box(75,455,380,98,'Code constructs maps',size=29)
    b += arrow(265,553,265,594)
    b += box(75,610,380,98,'Planner + controller',size=29)
    b += arrow(455,504,536,504)
    b += image(555,274,970,486,'recent-voxposer-maps-path.jpg','VoxPoser method figure crop showing 3D affordance maps, constraint maps, and planned motion around a vase')
    b += text(555,796,'VoxPoser project illustration · maps → planned robot motion',25,color=MUTED)
    out.append(slide('VoxPoser: turn language into spatial objectives',b,
        'VoxPoser uses language models and perception tools to generate code that composes three-dimensional value maps. Affordance maps encourage desired interactions and constraint maps discourage forbidden regions; motion planning and low-level control execute the resulting objectives. Cached code can be re-evaluated with new observations. The original method figure is cropped to the map and planning panels, omitting its dense code panel. This slide does not claim that the LLM directly outputs motor torques, or that all perception and control components were trained from scratch or without robot experience. RGB-D denotes color plus depth.',[VOX],SECTION))

    # 4. Define the numeric action interface before presenting its benchmark.
    b = text(75,211,'A target specifies the gripper pose; an execution stack turns it into motion.',32)
    b += box(75,332,300,110,'Observation','Images + robot state')
    b += arrow(375,387,451,387)
    b += box(466,332,285,110,'General model','Instruction + history')
    b += arrow(751,387,827,387)
    b += box(842,332,310,110,'End-effector target','Pose + gripper')
    b += arrow(1152,387,1228,387)
    b += box(1243,332,282,110,'IK + planner','Joint path')
    b += path('M 1384 442 V 575 H 225 V 470',MUTED,3)
    b += arrow(225,470,225,443,MUTED)
    b += text(800,617,'Controller executes → sensors observe again',31,700,BLUE,anchor='middle')
    b += text(75,739,'End effector = the robot’s working tip. IK = inverse kinematics.',29,color=MUTED)
    out.append(slide('A numeric target still needs a robot controller',b,
        'This conceptual loop describes direct end-effector control in the recent RoboDojo study. The model requests Cartesian gripper pose and opening; the surrounding stack validates actions, transforms the target, solves inverse kinematics and plans a joint trajectory. The physical controller or simulator then executes the joint path and returns new sensor information. A target is not a torque. Inverse kinematics maps an end-effector pose to robot joint configurations. The reported RoboDojo interface provides move_eef and give_up, not a semantic pick-and-place skill library. No learned motor policy supplies the direct method’s motion.',[UNEXPECTED],SECTION))

    # 5. The actual sensor views are the central visual.
    b = text(75,211,'What did the general model actually see?',34,700)
    b += image(75,270,1450,360,'recent-robodojo-three-views.png','Three simulated camera views from the RoboDojo study’s make-kong grounding example')
    b += text(75,670,'Simulation: one task example, three RGB views',27,color=MUTED)
    b += chips(75,717,['Robot state','Task instruction','Recent images','Text history'],[280,335,330,340],25)
    out.append(slide('RoboDojo: images are part of the policy interface',b,
        'This image is the paper’s three-view grounding illustration for a make-kong task, not a physical robot photo or a representative sample of all benchmark tasks. The undated gpt-6-astra alias is frozen and used at medium reasoning effort. Inputs include three RGB cameras, proprioception, wiki-derived goal and scoring descriptions, full text history, and the latest two image turns. The direct policy does not receive depth, privileged object poses, or a reward oracle. These observation and instruction choices are part of the measured system and can differ from published learned-policy baselines. See §§3.2–3.3 of the primary paper.',[UNEXPECTED],SECTION))

    # 6. A chart with a meaningful zero and the exact native metric.
    b = text(75,211,'GPT-6 Astra · 42 simulation tasks × 50 episodes · one seed',30,700)
    b += text(75,275,'Full-task success by category',30,700,BLUE)
    values = [('Generalization',30.50),('Precision',4.00),('Long horizon',8.25),('Memory',38.67),('Open language',31.00)]
    x,w = 375,795
    for tick in range(0,51,10):
        xx=x+w*tick/50
        b += line(xx,309,xx,690,'#d9e2e8') + text(xx,727,f'{tick}%',24,color=MUTED,anchor='middle')
    for i,(label,v) in enumerate(values):
        y=319+i*77
        b += text(75,y+34,label,29)
        b += rect(x,y,w*v/50,46,BLUE,'none')
        b += text(x+w*v/50+15,y+34,f'{v:.2f}%',28,700,BLUE)
    b += text(1290,405,'22.48%',54,700,BLUE,anchor='middle')
    b += lines(1290,456,['Macro success','Mean of categories'],27,gap=42,anchor='middle')
    b += text(75,796,'Paper-reported campaign; category names do not isolate cognitive mechanisms.',26,color=MUTED)
    out.append(slide('RoboDojo performance varies sharply by task type',b,
        'These are Table 1 native full-task success rates: 30.50, 4.00, 8.25, 38.67 and 31.00 percent, with five-category macro average 22.48 percent. The separate score metric is 28.97 and must not be relabeled as success rate. The paper evaluates 42 tasks with 50 trials each and one seed. Read the graph as a descriptive category pattern, not proof of a causal reasoning or memory mechanism. The pinned release README discloses two targeted task-cell replacements and a 170-call budget; the paper states 100 by default. Those campaign details remain unreconciled, so this is explicitly a paper-reported result, not an independently reproduced leaderboard rank. Input access differs from policy baselines, which are intentionally omitted here.',[UNEXPECTED,ROBO_RELEASE],SECTION))

    # 7. Each system's output and execution machinery are visible.
    b = text(75,211,'Astranomy evaluates three complete control systems on the same task suite.',31)
    rows = [
        (285,'Direct','Model → target','IK / controller','Short observation window'),
        (455,'Agentic','Persistent agent','Robot tools / controller','History + code + tool calls'),
        (625,'Orchestrated','Model reviews policy','Policy / controller','Cosmos3-Nano-Policy proposals'),
    ]
    for y,name,decision,executor,detail in rows:
        b += text(75,y+54,name,33,700,BLUE)
        b += box(365,y,490,100,decision,size=31)
        b += arrow(855,y+50,950,y+50)
        b += box(965,y,560,100,executor,fill='white',size=30)
        b += text(365,y+141,detail,26,color=MUTED)
    out.append(slide('Astranomy: the harness changes the control system',b,
        'Astranomy’s Direct EEF system chooses numeric targets using a short observation window. The agentic Codex system can maintain persistent history and use code and tools. The orchestration system supervises Cosmos3-Nano-Policy proposals. These are systems with different histories and harnesses, not three controlled doses of computation applied to an otherwise identical policy. All use the report’s 120-task simulation suite. A controller remains below each high-level interface. The study’s public report is retrieved and supports these claims; the linked implementation endpoint was not publicly retrievable in the audit. Model labels are those used in the report, without assuming an undisclosed dated checkpoint.',[ASTRA],SECTION))

    # 8. No cross-paper bars or unsupported compute attribution.
    b = text(75,211,'RoboLab-120 simulation · 120 tasks × 5 episodes · medium effort',30,700)
    cx,cy,cw,ch=220,702,1180,377
    for tick in [0,25,50,75,100]:
        yy=cy-ch*tick/100
        b += line(cx,yy,cx+cw,yy,'#d9e2e8') + text(cx-20,yy+8,f'{tick}%',25,color=MUTED,anchor='end')
    for xx,value,name,color in [(385,66.3,'Direct',BLUE),(795,74.8,'Agentic',TEAL),(1205,66.5,'Orchestrated',BLUE)]:
        hh=ch*value/100
        b += rect(xx-105,cy-hh,210,hh,color,'none')
        b += text(xx,cy-hh-23,f'{value:.1f}%',43,700,color,anchor='middle')
        b += text(xx,cy+47,name,30,700,anchor='middle')
    b += text(75,279,'Full-task success',28,700,BLUE)
    b += text(75,807,'Physics pauses during inference. History and harness differ across systems.',27,color=MUTED)
    out.append(slide('Astranomy reports system-level success rates',b,
        'The report evaluates Direct EEF at 66.3%, the persistent agent at 74.8%, and policy orchestration at 66.5% full-task success, over 120 tasks with five episodes each, seeds 0–4. It uses three camera views at 384×216 and medium effort. Physics is paused during inference, with 15 Hz action execution and task-specific action-step limits; there is no wall-time or retry cap. These system-level rates are not evidence that computation alone caused a gain. Five episodes per task also limit precision for small differences. Do not compare this suite’s rates directly with the separate RoboDojo 42-task campaign or the selected ten-task hybrid experiment. No error bars are invented.',[ASTRA],SECTION))

    # 9. A three-step timing demonstration; all segment widths are schematic.
    b = text(75,211,'Two clocks can tell different stories.',34,700)
    frames=[]
    for stage in range(3):
        s=text(75,326,'Wall clock',30,700,BLUE)+text(75,516,'Simulation clock',30,700,BLUE)
        s+=box(380,275,240,90,'Observe',fill=PALE if stage==0 else 'white')
        s+=box(645,275,560,90,'Model computes',fill=PALE if stage==1 else 'white')
        s+=box(1230,275,295,90,'Execute',fill='#e4f0ec' if stage==2 else 'white')
        s+=arrow(380,405,1525,405,MUTED)
        s+=line(380,501,1205,501,MUTED,4,'10 8')
        s+=text(790,557,'Physics stays frozen',31,700,BLUE,anchor='middle')
        for i in range(7): s+=line(1245+i*42,476,1245+i*42,526,TEAL,3)
        s+=text(1375,567,'15 Hz playback',29,700,TEAL,anchor='middle')
        xx=[500,930,1375][stage]
        s+=circle(xx,405,12,BLUE,BLUE)
        captions=['Capture the state at simulation time t.','Wall time passes; the simulated scene cannot move.','Execute targets; simulation time advances again.']
        s+=text(75,650,captions[stage],32,700,BLUE)
        frames.append(s)
    b += steps('recent-physics-clock',frames,y=740)
    b += text(75,704,'Direct mean: 18.56 s/call in successful episodes, including network delay.',27,color=MUTED)
    out.append(slide('A paused simulator does not test real-time reactions',b,
        'Step through this schematic timeline. Astranomy pauses physics while a model call is in flight, so an object cannot drift during that interval. Once a command arrives, action playback runs at 15 Hz. The reported 18.56-second mean Direct EEF call latency is computed over successful episodes only and includes network delay; it is not a control-loop period or an unconditional latency estimate. Timeline widths are schematic and do not reconstruct an episode. A physical experiment, or a simulation where physics continues while the model thinks, tests a different timing regime. Embodied First Exam explicitly distinguishes paused standard mode from standard-realtime, though its other protocol defaults also need to be matched.',[ASTRA,EFE],SECTION))

    # 10. Exclusive branch semantics are the entire slide.
    b = text(75,211,'A learned policy supplies candidate motion; the general model can revise it.',31)
    b += box(75,380,360,128,'Task-finetuned π0.5','50-step joint proposal',size=31)
    b += arrow(435,444,535,444)
    b += box(550,380,330,128,'Astra reviews','Images + proposal')
    b += path('M 880 444 H 960 V 340 H 1010',TEAL,3)
    b += arrow(1000,340,1040,340,TEAL)
    b += path('M 960 444 V 585 H 1010',WARM,3)
    b += arrow(1000,585,1040,585,WARM)
    b += box(1055,280,470,120,'Accept policy prefix','Execute 1–15 joint steps',fill='#e4f0ec',color=TEAL)
    b += text(1275,474,'OR',31,700,MUTED,anchor='middle')
    b += box(1055,525,470,120,'Replace with correction','Execute 1–5 target steps',fill='#f7efe7',color=WARM)
    b += text(75,755,'One branch executes, then the system observes again.',32,700,BLUE)
    out.append(slide('Hybrid control: accept a policy prefix or correct it',b,
        'In Systematically Exploring, released RoboDojo-task-finetuned π0.5 weights propose a 50-step joint-action sequence. Astra observes the images and proposal, then either accepts the first 1–15 policy steps or substitutes a 1–5-step end-effector correction. These are mutually exclusive branches, not two commands executed consecutively. Both pass through the robot execution stack and receive observations after execution. The direct baseline also outputs 1–5 end-effector steps. The explicit policy-training qualification matters: frozen general-model weights do not make the entire hybrid system free of robot-specific training. Here a target step refers to the paper’s direct action interface, not an unmediated torque.',[HYBRID],SECTION))

    # 11. Aggregated completions; no fabricated instance-level pairing.
    b = text(75,211,'10 selected RoboDojo tasks × 5 paired initializations per method',30,700)
    for start,n,label,color in [(130,13,'Direct',BLUE),(870,24,'Hybrid',TEAL)]:
        b += text(start,299,label,36,700,color)
        b += text(start+492,302,f'{n} / 50',46,700,color,anchor='end')
        for i in range(50):
            xx=start+(i%10)*49; yy=348+(i//10)*60
            b += rect(xx,yy,34,42,color if i<n else '#edf1f4','none')
        b += text(start,697,f'{n/50:.0%} full-task success',32,700,color)
    b += text(75,779,'Same cases and target interface; selected tasks emphasize low π0.5 success.',27,color=MUTED)
    out.append(slide('Hybrid control helps on this selected task subset',b,
        'The paper reports 13 of 50 full-task completions for Direct (26%) and 24 of 50 for Hybrid (48%). Each group of rectangles is an aggregate count, with filled cells placed together for legibility; rectangle positions do not depict matched per-instance outcomes. The study uses ten selected tasks and five paired initializations, with task instances, seeds, descriptions, reasoning settings, and end-effector interface held fixed. The selection is stratified by published π0.5 success and emphasizes low-success tasks: six from the lowest stratum, two from the next, one each from the remaining two. The gain is not uniform across tasks; some classification tasks favor Direct. The standalone π0.5 reference is a reweighted published number, not a locally rerun paired baseline, so it is omitted.',[HYBRID],SECTION))

    # 12. A controlled comparison is shown as a shared physical problem.
    b = text(75,211,'Compare the decision systems on a shared physical problem.',33)
    b += box(75,410,290,115,'Same test cases','Held-out layouts')
    b += path('M 365 467 H 430 V 344 H 495',BLUE,3)
    b += path('M 430 467 V 584 H 495',BLUE,3)
    b += arrow(480,344,520,344)+arrow(480,584,520,584)
    b += box(535,292,385,105,'System A',size=34)
    b += box(535,532,385,105,'System B',size=34)
    b += path('M 920 344 H 990 V 467 H 1050',BLUE,3)
    b += path('M 920 584 H 990 V 467',BLUE,3)
    b += arrow(1035,467,1075,467)
    b += box(1090,410,435,115,'Common action interface','Controller + environment')
    b += chips(75,726,['Allowed information','Adaptation + resets','Physics clock','Time + completion'],[360,345,320,350],25)
    out.append(slide('The evaluation protocol is part of the result',b,
        'This is an authored controlled-experiment diagram, not a description shared by all cited benchmarks. Hold test cases and the action interface fixed when asking about a decision component. Explicitly declare sensor access, task descriptions, tools, memory, prior trials, code generation, resets, and adaptation. Freeze development before held-out evaluation. Report the physics clock: paused during inference or advancing while thinking. Measure elapsed time alongside native completion. A separate engineering question may legitimately allow each complete system its own interface, but that is a best-system comparison and cannot isolate the decision model alone. Embodied First Exam illustrates why mode names are insufficient: standard and standard-realtime differ in clock and can have different reset defaults.',[EFE,ASTRA],SECTION))

    # 13. Original question, not an attributed interview or measured experiment.
    b = text(75,211,'Authored exercise: put a cup inside a drawer in an unseen layout.',31)
    b += arm(150,560,1.1)+cup(370,455)
    b += rect(540,425,280,140,'white',BLUE)+rect(560,445,240,99,PALE,BLUE)
    b += line(625,493,735,493,BLUE,5)
    b += rect(630,395,280,140,'none',WARM,'stroke-dasharray="10 8"')
    b += arrow(660,355,855,355,WARM)
    b += text(545,311,'Drawer shifts during inference',28,700,WARM)
    b += line(75,590,930,590)
    b += box(75,657,390,104,'System A','VLA action chunk')
    b += box(520,657,410,104,'System B','General model + planner')
    b += text(1035,371,'What stays fixed?',32,700,BLUE)
    b += text(1035,501,'What can each access?',30,700,BLUE)
    b += text(1035,631,'What do you measure?',30,700,BLUE)
    out.append(slide('Question: how would you compare these systems?',b,
        'Pause here for audience discussion. This is an authored research exercise and diagram, not a real trial or a reported interview question. The drawer moves during the model’s inference delay. Ask learners to identify information access, timing, task adaptation, and outcome variables before choosing a winner. A VLA action chunk and a general-model-plus-planner stack can differ in both learned knowledge and action interface. The question should elicit the distinction between a controlled component comparison and an engineering comparison of complete systems. Stale observations and commands matter when physics continues during computation.',[], 'Research exercise'))

    # 14. The curve is conspicuously illustrative, with no fabricated data.
    b = text(75,211,'Match cases and disturbances; compare across elapsed-time budgets.',32)
    left=[('Fix the cases','Layouts + disturbance schedule'),('Declare access','Sensors, tools, training, resets'),('Measure outcomes','Completion, time, failures')]
    for i,(heading,sub) in enumerate(left):
        y=335+i*150
        b += circle(105,y-11,26,PALE,LINE)+text(105,y,i+1,29,700,BLUE,anchor='middle')
        b += text(155,y,heading,32,700,BLUE)+text(155,y+51,sub,27)
    b += text(905,300,'Illustrative curves · no measured values',27,color=MUTED)
    b += arrow(900,680,1515,680,MUTED)+arrow(900,680,900,345,MUTED)
    b += text(910,338,'Completion rate',28,700,BLUE)
    b += text(1205,738,'Elapsed-time budget',29,anchor='middle')
    b += path('M 925 653 C 975 526 1025 480 1120 463 C 1240 445 1380 442 1485 440',TEAL,5)
    b += path('M 925 665 C 1050 655 1150 560 1250 464 C 1350 377 1435 364 1485 360',BLUE,5)
    b += text(1390,489,'System A',27,700,TEAL,anchor='middle')
    b += text(1390,336,'System B',27,700,BLUE,anchor='middle')
    b += text(75,806,'Repeat matched trials and report uncertainty. Do not infer a universal winner.',27,color=MUTED)
    out.append(slide('Solution: evaluate the operating curve',b,
        'The curves are authored and schematic, with no numeric predictions or measured points. Compare repeated matched cases over meaningful resource budgets and report uncertainty at the task or scene level. Elapsed time includes model computation, tool use, and execution. State and match whether physics advances during computation, and use the same disturbance schedule. Declare training, demonstrations, prior exploration, reusable code, and resets even when equalizing them is impossible. A shared action interface isolates a decision-component difference. A best-system experiment instead lets each method use its preferred executor and must expose those design differences. Connect the discussion back to observation, action, feedback, and data from the beginning of the lecture.',[EFE], 'Research exercise'))

    assert len(out) == 14
    return out
