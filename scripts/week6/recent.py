"""Slides 23–30: language interfaces, recent evidence, and an experiment.

All diagrams are authored teaching illustrations. Reported values are explicitly
scoped to their source protocol. No robot experiment was run for this deck.
"""
from .common import *

SAYCAN = ('SayCan', 'https://say-can.github.io/')
CAP = ('Code as Policies', 'https://code-as-policies.github.io/')
VOX = ('VoxPoser', 'https://voxposer.github.io/')
UNEXPECTED = ('RoboDojo study §3–4', 'https://arxiv.org/html/2609.24170v1')
ROBO_RELEASE = ('Released run notes', 'https://github.com/RoboProbe/RoboProbe/blob/3c0bc68383a75bf3fa2d7bfc5ab1d1241c7e171e/results/l3_inspect_eef_official_2100/README.md')
ASTRA = ('Astranomy §2–4', 'https://research.nvidia.com/labs/srl/projects/astranomy/index.html')
HYBRID = ('Astra policy study §3', 'https://arxiv.org/html/2609.38537v1')
EFE = ('Embodied First Exam', 'https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md')
SECTION = 'Language models and robot control'


def block(x, y, w, h, title, subtitle=None, fill=PALE, title_size=29):
    out = rect(x, y, w, h, fill)
    if subtitle is None:
        return out + text(x+w/2, y+h/2+10, title, title_size, 700, anchor='middle')
    out += text(x+w/2, y+40, title, title_size, 700, anchor='middle')
    out += text(x+w/2, y+77, subtitle, 25, color=MUTED, anchor='middle')
    return out


def joint_arm(x, y):
    """An illustrative side-view arm, deliberately not a named embodiment."""
    out = rect(x-40, y, 80, 18, PALE, BLUE)
    out += path(f'M {x} {y} L {x+20} {y-105} L {x+135} {y-165} L {x+230} {y-105}', BLUE, 13)
    for xx, yy in [(x,y),(x+20,y-105),(x+135,y-165),(x+230,y-105)]:
        out += circle(xx, yy, 13, 'white', BLUE)
    out += path(f'M {x+230} {y-105} L {x+230} {y-75} M {x+210} {y-75} L {x+250} {y-75} M {x+210} {y-75} L {x+210} {y-45} M {x+250} {y-75} L {x+250} {y-45}', BLUE, 5)
    return out


def get_slides():
    out = []

    # 23. Different language outputs attach to different execution machinery.
    body = text(75, 196, 'The language output determines what the robot stack must supply.', 31, 700)
    rows = [
        (255, 'SayCan', 'Choose a skill', 'Useful × feasible', 'Learned skill policy', 'Skill invocation'),
        (405, 'Code as Policies', 'Write a program', 'Logic + feedback loops', 'Perception / control APIs', 'Executable code'),
        (555, 'VoxPoser', 'Construct 3D maps', 'Affordances + constraints', 'Motion planner + controller', 'Spatial objective'),
    ]
    for y, name, decision, detail, executor, payload in rows:
        body += text(75, y+45, name, 29, 700, color=BLUE)
        body += block(355, y, 350, 100, decision, detail)
        body += arrow(705, y+50, 970, y+50)
        body += text(837, y+28, payload, 24, color=MUTED, anchor='middle')
        body += block(985, y, 540, 100, executor, fill='white')
    body += takeaway('Language-guided robotics predates general models choosing numeric targets.',
                      'Ask what the language model produces, then follow it to physical execution.')
    out.append(slide('Language enters the robot stack at several levels', body,
        'SayCan multiplies language-model usefulness with a learned value function estimating skill feasibility, then selects an existing skill. Code as Policies writes programs that process perception outputs and call control APIs, including reactive loops. VoxPoser generates code that constructs spatial value maps from observations and language; planners use those maps as objectives. Its cached code can be re-evaluated with fresh observations. These rows describe outputs and executors, not mutually exclusive model families. The diagrams simplify the papers rather than reproducing one shared implementation.',
        [SAYCAN, CAP, VOX], SECTION))

    # 24. Interface comparison before any metric.
    body = text(75, 196, 'General models can participate through three different outputs.', 31, 700)
    body += text(75, 244, 'Model input: images + robot measurements + instruction + selected history', 27, color=MUTED)
    specs = [
        (300, 'Numeric target', ['pose + gripper'], 'IK / motion planning', 'Controller'),
        (430, 'Executable program', ['loops + API calls'], 'Perception / control tools', 'Controller'),
        (560, 'Policy review', ['accept or correct'], 'Policy prefix / new target', 'Controller'),
    ]
    for y, title, sub, executor, end in specs:
        body += block(75, y, 365, 98, title, sub[0])
        body += arrow(440, y+49, 540, y+49)
        body += block(555, y, 520, 98, executor, fill='white')
        body += arrow(1075, y+49, 1180, y+49)
        body += block(1195, y, 330, 98, end)
    body += takeaway('A numeric robot target still relies on an execution interface.',
                      'Frozen model weights can coexist with changing context, code, tools, and reusable skills.')
    out.append(slide('What does a general model send to the robot?', body,
        'Read each row as the output of the language-related component followed by its executor. A numeric target requests a configuration or pose rather than a motor torque. A generated program can close feedback loops through API calls. In policy review, learned robot behavior supplies candidate actions and the general model can retain or replace them. Actual systems can combine rows. This taxonomy extends the original control loop and separates action representation from which parameters were trained.',
        [CAP, UNEXPECTED, HYBRID], SECTION))

    # 25. A full-suite result with category variation, without cross-paper ranks.
    body = text(75, 190, 'GPT-6 Astra · RoboDojo simulation · 42 tasks × 50 episodes · one seed', 28, 700)
    body += block(75, 225, 300, 96, 'RGB + robot state', 'Instruction + history')
    body += arrow(375, 273, 445, 273)
    body += block(460, 225, 240, 96, 'Astra', 'Chooses a target')
    body += arrow(700, 273, 780, 273)
    body += block(795, 225, 315, 96, 'End-effector target', 'Pose + gripper')
    body += arrow(1110, 273, 1180, 273)
    body += block(1195, 225, 330, 96, 'IK + planner', 'Joint execution')
    body += text(75, 378, 'Full-task success by category', 29, 700, color=BLUE)
    values = [('Generalization',30.50),('Precision',4.00),('Long horizon',8.25),('Memory',38.67),('Open language',31.00)]
    chart_x, chart_w = 335, 530
    for i, (name, val) in enumerate(values):
        y = 410+i*51
        body += text(75, y+25, name, 27)
        body += rect(chart_x, y, chart_w, 31, '#f3f5f6', 'none')
        body += rect(chart_x, y, chart_w*val/50, 31, BLUE, 'none')
        body += text(chart_x+chart_w*val/50+13, y+25, f'{val:.2f}%', 25, 700)
    body += text(chart_x, 694, '0', 22, color=MUTED)
    body += text(chart_x+chart_w, 694, '50% SR', 22, color=MUTED, anchor='end')
    body += line(975, 375, 975, 685)
    body += text(1030, 440, '22.48%', 66, 700, color=BLUE)
    body += lines(1030, 490, ['Reported macro SR', 'Mean of five categories'], 28, gap=40)
    body += lines(1030, 595, ['Category labels do not isolate', 'reasoning or memory mechanisms.'], 25, gap=36, color=MUTED)
    body += takeaway('Useful task decisions coexist with weak precision and long-horizon completion.',
                      'Paper-reported result. Inputs differ from policy baselines; released reruns and call budgets need reconciliation.')
    out.append(slide('RoboDojo exposes uneven task performance', body,
        'The model sees three RGB views, proprioception, text history and the latest two image turns. Wiki-derived scoring descriptions differ from baseline inputs. A non-learned stack executes targets at 25 Hz. Macro SR averages categories, not pooled trials. The repository discloses replacement of two task cells with targeted reruns and a 170-call budget, while the paper specifies 100 by default. Treat this as the paper-reported campaign rather than an independently verified current leaderboard rank. The category pattern does not identify a causal cognitive mechanism.',
        [UNEXPECTED, ROBO_RELEASE], SECTION))

    # 26. Timing diagram makes paused physical time visible.
    body = text(75, 195, 'Astranomy: RoboLab-120, five episodes per task, medium effort', 29, 700)
    body += text(75, 254, 'Wall-clock time', 28, 700, color=BLUE)
    body += block(355, 220, 810, 80, 'Model chooses the next target', fill=PALE)
    body += block(1180, 220, 345, 80, 'Execute target', fill='#e4f0ec')
    body += text(760, 335, 'Direct EEF: 18.56 s/call, successful episodes', 28, 700, anchor='middle')
    body += text(75, 402, 'Simulation clock', 28, 700, color=BLUE)
    body += line(355, 390, 1165, 390, MUTED, 3, '9 7')
    body += text(760, 435, 'Physics paused during inference', 27, color=MUTED, anchor='middle')
    for i in range(8):
        xx = 1190+i*44
        body += line(xx, 366, xx, 409, TEAL, 3)
    body += text(1350, 445, '15 Hz playback', 28, 700, color=TEAL, anchor='middle')
    body += text(75, 510, 'Same task suite, different policy systems', 29, 700)
    body += table(75, 535, 1450, ['Direct targets','Agentic control','Policy orchestration'],
                  [['66.3% SR','74.8% SR','66.5% SR']], ratios=[1/3]*3, row_h=66, size=29)
    body += text(75, 707, 'History and harness differ across systems. Timeline widths are schematic.', 25, color=MUTED)
    body += takeaway('Fast command playback can coexist with slow decisions.',
                      'No wall-time limit in this study. Reported cost means condition on successful episodes.')
    out.append(slide('A fast playback loop can hide slow decisions', body,
        'The reported 18.56-second mean call latency uses successful episodes only and includes network delay. It is distinct from the 15 Hz execution rate. Physics pauses while inference runs. Astranomy uses task-specific action-step limits, without wall-time or retry limits. Direct retains a short observation window, whereas the agentic system retains persistent history; their success difference cannot be assigned to compute alone. Orchestration supervises Cosmos3-Nano-Policy proposals. Five trials per task give limited precision for small differences. The drawn segment widths are schematic, not a reconstructed episode. Successful-episode cost averages must not be relabeled as unconditional cost per attempt.',
        [ASTRA], SECTION))

    # 27. Exact branch semantics from the paired RoboDojo subset.
    body = text(75, 194, 'Ten selected RoboDojo tasks × five paired starts per method', 29, 700)
    body += block(75, 250, 310, 102, 'Task-finetuned π0.5', '50-step joint proposal')
    body += arrow(385, 301, 485, 301)
    body += block(500, 250, 330, 102, 'Astra reviews', 'Images + proposal')
    body += path('M 830 301 H 885 V 275 H 940', BLUE, 2)
    body += arrow(930, 275, 960, 275)
    body += path('M 885 301 V 413 H 940', WARM, 2)
    body += arrow(930, 413, 960, 413, WARM)
    body += block(975, 225, 530, 100, 'Accept policy prefix', 'Execute 1–15 joint-control steps', fill='#e4f0ec')
    body += block(975, 364, 530, 100, 'Replace with a correction', 'Execute 1–5 end-effector steps', fill='#fbf0e7')
    body += text(930, 342, 'OR', 24, 700, color=MUTED, anchor='middle')
    body += path('M 1505 275 H 1520 V 497 H 665 V 372', MUTED, 2)
    body += line(1505, 414, 1520, 414, MUTED, 2)
    body += arrow(665, 372, 665, 352, MUTED)
    body += text(1100, 530, 'New observation after execution', 25, color=MUTED, anchor='middle')
    body += text(75, 430, 'Both branches use the robot', 27)
    body += text(75, 470, 'execution and feedback stack.', 27)
    body += line(75, 564, 1525, 564)
    body += text(75, 619, 'Direct', 30, 700)
    body += text(290, 625, '13 / 50', 44, 700, color=BLUE)
    body += text(625, 619, 'Hybrid', 30, 700)
    body += text(840, 625, '24 / 50', 44, 700, color=TEAL)
    body += text(1165, 618, 'Full task completions', 26)
    body += text(75, 698, 'The subset emphasizes tasks with low published π0.5 success. Benefits vary by task.', 25, color=MUTED)
    body += takeaway('The learned policy supplies most motion; the general model can make local corrections.',
                      'This is a selected 50-trial simulation study, separate from the full RoboDojo campaign.')
    out.append(slide('Reasoning can revise a learned policy’s proposal', body,
        'The RoboDojo hybrid uses released task-finetuned π0.5 weights. Astra either accepts a prefix or substitutes an end-effector correction; the two action branches are exclusive. Direct and Hybrid share task instances, seeds, descriptions, reasoning setting and EEF interface. Ten tasks were stratified by published policy success, with more drawn from low-success intervals. Hybrid completes 24/50 versus Direct 13/50. Classification tasks can favor Direct. Published standalone-policy references were reweighted, not rerun in the paired experiment. The diagram makes the cooperative mechanism explicit without treating this subset as the full benchmark.',
        [HYBRID], SECTION))

    # 28. Shared cases and executor with auditable protocol fields.
    body = text(75, 197, 'A matched-interface experiment holds the physical problem and access fixed.', 29, 700)
    body += block(75, 332, 310, 125, 'Held-out task cases', 'Same initial layouts')
    body += path('M 385 394 H 430 V 296 H 490', BLUE, 2)
    body += arrow(480, 296, 510, 296)
    body += path('M 430 394 V 485 H 490', BLUE, 2)
    body += arrow(480, 485, 510, 485)
    body += block(525, 245, 365, 102, 'Learned action policy', 'RGB + proprioception')
    body += block(525, 434, 365, 102, 'General model', 'Same allowed information')
    body += path('M 890 296 H 955 V 394 H 1010', BLUE, 2)
    body += path('M 890 485 H 955 V 394', BLUE, 2)
    body += arrow(1000, 394, 1030, 394)
    body += block(1045, 332, 480, 125, 'Common action interface', 'Controller + environment')
    body += text(75, 585, 'Before evaluation', 29, 700, color=BLUE)
    body += text(75, 628, 'Freeze code, task knowledge,', 27)
    body += text(75, 666, 'exploration and reset rules.', 27)
    body += text(610, 585, 'During evaluation', 29, 700, color=BLUE)
    body += text(610, 628, 'Record action budgets, elapsed', 27)
    body += text(610, 666, 'time and native completion.', 27)
    body += text(1125, 585, 'Which physics clock?', 29, 700, color=BLUE)
    body += text(1125, 628, 'Paused during inference', 26)
    body += text(1125, 666, 'or advancing in real time?', 26)
    body += takeaway('The protocol is part of the measured system.',
                      'Best-system comparisons may allow different interfaces, but must expose those differences.')
    out.append(slide('A robot result includes its evaluation protocol', body,
        'This is an authored design for a controlled comparison, not a claim that all cited studies share one interface. Use identical held-out cases and declare what each agent can observe, call, remember and reset. Freeze any development before measuring held-out performance. Embodied First Exam separates standard mode from standard-realtime: in realtime, physics advances during thinking and holds the last target between commands. Its default reset budgets also differ, so the mode label alone is insufficient. A second, useful engineering question compares the best complete systems with their own interfaces; report those access and budget differences rather than calling that a model-only experiment.',
        [EFE, UNEXPECTED, ASTRA], SECTION))

    # 29. Original exercise, visibly separated from reported experiments.
    body = text(75, 194, 'Authored exercise: put a cup in a drawer under an external disturbance.', 29, 700)
    body += text(75, 239, 'Initial layouts are held out. The drawer shifts while the model is thinking.', 27, color=MUTED)
    body += joint_arm(145, 483)
    body += path('M 335 403 L 340 438 L 370 438 L 375 403 Z', TEAL, 3, '#e4f0ec')
    body += path('M 375 412 C 395 407 395 432 373 428', TEAL, 3)
    body += rect(445, 372, 235, 115, 'white', BLUE)
    body += rect(463, 392, 199, 76, PALE, BLUE)
    body += line(520, 429, 605, 429, BLUE, 5)
    body += rect(535, 347, 235, 115, 'none', WARM, 'stroke-dasharray="8 6"')
    body += arrow(598, 326, 717, 326, WARM)
    body += text(520, 299, 'Disturbance during inference', 24, color=WARM)
    body += line(75, 506, 805, 506)
    body += text(75, 556, 'System A', 28, 700, color=BLUE)
    body += block(75, 579, 320, 86, 'VLA action chunk', fill=PALE, title_size=27)
    body += text(460, 556, 'System B', 28, 700, color=BLUE)
    body += block(460, 579, 345, 86, 'LLM target + planner', fill=PALE, title_size=27)
    body += line(865, 280, 865, 696)
    questions = [('1', 'What should stay fixed?'), ('2', 'What can each system access?'), ('3', 'What must you measure?')]
    for i,(num,q) in enumerate(questions):
        yy=337+i*125
        body += circle(933, yy-9, 23, PALE, LINE)
        body += text(933, yy, num, 26, 700, color=BLUE, anchor='middle')
        body += text(980, yy, q, 28, 700)
    body += text(920, 691, 'Discuss information, adaptation, time and outcome.', 24, color=MUTED)
    body += takeaway('Design the comparison before choosing a winner.',
                      'Teaching scenario and diagram are illustrative. No physical trial or interview question is being reported.')
    out.append(slide('Question: what makes this comparison informative?', body,
        'Give the audience a minute before revealing the next page. Ask for a distinction between a matched-interface experiment and an engineering comparison of complete systems. The drawer displacement is an authored disturbance, not a claim about a published task. Useful answers identify pretraining and task-adaptation differences, observation timestamps and calibration, stale actions during inference, reset rules, and native physical completion. An action count alone does not capture the consequences of a real-time disturbance. No fabricated benchmark result or interview attribution accompanies this exercise.',
        [], 'Research exercise'))

    # 30. Protocol plus operating curve, explicitly schematic.
    body = text(75, 195, 'Use matched cases, then compare performance across resource budgets.', 30, 700)
    decisions = [
        ('1', 'Fix the test cases', ['Same layouts and disturbance schedule.']),
        ('2', 'Declare information and adaptation', ['Sensors, tools, prior trials, code and resets.']),
        ('3', 'Measure the physical outcome', ['Completion, elapsed time and failures.']),
        ('4', 'Sweep the inference budget', ['Repeated cases, uncertainty and cost.']),
    ]
    for i,(num,title,detail) in enumerate(decisions):
        yy=285+i*103
        body += circle(99, yy-9, 23, PALE, LINE)
        body += text(99, yy, num, 26, 700, color=BLUE, anchor='middle')
        body += text(145, yy, title, 29, 700)
        body += text(145, yy+39, detail[0], 25, color=MUTED)
    body += line(805, 242, 805, 689)
    body += text(895, 258, 'Illustrative operating curves', 28, 700, color=BLUE)
    body += text(895, 298, 'Schematic only; no measured values', 24, color=MUTED)
    body += arrow(915, 625, 1510, 625, MUTED)
    body += arrow(915, 625, 915, 342, MUTED)
    body += text(925, 337, 'Completion rate', 25, color=MUTED)
    body += text(1205, 680, 'Elapsed-time budget', 27, anchor='middle')
    body += path('M 940 594 C 975 503 1010 458 1100 443 C 1200 425 1335 424 1480 421', TEAL, 4)
    body += path('M 940 606 C 1070 599 1125 514 1235 447 C 1330 385 1405 367 1480 362', BLUE, 4)
    body += text(1360, 472, 'System A', 25, 700, color=TEAL)
    body += text(1360, 342, 'System B', 25, 700, color=BLUE)
    body += takeaway('A single success rate hides the information, time and experience used to obtain it.',
                      'Matched-interface studies test a controlled difference. Best-system studies compare complete designs.')
    out.append(slide('Solution: matched trials and a budget curve', body,
        'This is a proposed experimental design and an invented curve, not a numerical prediction. Evaluate repeated matched cases and report uncertainty appropriate to the task and scene sampling. Time includes model computation, tools and execution. State whether physics continues during computation; use the same perturbation schedule in a controlled comparison. Record training and adaptation evidence even when it cannot be equalized. A matched action interface asks about a controlled component change, while allowing each system its preferred executor asks about engineering performance. End by linking this experiment back to the observation, action, feedback and data concepts introduced at the start of the lesson.',
        [EFE], 'Research exercise'))

    assert len(out) == 8
    return out
