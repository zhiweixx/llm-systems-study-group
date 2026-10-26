"""Three overview slides on retained code, robot execution, and improvement.

All diagrams are authored explanations of the cited papers. They are not
screenshots of either implementation or records of new robot experiments.
"""
from .common import (
    BLUE, INK, LINE, MUTED, PALE, TEAL, WARM,
    arrow, line, lines, path, rect, slide, takeaway, text,
)


COAP = ('COAP · §§3.1–3.4', 'https://arxiv.org/abs/2610.12369')
RSI_SETUP = ('123-round study · §2', 'https://arxiv.org/html/2609.31760v1#S2')
RSI_RESULTS = ('123-round study · Tables 5–7', 'https://arxiv.org/html/2609.31760v1')
SECTION = 'Code, execution, and improvement'


def _node(x, y, w, h, title, details=(), color=BLUE, fill=PALE):
    """A named component in a dataflow diagram, with explicit input/output role."""
    body = rect(x, y, w, h, fill, LINE)
    body += text(x + w / 2, y + 43, title, 29, 700, color, anchor='middle')
    for i, detail in enumerate(details):
        body += text(x + w / 2, y + 88 + i * 37, detail, 26,
                     color=INK, anchor='middle')
    return body


def get_slides():
    out = []

    # 25. Source code is fixed within an episode, while measured state changes.
    b = text(75, 217, 'Coding agents', 31, 700, color=BLUE)
    b += arrow(310, 207, 410, 207)
    b += rect(430, 175, 400, 70, PALE)
    b += text(630, 220, 'Accepted source code', 31, 700,
              color=BLUE, anchor='middle')
    b += lines(890, 201, ['Developed between episodes',
                          'Fixed source during execution'], 28, gap=42, color=MUTED)
    b += path('M 630 245 V 276 H 815 V 320', BLUE, 2)
    b += arrow(815, 320, 815, 345)
    b += text(75, 307, 'During each episode', 31, 700, color=TEAL)

    b += _node(75, 345, 235, 160, 'Observations',
               ['RGB views', 'robot readings'], TEAL)
    b += _node(365, 345, 255, 160, 'Perception + state',
               ['object poses', 'task stage'], TEAL)
    b += _node(675, 345, 280, 160, 'Code + geometry',
               ['choose next move', 'IK and collision'], BLUE)
    b += _node(1010, 345, 235, 160, 'Controller',
               ['track trajectory', 'move joints'], BLUE)
    b += _node(1300, 345, 225, 160, 'Robot',
               ['act in the', 'environment'], TEAL)
    for x1, x2 in [(310, 365), (620, 675), (955, 1010), (1245, 1300)]:
        b += arrow(x1 + 7, 425, x2 - 7, 425)
    b += path('M 1412 505 V 605 H 192 V 535', TEAL, 2)
    b += arrow(192, 535, 192, 505, TEAL)
    b += text(805, 578, 'Fresh measurements update the state', 29, 600,
              color=TEAL, anchor='middle')
    b += text(75, 670, 'Reported runtime: no LLM, VLM, or VLA service', 28, 700, color=BLUE)
    b += text(75, 713, 'Setup includes calibrated cameras, known object shapes, and robot geometry.',
              27, color=MUTED)
    b += takeaway('Frozen code can still branch, remeasure, and retry.')
    out.append(slide(
        'COAP: Code-Only-as-Policy', b,
        'COAP is Code-Only-as-Policy, the Ropedia A0 paper at arXiv:2610.12369. '
        'It is distinct from the earlier Code as Policies method. This authored diagram '
        'summarizes §§3.1–3.4 and Figure 3. The paper places coding agents in development, '
        'where they inspect logged failures and revise a shared library. Once an episode '
        'starts, source code θ stays fixed. The evolving state x_t and current observations '
        'o_t determine both the next state and trajectory segment through '
        '(x_{t+1}, a_t) = F_θ(x_t, o_t). A fixed program therefore does not mean replaying '
        'a fixed action sequence. Its conditions, measurements, recovery branches, and retry '
        'counts can produce different movements in different episodes.\n\n'
        'The observation interface includes three RGB views, from the head and both wrists, '
        'joint angles, end-effector poses, and gripper command echoes. Explicit state records '
        'object measurements and their confidence/source/time, robot configuration, and task '
        'progress or memory. NumPy/OpenCV perception fits known object geometry to image '
        'regions. Motion code uses kinematics and collision checks to choose feasible '
        'trajectories, and the simulator controller executes commands. The simplified '
        'diagram groups some routines for readability. It does not show a neural policy '
        'hidden in the geometry box.\n\n'
        'The paper reports no LLM, VLM, or VLA service during execution. An online GPT '
        'controller places model calls on the observation-to-decision path instead. That is '
        'an execution boundary, not an intrinsic ranking of the two approaches. COAP uses '
        'benchmark-provided camera calibration, robot meshes, known object shapes, and '
        'table height as constants. It measures episode-specific object states from RGB. '
        'Development agents can inspect simulator ground truth, while the deployed policy '
        'does not receive oracle object state. Do not generalize its perception assumptions '
        'to arbitrary unseen household objects or compare its scores directly with the '
        'following study. Source: https://arxiv.org/abs/2610.12369, §§3.1–3.4.',
        [COAP], SECTION,
    ))

    # 26. The human-maintained harness is visibly outside the editable library.
    b = text(75, 209, 'A separate 123-round study in RoboCasa', 34, 700, color=BLUE)
    b += text(1030, 209, 'PandaOmron robot', 30, 600, color=TEAL)
    b += text(75, 264, 'Target: move condiments onto the top fridge shelf', 31)
    b += text(75, 309, 'Provider = external perception or planning component', 26, color=MUTED)
    b += _node(75, 335, 270, 170, 'Run program',
               ['Python skills', 'call providers'], TEAL)
    b += _node(395, 335, 270, 170, 'Judge effects',
               ['simulator truth', 'failure diagnosis'], WARM, '#faf1e8')
    b += _node(715, 335, 310, 170, 'Coding agents',
               ['revise skill code', 'or add a provider'], BLUE)
    b += _node(1075, 335, 450, 170, 'Test candidate changes',
               ['replay + fresh paired trials', 'accept or reject'], WARM, '#faf1e8')
    for x1, x2 in [(345, 395), (665, 715), (1025, 1075)]:
        b += arrow(x1 + 7, 420, x2 - 7, 420)
    b += path('M 1300 505 V 567 H 210 V 535', BLUE, 2)
    b += arrow(210, 535, 210, 505)
    b += rect(455, 545, 610, 45, 'white', 'none')
    b += text(760, 576, 'Accepted code enters the next round', 28, 600,
              color=BLUE, anchor='middle')
    b += line(75, 625, 1525, 625, WARM, 2)
    b += text(75, 669, 'Human-only changes to the harness', 31, 700, color=WARM)
    b += text(75, 708, 'Evaluators, test gates, memory, and descriptions of the robot’s actuators',
              28, color=INK)
    b += takeaway('Coding agents revise the software used by later robot runs.')
    out.append(slide(
        'A robot software improvement loop', b,
        'This is Wang’s technical report What Stops Recursive Self-Improvement in Robotics? '
        'Lessons from 123 Rounds of Agentic Skill Discovery, arXiv:2609.31760v1. The diagram '
        'adapts its Figure 2. The simulated PandaOmron mobile manipulator receives head and '
        'wrist RGB-D images and joint readings. Its task program calls typed skills for '
        'localization, finding fixture regions, approach, grasp, and motion. Skills call '
        'external providers such as segmentation, depth, grasp generation, motion planning, '
        'and active-view selection. Unlike COAP, this runtime includes external model '
        'providers. The GPT coding agents themselves are never called by the deployed robot.\n\n'
        'Three agent roles operate between trials. The orchestrator writes task programs. '
        'The proposer sees diagnoses, execution traces, past attempts, and camera keyframes, '
        'then proposes a skill edit or research request. A research agent finds, installs, '
        'wraps, tests, and registers external components. Routing is refitted from skill-call '
        'logs. Accepted changes go live in a later round. The report describes code/library '
        'changes and provider acquisition, not gradient-based training of an action policy. '
        'Humans chose the coding model and reasoning setting: gpt-5.6-sol for rounds 1–75, '
        'then gpt-6-astra. Those model-selection changes were not made by the loop.\n\n'
        'Declared step effects are judged using privileged simulator state. Exact failure '
        'replay checks an observed case, while paired old/new tests use fresh states and a '
        'held-out cabinet-to-counter task. Sealed failures and checks on other affected '
        'skills were added later. This protocol evolved during the run. Four scene seeds '
        'per task were repeatedly used for collection in a single kitchen layout. Cabinet '
        'and Sink were added after the initial Fridge phase. The held-out task participates '
        'in adaptive acceptance, so it is not an untouched final benchmark. From round 74, '
        'containers hid the run repository and held-out task from agents.\n\n'
        'The warm boundary is central: humans were prohibited from writing skills, programs, '
        'or model wrappers, but could repair the evaluator, tests, memory, and actuator '
        'description. They made 113 such repairs and suggested the later VLM grounding '
        'class. The harness therefore remained a substantial human-maintained component. '
        'The term self-improvement describes a bounded software loop, not a robot changing '
        'every part of its own learning process. Source: https://arxiv.org/html/2609.31760v1, '
        '§2, Table 1, and Appendices A–B.',
        [RSI_SETUP], SECTION,
    ))

    # 27. Counts describe activity, deployment, and task completion separately.
    b = text(75, 208, 'One development run across repeated simulated scenes',
             30, color=MUTED)
    b += text(100, 338, '57', 96, 700, color=BLUE)
    b += text(100, 390, 'accepted library changes', 32, 600)
    b += text(100, 433, 'Code changed between rounds', 27, color=MUTED)
    b += line(765, 260, 765, 445, LINE, 2)
    b += text(850, 338, '0 / 486', 90, 700, color=WARM)
    b += text(850, 390, 'fridge task completions', 32, 600)
    b += text(850, 433, 'among completed collection rollouts', 27, color=MUTED)

    b += line(75, 485, 1525, 485)
    b += text(75, 534, 'A registered provider must enter a skill’s execution path',
              31, 700, color=BLUE)
    b += text(75, 603, '11 late providers', 35, 700, color=BLUE)
    b += text(75, 648, 'registered in rounds 114–121', 27, color=MUTED)
    b += arrow(535, 594, 675, 594, WARM)
    b += text(730, 603, '0 later deployed calls', 35, 700, color=WARM)
    b += text(730, 648, 'during subsequent task collection', 27, color=MUTED)
    b += text(75, 712, 'One kitchen layout, evolving agents and harness, no independent repeats',
              27, color=MUTED)
    b += takeaway('Changed artifacts, deployed calls, and task outcomes are separate measurements.')
    out.append(slide(
        '57 accepted changes, zero fridge completions', b,
        'The two headline counts come from Appendix B, Tables 5–6. There were 57 accepted '
        'library changes, of which 51 concerned perception, and no Fridge completion among '
        '486 completed collection rollouts. The other completed-rollout totals were Cabinet '
        '0/256 and Sink 14/230. Nine Sink successes had failed intermediate program steps. '
        'The narrative explains that many succeeded when an object happened to fall into '
        'the sink after the intended placement sequence failed. Native task success and '
        'successful execution of the declared chain are different measurements. The slide '
        'uses cumulative collection outcomes, not a final held-out test success estimate.\n\n'
        'The full run contained 847 proposal attempts, 993 attempted and 972 completed '
        'rollouts, and 113 human harness repairs. Routine routing refits are excluded from '
        'the 57 accepted changes. Eighty-six provider registration events correspond to '
        'only 20 identities, because repaired providers could be registered again. Eight '
        'identities were later used by deployed skills. All eleven identities introduced '
        'during rounds 114–121, including new vision-language grounding components, had '
        'zero calls in subsequent task collection. The lower diagram represents this '
        'late subset. It does not claim that all providers were unused. The arrow compares '
        'two observed stages of integration, not a successful transition.\n\n'
        'Three mechanisms help explain the outcome. The segmentation pipeline repeatedly '
        'confused relational targets such as the top shelf. Upstream failures prevented '
        'rollouts from reaching later skills: 57% of fridge rollouts in the three-task '
        'phase failed at step 1, and only 2.7% passed four steps. Finally, evaluator errors '
        'and incorrectly recorded research requests distorted selection and memory. A '
        'grasp change in round 118 won reproducible fresh-state and held-out comparisons, '
        'yet the next 60 task rollouts had no completion. A local improvement can be real '
        'and still make little measurable difference to a long task.\n\n'
        'This is one long run in one simulator and kitchen layout. It has no independent '
        'replications, matched-budget baselines, or controlled ablations. Agents, providers, '
        'and the human-maintained harness changed together. It supports specific failure '
        'diagnoses, not a universal claim that robot self-improvement cannot work or that '
        'a stronger agent cannot help. Its setup differs substantially from COAP, so the '
        'headline outcomes are not a head-to-head comparison. Source: '
        'https://arxiv.org/html/2609.31760v1, §§5–7 and 10, Tables 5–7.',
        [RSI_RESULTS], SECTION,
    ))

    return out
