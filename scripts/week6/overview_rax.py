"""Three figure-led RAX slides for the 30-slide robotics overview.

The diagrams are editable SVG. The air-fryer image is RAX Figure 3,
retained with attribution. No robot execution is implied by this module.
"""

from .common import (
    BLUE, INK, LINE, MUTED, PALE, TEAL, WARM,
    arrow, image, line, lines, path, rect, slide, text,
)


SECTION = 'Robot agents and program revision'
PAPER_URL = 'https://arxiv.org/html/2610.22496v1'
RAX_SHA = '8858d7f3e34eff382abae738cf0578d2726a9a0a'
SIM_SHA = '9d03ba60cee50f5a39c0031774b337dc1ff75245'
SKILL_SHA = 'f1845de5c915b9e16590b86fbb64e957f020f5a1'
RAX = 'https://github.com/robot-agent-exploration/rax'
SKILLS = 'https://github.com/robot-agent-exploration/skills-lib'


def _node(x, y, w, h, title, subtitle, fill=PALE, color=BLUE):
    """A two-line node for an explicitly requested control-flow diagram."""
    return (
        rect(x, y, w, h, fill, LINE)
        + text(x + w / 2, y + 56, title, 32, 700, color, anchor='middle')
        + text(x + w / 2, y + 103, subtitle, 27, color=MUTED, anchor='middle')
    )


def get_slides():
    slides = []

    # 19. Two feedback loops: ordinary program execution and code revision.
    b = text(75, 210, 'Task experience changes the program and its memory.', 33, 600)
    b += _node(100, 295, 365, 145, 'Frozen coding agent', 'LLM weights stay fixed')
    b += _node(615, 295, 385, 145, 'Python program', 'Code + notes + skills')
    b += _node(1150, 295, 350, 145, 'Robot + controller', 'Tracks numeric targets', '#e4f0eb', TEAL)
    b += arrow(465, 367, 615, 367)
    b += text(540, 278, 'write / revise', 27, 600, BLUE, anchor='middle')
    b += arrow(1000, 367, 1150, 367, TEAL)
    b += text(1075, 278, 'execute', 27, 600, TEAL, anchor='middle')

    # Inner loop: the program reads feedback without another main-agent call.
    b += path('M 1325 440 V 535 H 865 V 458', TEAL, 3)
    b += arrow(865, 458, 865, 440, TEAL)
    b += text(1110, 509, 'New sensor readings', 29, 600, TEAL, anchor='middle')

    # Outer loop: experience causes edits during exploration, before freezing.
    b += path('M 700 440 V 632 H 282 V 458', BLUE, 3)
    b += arrow(282, 458, 282, 440, BLUE)
    b += text(490, 606, 'Outcomes + failures', 29, 600, BLUE, anchor='middle')
    b += text(75, 708, 'After freezing: the program can execute without the main coding agent.', 29, color=MUTED)

    notes = (
        'Read the upper arrows first. The coding agent writes a Python program that calls the '
        'same robot tools available during interactive exploration. The green return path is '
        'the program’s sensor feedback; the blue return path is the slower software revision '
        'process. These are different loops. A frozen main LLM can still produce an improving '
        'system because source code and external memory change. In RAX, compilation means '
        'turning successful decisions into executable software, not neural-network distillation. '
        'The general workflow begins with interactive tool use, then tests a program on new '
        'layouts and edits its perception, action logic, checks, and recovery. Task-specific '
        'programs belong to the learned skill library; instruction skills describe how the '
        'agent should use the interface and record experience. Tool APIs are the executable '
        'boundary underneath both. Programs may invoke a VLM where needed, so running without '
        'the main coding agent does not universally mean no model calls. Real-robot tests '
        'require operator scene resets; intervention counts as debugging rather than unattended '
        'success. Sources: repository overview, https://github.com/robot-agent-exploration/rax; '
        f'autonomous workflow, {RAX}/blob/{RAX_SHA}/agent_skills/task/autonomous-control/SKILL.md. '
        'The simulator’s offline container refers to network isolation, not offline learning: '
        'robot interaction and model access remain available through relays. Architecture: '
        f'{RAX}/blob/{SIM_SHA}/docs/ARCHITECTURE.md. '
        'Teaching interpretation: this is program improvement from experience. It does not '
        'demonstrate recursive self-improvement of the foundation-model weights.'
    )
    slides.append(slide(
        'RAX: explore, test, and revise a robot program', b, notes,
        [('RAX · workflow', RAX),
         ('Autonomous control', f'{RAX}/blob/{RAX_SHA}/agent_skills/task/autonomous-control/SKILL.md')],
        SECTION,
    ))

    # 20. One actual paper figure, with a concise numerical execution pathway.
    b = text(75, 211, 'Fitted object geometry and planned hand poses', 30, 600, BLUE)
    b += image(
        75, 245, 900, 462, 'overview-rax-articulation.png',
        'RAX Figure 3: articulated air-fryer model fitted to point-cloud measurements, '
        'and hand poses planned to open, fill, and close its drawer',
    )
    b += line(1005, 200, 1005, 710, LINE, 1.5)

    b += text(1045, 242, '1  Pixels → metres', 31, 700, BLUE)
    b += lines(1085, 287, ['RGB-D + calibration', '→ base-frame 3D points'], 27, gap=40)
    b += arrow(1265, 345, 1265, 374, BLUE)

    b += text(1045, 417, '2  Geometry → targets', 31, 700, BLUE)
    b += lines(1085, 462, ['Fit handle pull direction d', 'IK + clearance checks'], 27, gap=40)
    b += arrow(1265, 520, 1265, 549, TEAL)

    b += text(1045, 589, '3  Execute and check', 31, 700, TEAL)
    b += lines(1085, 634, ['Pull: 0.03 m along unit vector d', 'Width: 0.025 m; force: 50 N'], 27, gap=40)

    notes = (
        'The image is the authors’ Figure 3, reproduced from '
        f'{PAPER_URL}/articulation.png; RAX / Huang et al., 2026, CC BY 4.0. '
        'It depicts the fitted articulated object and planned hand poses, not a learned video '
        'world model. The left panel labels the fixed body, drawer, and pull axis; the right '
        'panel shows grasp, pull, release, and close poses. Explain the numeric pathway on '
        'the right. geom.py reads depth, intrinsics K, and camera-to-base transform T, '
        'back-projects valid depth pixels, and transforms their points into the robot base '
        'frame. Depth is converted to metres. Source: '
        f'{SKILLS}/blob/{SKILL_SHA}/skills/air-fry-pan/code/air_fry_pan/geom.py. '
        'The released perception code uses color/height filters and geometric fits for this '
        'specific white fryer, dark handle, and yellow banana. This task’s autonomous program '
        'does not require a runtime learned detector. Source: '
        f'{SKILLS}/blob/{SKILL_SHA}/skills/air-fry-pan/code/air_fry_pan/perception.py. '
        'The controller searches inverse-kinematics candidates and rejects insufficient joint '
        'margin or clearance. Its pulling motion advances along the measured handle direction '
        'in 3-cm increments. Here d is drawn as a horizontal unit vector embedded in the base '
        'frame, and p is the hand target in metres; this equation explains the implementation '
        'rather than quoting a measured pose. The 0.025-m width and 50-N force are actual '
        'handle-grasp settings, not a claim that the measured grasp width equals its target. '
        'Source: '
        f'{SKILLS}/blob/{SKILL_SHA}/skills/air-fry-pan/code/air_fry_pan/controller.py. '
        'Motion wrappers send position/quaternion or joint targets and gripper commands '
        'through MCP. The vendor controller generates physical motion; the LLM does not '
        'emit motor torques. FR3 uses metres and [x,y,z,w] quaternions, targeting the point '
        'between the fingers. Interface: '
        f'{RAX}/blob/{RAX_SHA}/agent_skills/control/franka/robot-interface/SKILL.md. '
        'Generalization is bounded: this was one rig/day with specific colors, dimensions '
        'and tested placements. Four later runs were clean unattended successes, while '
        'earlier runs included a camera collision and interventions. The published skill '
        'entry remains marked draft; transferring it requires calibration and fresh checks. '
        'Evidence and limits: '
        f'{SKILLS}/blob/{SKILL_SHA}/skills/air-fry-pan/SKILL.md.'
    )
    slides.append(slide(
        'Air fryer: geometry becomes numeric actions', b, notes,
        [('RAX · Fig. 3 (CC BY 4.0)', PAPER_URL + '#S3.F3'),
         ('Released air-fryer code', f'{SKILLS}/tree/{SKILL_SHA}/skills/air-fry-pan')],
        SECTION,
    ))

    # 21. A task-level result, explicitly distinct from pooled benchmark headlines.
    b = text(75, 210, 'RoboDojo · plug in charger', 33, 700, BLUE)
    b += text(75, 259, 'Paper-reported success on 50 unseen layouts', 28, color=MUTED)
    x_axis, y_base, y_top = 170, 648, 310
    chart_right = 780
    for value in (0, 50, 100):
        yy = y_base - (y_base - y_top) * value / 100
        b += line(x_axis, yy, chart_right, yy, LINE, 1.3)
        b += text(145, yy + 9, f'{value}%', 27, color=MUTED, anchor='end')
    b += line(x_axis, y_base, chart_right, y_base, INK, 2)
    for x, episodes, score, color in ((250, 4, 26, BLUE), (535, 8, 94, TEAL)):
        height = (y_base - y_top) * score / 100
        b += rect(x, y_base - height, 145, height, color, 'none')
        b += text(x + 72.5, y_base - height - 18, f'{score}%', 43, 700, color, anchor='middle')
        b += text(x + 72.5, 694, f'{episodes} episodes', 29, 600, anchor='middle')

    b += line(870, 205, 870, 708, LINE, 1.5)
    b += text(940, 239, 'Keep the evaluation boundary explicit', 29, 600)
    b += _node(985, 282, 490, 112, 'Explore', '4 or 8 episodes', PALE, BLUE)
    b += arrow(1230, 394, 1230, 423, BLUE)
    b += _node(985, 425, 490, 112, 'Freeze', 'Submit the rehearsed program', '#f8f4ef', WARM)
    b += arrow(1230, 537, 1230, 566, TEAL)
    b += _node(985, 568, 490, 112, 'Evaluate', 'Fresh layouts · no agent edits', '#e4f0eb', TEAL)

    notes = (
        'This is one controlled task comparison from RAX Table 2, not the full RoboDojo '
        'benchmark. Both runs use the same task interface and differ in exploration budget. '
        'Plug in charger has 50 evaluation layouts and binary scoring: 26% versus 94% after '
        'budgets of four versus eight exploration episodes. Rehearsals consume that budget. '
        'The code bundle is frozen and run without the coding agent, edits, or retries on '
        'unseen layouts; evaluation results are not returned to the explorer. The explorer '
        'receives RGB-D, calibration, the task rubric, a completion image, and tool access. '
        'These are paper-reported results, not our reproduction. Do not infer uncertainty '
        'intervals or generalize monotonic improvement to every task. Other paper results '
        'use different protocols: BEHAVIOR is mostly agent-in-the-loop, including cross-test '
        'review reports on three tasks; HumanoidArena’s main setting supplies three action '
        'demonstrations. The +37.4 headline averages percentage-point gains across different '
        'benchmarks. The 241× air-fryer claim concerns inference overhead, excluding robot '
        'execution time. Sources: RAX §§5.2.1, 5.3.3, 10.3, Table 2 and Table 13; '
        f'{PAPER_URL}. '
        'The current usage guide exposes observation profile, demonstration context, rubric '
        'access, budget, layout split, review feedback, and evaluation mode as configuration '
        'choices. Treat these as experimental conditions; current defaults are not a full '
        'reconstruction of the paper’s historical runs. The formal code mode and formal '
        'agent mode are distinct. Source: '
        f'{RAX}/blob/{SIM_SHA}/docs/USAGE.md. '
        'Teaching point: more interaction can help an agent construct a more reliable '
        'program, but the experiment must separate acquisition experience from evaluation '
        'and report the observation/action interface. This task-level plot avoids comparing '
        'heterogeneous system scores as though they measured one common capability.'
    )
    slides.append(slide(
        'RoboDojo: explore, freeze, then evaluate', b, notes,
        [('RAX · Table 2 + protocol', PAPER_URL + '#S5.SS2.SSS1'),
         ('Evaluation modes', f'{RAX}/blob/{SIM_SHA}/docs/USAGE.md#4-evaluation')],
        SECTION,
    ))

    return slides
