"""Slides 1–11: robot interfaces, demonstrations, and imitation learning."""
from .common import *

MIT_PICK = ('MIT · Pick and place', 'https://manipulation.mit.edu/pick.html')
CONTROL = ('Modern Robotics · Control', 'https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/')
CS224R = ('CS224R · Imitation', 'https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf')
ACT = ('ACT · §IV', 'https://arxiv.org/html/2304.13705v1')
DP = ('Diffusion Policy · §2', 'https://arxiv.org/html/2303.04137v5')
UMI = ('UMI · Data and timing', 'https://arxiv.org/html/2402.10329')
DAGGER = ('DAgger · Algorithm 3.1', 'https://proceedings.mlr.press/v15/ross11a/ross11a.pdf')
META = ('ALOHA · Metadata', 'https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json')
ROWS = ('Episode 0 · Stored rows', 'https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2')
CONVERT = ('ALOHA · Converter', 'https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py')


def _block(x, y, w, h, heading, detail=(), fill=PALE):
    """A block in a process diagram, with readable multiline labels."""
    return (rect(x, y, w, h, fill)
            + text(x + w / 2, y + 45, heading, 30, 700, anchor='middle')
            + lines(x + w / 2, y + 87, detail, 25, gap=34, anchor='middle'))


def _joint_arm():
    """Planar teaching schematic, not a scale model of ALOHA."""
    s = line(115, 555, 655, 555, MUTED, 2)
    s += rect(155, 520, 110, 35, PALE, BLUE)
    pts = [(210, 520), (260, 385), (440, 330), (535, 390)]
    for a, b in zip(pts, pts[1:]):
        s += line(*a, *b, LINE, 22) + line(*a, *b, BLUE, 3)
    for i, (x, y) in enumerate(pts[:-1]):
        s += circle(x, y, 17, 'white') + text(x - 37, y - 28, f'q{i+1}', 29, 700)
    s += line(535, 390, 570, 390, BLUE, 6)
    s += path('M 570 369 L 598 369 M 570 411 L 598 411 M 570 369 L 570 411', BLUE, 5)
    s += text(475, 505, 'tool / gripper', 26, color=MUTED)
    # World frame and a deliberately rotated tool frame.
    s += arrow(115, 520, 115, 442, TEAL) + arrow(115, 520, 183, 520, TEAL)
    s += text(100, 433, 'z', 24, color=TEAL) + text(183, 548, 'x', 24, color=TEAL)
    s += text(78, 595, 'World frame W', 27, 700, color=TEAL)
    s += arrow(598, 390, 650, 348, WARM) + arrow(598, 390, 640, 442, WARM)
    s += text(660, 347, 'x′', 24, color=WARM) + text(647, 465, 'z′', 24, color=WARM)
    s += text(448, 284, 'Tool frame', 27, 700, color=WARM)
    s += line(568, 292, 598, 365, WARM, 1.5, '5 5')
    return s


def _action_cells(x, y, start, count, fill=PALE, executed=0, replaced=False):
    s = ''
    subs = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    supers = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
    for j in range(count):
        f = '#e5f0ec' if j < executed else ('#f1f2f3' if replaced else fill)
        color = TEAL if j < executed else (MUTED if replaced else BLUE)
        xx = x + j * 111
        s += rect(xx, y, 101, 58, f, LINE)
        command = 'a' + str(start+j).translate(subs) + '⁽' + str(start).translate(supers) + '⁾'
        s += text(xx + 50.5, y + 39, command, 29, 600, color, 'middle')
        if replaced and j >= executed:
            s += line(xx + 15, y + 45, xx + 86, y + 13, '#a9b1b6', 1.5)
    return s


def get_slides():
    out = []

    # 1. The three questions are the spine of the whole lesson.
    b = text(75, 205, 'Illustrative task: put the cup into the drawer.', 34)
    b += text(75, 257, 'A useful answer must eventually become an executable command.', 29, color=MUTED)
    blocks = [(75, 280, 'Observe', ['images + robot sensors']),
              (485, 280, 'Decide', ['policy, planner, or LLM']),
              (895, 280, 'Execute', ['controller + robot'])]
    for x, w, h, detail in blocks:
        b += _block(x, 330, w, 130, h, detail)
    b += arrow(355, 395, 485, 395) + text(420, 374, 'input', 24, color=BLUE, anchor='middle')
    b += arrow(765, 395, 895, 395) + text(830, 374, 'command', 24, color=BLUE, anchor='middle')
    b += arrow(1175, 395, 1490, 395) + text(1332, 374, 'world changes', 26, color=BLUE, anchor='middle')
    b += line(1490, 395, 1490, 515, BLUE, 2) + line(1490, 515, 215, 515, BLUE, 2)
    b += arrow(215, 515, 215, 460)
    b += text(825, 502, 'new observations', 25, color=BLUE, anchor='middle')
    questions = [('What is an action?', 'Interfaces, units, and feedback'),
                 ('How is it learned?', 'Demonstrations → action policies'),
                 ('What does language add?', 'VLAs and LLM-controlled robots')]
    for i, (head, sub) in enumerate(questions):
        x = 75 + i * 500
        b += line(x, 588, x + 450, 588) + text(x, 635, head, 31, 700, color=BLUE)
        b += text(x, 682, sub, 26)
    b += takeaway('Follow the interface from an observation to motion, then back to observation.',
                  '30 slides · robot data, imitation learning, vision–language–action models, and language-driven control.')
    out.append(slide('From images and language to robot motion', b,
        'Use the cup-to-drawer instruction as an illustrative running task, not as an experimental claim or the task in the ALOHA dataset example. Ask what exactly leaves the model: a skill name, a tool pose, a vector of joint targets, or a sequence of commands. Every choice requires an execution interface. The robot changes its world, so subsequent inputs depend on past decisions. This is the conceptual bridge from familiar statistical prediction to embodied feedback. The first eleven pages establish interfaces and imitation learning before adding pretrained visual/language representations. No numerical benchmark is implied by this diagram.',
        [MIT_PICK, CS224R], 'Opening'))

    # 2. Distinct visual and low-level feedback loops.
    b = text(75, 201, 'The policy chooses a command; the controller uses feedback to track it.', 32)
    b += _block(75, 280, 290, 155, 'Observation oₜ', ['camera images', 'joint measurements'])
    b += _block(455, 280, 285, 155, 'Policy / planner', ['choose the next', 'desired motion'])
    b += _block(845, 280, 285, 155, 'Controller', ['compare target', 'with measurements'])
    b += _block(1235, 280, 290, 155, 'Robot + world', ['motors, contact,', 'objects, disturbances'])
    b += arrow(365, 355, 455, 355)
    b += arrow(740, 355, 845, 355) + text(792, 267, 'command', 24, color=BLUE, anchor='middle')
    b += arrow(1130, 355, 1235, 355) + text(1182, 267, 'actuation', 24, color=BLUE, anchor='middle')
    b += line(1380, 435, 1380, 500, TEAL, 2) + line(1380, 500, 988, 500, TEAL, 2)
    b += arrow(988, 500, 988, 435, TEAL)
    b += text(1185, 539, 'rapid joint feedback', 27, 700, color=TEAL, anchor='middle')
    b += line(1490, 435, 1490, 585, BLUE, 2) + line(1490, 585, 220, 585, BLUE, 2)
    b += arrow(220, 585, 220, 435)
    b += text(610, 574, 'new images + measurements', 27, color=BLUE, anchor='middle')
    b += text(75, 645, 'Proprioception = measurements of the robot itself, such as joint position and velocity.', 28)
    b += text(75, 694, 'Later: an LLM may call a skill that produces commands for this same execution stack.', 27, color=MUTED)
    b += takeaway('Observations reveal part of the physical state; commands influence what is observed next.',
                  'A dataset field named “state” may contain only measured robot variables, not the complete world state.')
    out.append(slide('A robot acts inside a feedback loop', b,
        'Read the outer loop first. Cameras and joint sensors supply observations to a policy or a planner. The output is a desired motion, while a controller uses measurements to track it through actuation. The inner green loop remains meaningful even when visual decisions are slower. Proprioception means robot-self measurements, such as encoders and velocity estimates; it is not another camera view. The world state also includes object pose, contact, friction, and other quantities that are not directly observed. A current image plus joint positions need not be Markov; histories or state estimation may help. “Policy” and “controller” can overlap in robotics terminology; here we separate a learned decision interface from its lower-level tracking implementation. A skill-call interface adds a skill executor, and Cartesian targets may add inverse kinematics or planning, before reaching low-level control. A torque policy would move the boundary, so the diagram is a position-target example rather than a universal stack.',
        [CONTROL, MIT_PICK, ('MIT · Output feedback', 'https://underactuated.mit.edu/output_feedback.html')], 'Robot interfaces'))

    # 3. Minimal kinematics, with both frames and units.
    b = text(75, 201, 'A vector becomes meaningful only after the robot interface is specified.', 32)
    b += _joint_arm()
    b += rect(785, 235, 740, 125, PALE)
    b += text(812, 276, 'Joint target', 31, 700, color=BLUE)
    b += text(812, 318, 'q* = [q₁*, q₂*, …]   +   gripper command', 29)
    b += text(1060, 398, 'Inverse kinematics (IK)', 27, color=BLUE)
    b += arrow(1395, 432, 1395, 360)
    b += rect(785, 432, 740, 140, PALE)
    b += text(812, 476, 'End-effector pose target', 31, 700, color=BLUE)
    b += text(812, 519, 'tool position + orientation, relative to frame W', 28)
    b += text(812, 555, 'position in meters; orientation convention required', 25, color=MUTED)
    b += text(75, 652, 'Forward kinematics: joint configuration q → tool pose.', 29)
    b += text(75, 701, 'Also specify: absolute target or change?  Joint angles in rad?  Gripper width or normalized value?', 27)
    b += takeaway('The same numbers in two coordinate frames can request different physical motions.',
                  'IK finds a joint configuration for a desired tool pose; it does not by itself guarantee a collision-free path.')
    out.append(slide('What does an action vector mean?', b,
        'The arm is a planar teaching schematic, not ALOHA geometry or a complete three-dimensional frame drawing. A real frame has three orthogonal axes; the two displayed axes emphasize that tool and world axes can differ. Joint configuration describes the robot’s internal coordinates. End-effector pose describes tool position and orientation relative to a reference frame. Forward kinematics maps configuration to pose; inverse kinematics seeks configurations achieving a desired pose and may have zero, one, or multiple solutions. A valid endpoint is different from a feasible, collision-free trajectory to it. The starred q is a desired configuration. Units and joint order must be supplied by the interface; revolute coordinates commonly use radians, whereas prismatic coordinates use length. Cartesian orientation can use a rotation matrix, quaternion, axis-angle, or another convention. Absolute, relative-to-current, and sequential-delta commands also differ. A gripper coordinate might be width, normalized opening, velocity, or force. None of these conventions should be inferred from a vector dimension alone.',
        [MIT_PICK, ('Modern Robotics · Transforms', 'https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/')], 'Robot interfaces'))

    # 4. Explicitly invented timing example, not a dataset or model benchmark.
    b = text(75, 199, 'Illustrative asynchronous schedule: physical time continues while the model computes.', 30)
    x0, span = 375, 1100
    px = lambda ms: x0 + span * ms / 200
    for ms in (0, 50, 100, 150, 200):
        x = px(ms)
        b += line(x, 242, x, 628, '#dbe3e8', 1.5, '4 5') + text(x, 234, f'{ms} ms', 24, color=MUTED, anchor='middle')
    labels = [('Camera', '30 Hz', 294), ('Model call', '80 ms latency', 397), ('Command playback', '50 Hz', 504), ('Joint servo', '1 kHz', 604)]
    for main, sub, y in labels:
        b += text(75, y, main, 29, 700) + text(75, y + 34, sub, 25, color=MUTED)
        b += line(x0, y, 1490, y, LINE)
    for ms in (0, 33.3, 66.7, 100, 133.3, 166.7, 200):
        b += circle(px(ms), 294, 9, PALE, BLUE)
    for start in (0, 100):
        b += rect(px(start), 368, px(80)-px(0), 58, PALE)
        b += text(px(start+40), 405, 'infer from latest image', 24, 600, anchor='middle')
        b += arrow(px(start), 306, px(start), 366)
        b += arrow(px(start+80), 426, px(start+80), 477)
    for ms in (80, 100, 120, 140, 160, 180):
        b += rect(px(ms), 477, px(20)-px(0)-4, 54, '#e5f0ec', TEAL)
        b += text(px(ms+10)-2, 513, 'target', 24, color=TEAL, anchor='middle')
    b += rect(x0, 477, px(80)-x0-4, 54, 'white', LINE)
    b += text(px(39), 512, 'prior command / hold', 24, color=MUTED, anchor='middle')
    b += rect(x0, 580, span, 48, '#f2f6f3', 'none')
    for ms in range(0, 201, 5):
        b += line(px(ms), 586, px(ms), 622, TEAL, 1.5)
    b += text(75, 688, 'At 80 ms, the first new target is based on an image captured 80 ms earlier.', 29)
    b += takeaway('A fast command stream does not imply equally fast model decisions.',
                  'All rates here are illustrative. Servo ticks are compressed; actual sensing and execution delays must be measured.')
    out.append(slide('The model and the motors run on different clocks', b,
        'This is an authored asynchronous timing example, not measured performance from ACT, UMI, or a VLA. Camera captures are approximately every 33 milliseconds. Two model calls start 100 milliseconds apart and each takes 80 milliseconds; a call uses the latest image at its start. New commands begin when results are available and are played at 20-millisecond intervals. A previous command or a hold behavior covers the initial computation period. Servo feedback remains active throughout, including during model computation; its drawn ticks are compressed and do not each represent one millisecond. Under the simplified zero-transport-delay assumption, the first result uses an 80-millisecond-old image. Real systems also incur sensor, communication, actuator, and scheduling delays. UMI makes this concrete by aligning measurements to image capture times and matching action execution times to measured latency. A simulator that pauses during inference would have different physical-time behavior. Lower model latency and faster actuator command playback address different bottlenecks.',
        [UMI, CONTROL, DP], 'Timing'))

    # 5. Actual documented rows and images, with nominal timestamps.
    b = text(75, 195, 'LeRobot · aloha_sim_insertion_human · episode 0 · 500 records at nominal 50 Hz', 29)
    b += image(75, 228, 360, 270, 'aloha-frame0.png', 'ALOHA simulated insertion, episode 0, frame 0')
    b += image(460, 228, 360, 270, 'aloha-frame100.png', 'ALOHA simulated insertion, episode 0, frame 100')
    b += text(255, 531, 'frame 0 · nominal 0.00 s', 25, anchor='middle')
    b += text(640, 531, 'frame 100 · nominal 2.00 s', 25, anchor='middle')
    b += text(875, 258, 'One stored row', 31, 700, color=BLUE)
    b += lines(875, 306, ['RGB image: 480 × 640 × 3', 'observation.state: float32[14]', 'action: float32[14]'], 26, gap=42)
    b += text(875, 449, '14 = (6 arm + 1 gripper) × 2', 28, 700)
    b += text(875, 495, 'Task metadata: insert peg into socket.', 25, color=MUTED)
    b += table(75, 562, 1450,
               ['Nominal frame time', 'Measured left shoulder', 'Stored shoulder target'],
               [['0.00 s', '−0.9600 rad', '−0.9342 rad'],
                ['0.02 s', '−0.9567 rad', '−0.9342 rad']],
               ratios=[.30, .35, .35], row_h=51, size=27)
    b += takeaway('The action is a desired joint configuration, not the next measured state.',
                  'Simulation · joint targets in rad · gripper nominally 0–1 · time = frame index / 50 Hz.')
    out.append(slide('Open one demonstration episode', b,
        'This is the public simulated ALOHA insertion dataset, not a physical-robot episode. The two images are actual frames 0 and 100 of episode 0. The numeric table deliberately compares rows 0 and 1 instead, to expose the distinction between a measured position and its stored target. Episode 0 contains 500 records. Each state and action vector has left six arm joints, left gripper, right six arm joints, and right gripper. The simulation interface defines absolute desired joint coordinates, with revolute arm angles in radians and normalized gripper opening nominally zero closed and one open; recorded normalized measurements may slightly exceed that nominal range. The inspected historical converter copies qpos, action, and images at the same index and constructs timestamp = frame_index / fps. These are nominal episode times, not measured sensor timestamps or proof that a human issued a command at that exact wall-clock instant. The ACT simulation loader pairs qpos/images at t with actions starting at t. The task string in current dataset metadata is “Insert the peg into the socket.” This is a task label, not per-frame speech, and original ACT does not consume language. ACT simulation interface: https://github.com/tonyzhaozh/act/blob/main/sim_env.py ; original loader: https://github.com/tonyzhaozh/act/blob/main/utils.py . The converter sets next.done on the final record, so that field does not by itself certify success.',
        [META, ROWS, CONVERT], 'A real dataset record'))

    # 6. Windows that preserve episode boundaries.
    b = text(75, 201, 'Choose an observation at t; use the following H stored commands as the target.', 31)
    b += text(75, 293, 'Observation', 28, 700) + text(75, 394, 'Action', 28, 700)
    xs = [360, 530, 700, 870, 1040, 1300]
    idx = ['t', 't+1', 't+2', 't+3', 't+4', '0']
    for i, (x, lab) in enumerate(zip(xs, idx)):
        b += text(x + 75, 249, lab, 25, color=MUTED, anchor='middle')
        b += label_box(x, 267, 150, 63, 'o' + ('ₜ' if i == 0 else ''), '#e5f0ec' if i == 0 else 'white', 28)
        b += label_box(x, 366, 150, 63, 'a' + ('ₜ' if i == 0 else ''), PALE if i < 4 else 'white', 28)
    b += line(1235, 232, 1235, 461, WARM, 2, '7 6')
    b += text(1375, 461, 'next episode', 24, color=WARM, anchor='middle')
    b += line(360, 451, 1020, 451, BLUE, 3)
    b += mathml(360, 455, 660, 66,
        '<mrow><msub><mi>A</mi><mi>t</mi></msub><mo>=</mo><mo>[</mo>'
        '<msub><mi>a</mi><mi>t</mi></msub><mo>,</mo><mo>…</mo><mo>,</mo>'
        '<msub><mi>a</mi><mrow><mi>t</mi><mo>+</mo><mi>H</mi><mo>−</mo><mn>1</mn></mrow></msub>'
        '<mo>]</mo><mspace width="1em"/><mi>H</mi><mo>=</mo><mn>4</mn></mrow>', 32)
    b += _block(75, 545, 410, 112, 'Input', ['oₜ = images + measured qₜ'])
    b += arrow(485, 601, 600, 601)
    b += label_box(600, 558, 300, 85, 'Policy πθ', size=31)
    b += arrow(900, 601, 1015, 601)
    b += _block(1015, 545, 510, 112, 'Training target')
    b += mathml(1040, 600, 460, 52,
        '<mrow><msub><mi>A</mi><mi>t</mi></msub><mo>∈</mo><msup><mi>ℝ</mi>'
        '<mrow><mi>H</mi><mo>×</mo><msub><mi>d</mi><mi>a</mi></msub></mrow></msup>'
        '<mspace width="0.7em"/><mtext>H commands</mtext></mrow>', 29)
    b += text(75, 706, 'Language is optional conditioning; original ACT uses images and joint measurements.', 28)
    b += takeaway('A training pair links observations to a future sequence of commands.',
                  'Split by episode; near the end, drop a window or pad with an explicit validity mask. Never join two episodes.')
    out.append(slide('Turn an episode into training examples', b,
        'This diagram is a pedagogical H=4 training window, not the reported ACT chunk length. d_a is the dimension of one command, fourteen in the shown ALOHA interface, while H is the number of commands predicted together. Inputs may include a history rather than only one observation. The action at the same stored index is the first target under the verified simulation loader convention; another dataset may use a different alignment, so inspect acquisition and conversion code before applying a shift. Sample windows within episodes and avoid train/validation leakage from neighboring frames. End windows can be omitted or padded with an explicit mask; padding must not be mistaken for new valid control labels. ACT uses a conditional latent-variable model during training, but that architecture is secondary to the input/target relationship here. Original ACT conditions on images and joint measurements, without a language input. A later language-conditioned policy may add an instruction to the input; the dataset’s task metadata alone does not make ACT language-conditioned. An offline inspector can check these windows and shapes, but cannot establish closed-loop success, recovery, or real-world transfer.',
        [ACT, ('ACT · Data loader', 'https://github.com/tonyzhaozh/act/blob/main/utils.py'), ('LeRobot · Dataset format', 'https://huggingface.co/docs/lerobot/lerobot-dataset-v3')], 'Training data'))

    # 7. Collection and aggregation require an interface contract.
    b = text(75, 200, 'More scenes and robots broaden experience—and introduce incompatible conventions.', 31)
    entries = [(265, 'Robot teleoperation', 'DROID: a person drives the robot'),
               (389, 'Portable demonstrations', 'UMI: a tracked handheld gripper'),
               (513, 'Multi-robot aggregation', 'Open X-Embodiment: many datasets')]
    for y, head, detail in entries:
        b += text(75, y, head, 30, 700, color=BLUE) + text(75, y + 40, detail, 26)
        b += arrow(600, y + 7, 735, y + 7)
    b += rect(735, 235, 450, 335, PALE)
    b += text(760, 278, 'Keep the interface metadata', 29, 700)
    b += lines(760, 325, ['joint / tool-pose / velocity', 'units and coordinate frames', 'absolute / relative / delta', 'sensor and action alignment', 'robot identity + episode limits'], 26, gap=45)
    b += arrow(1185, 402, 1290, 402)
    b += _block(1290, 337, 235, 130, 'Train', ['compatible', 'representations'])
    b += line(75, 610, 1525, 610)
    b += text(75, 650, 'Numerical normalization', 28, 700, color=BLUE)
    b += text(75, 697, '(a − μ) / σ rescales coordinates.', 29)
    b += text(805, 650, 'Geometric frame conversion', 28, 700, color=TEAL)
    b += text(805, 697, 'Camera C → world W:', 26)
    b += mathml(1095, 660, 430, 67,
        '<mrow><msub><mi>p</mi><mi>W</mi></msub><mo>=</mo>'
        '<msub><mi>R</mi><mrow><mi>W</mi><mi>C</mi></mrow></msub><msub><mi>p</mi><mi>C</mi></msub><mo>+</mo>'
        '<msub><mi>t</mi><mrow><mi>W</mi><mi>C</mi></mrow></msub></mrow>', 32)
    b += takeaway('A common storage format does not make two action spaces equivalent.',
                  'LeRobot is a dataset format/library; DROID and Open X-Embodiment are data resources.')
    out.append(slide('More robot data also means more interfaces', b,
        'These are collection and aggregation examples, not three interchangeable datasets. DROID collects demonstrations on a common Franka platform in many environments. UMI uses a handheld gripper with visual tracking and carefully aligns observations/actions for deployment. Open X-Embodiment combines datasets from many embodiments. A common RLDS or LeRobot container does not by itself align physical meanings. A fourteen-dimensional joint target differs from a seven-dimensional tool/gripper command, even after scaling each coordinate to unit variance. The illustrated point-frame transform has both rotation and translation; transformation rules for rotations, relative displacements, twists, and forces require their own conventions. Numerical normalization changes coordinate scale, while geometric calibration relates frames. UMI’s relative trajectory representation reduces some embodiment-specific variation but still needs calibration, timing, and an execution interface. Some Open X-Embodiment training adaptations use per-dataset normalization without geometrically aligning all frames. Preserve metadata rather than treating a float array as self-describing. LeRobot’s numeric tables, videos, and episode metadata are storage/access machinery, not a universal robot action schema.',
        [('DROID · Dataset', 'https://arxiv.org/html/2403.12945v2'), UMI, ('Open X-Embodiment · §III–IV', 'https://arxiv.org/html/2310.08864v6')], 'Data at scale'))

    # 8. BC as conditional density learning, with its scope made explicit.
    b = text(75, 200, 'Demonstrations provide examples of what action to take under each observation.', 31)
    b += _block(75, 275, 350, 130, 'Demonstrations 𝒟', ['(oᵢ, Aᵢ) pairs'])
    b += arrow(425, 340, 550, 340)
    b += _block(550, 275, 310, 130, 'Policy πθ', ['train on those pairs'])
    b += arrow(860, 340, 1000, 340)
    b += text(1250, 264, 'πθ(A | o)', 34, 700, color=BLUE, anchor='middle')
    b += line(1030, 408, 1490, 408, MUTED, 2)
    b += path('M 1040 404 C 1090 404 1090 307 1135 307 C 1180 307 1180 401 1222 401 C 1280 401 1275 290 1326 290 C 1377 290 1370 402 1477 404', BLUE, 4)
    b += text(1255, 448, 'one action coordinate (illustration)', 24, color=MUTED, anchor='middle')
    b += mathml(120, 488, 1360, 105,
        '<mrow><munder><mo>min</mo><mi>θ</mi></munder><mspace width="0.6em"/>'
        '<msub><mi>𝔼</mi><mrow><mo>(</mo><mi>o</mi><mo>,</mo><mi>A</mi><mo>)</mo><mo>∼</mo><mi>𝒟</mi></mrow></msub>'
        '<mo>[</mo><mo>−</mo><mi>log</mi><mspace width="0.2em"/><msub><mi>π</mi><mi>θ</mi></msub>'
        '<mo>(</mo><mi>A</mi><mo>|</mo><mi>o</mi><mo>)</mo><mo>]</mo></mrow>', 43)
    b += text(75, 632, 'Fixed-variance Gaussian likelihood → squared-error regression.', 30)
    b += text(75, 690, 'Diffusion and flow policies use denoising or vector-field training objectives.', 29)
    b += takeaway('Behavioral cloning is the learning setup; the output distribution is a modeling choice.',
                  'This supervised imitation objective needs action labels, but no reward function.')
    out.append(slide('Behavioral cloning learns an action distribution', b,
        'For this audience, begin from conditional density estimation. Demonstrations supply observations and action labels, including whole chunks if that is the chosen output. The expectation is over stored demonstration pairs; a language-conditioned extension adds the instruction after the conditioning bar. Negative log likelihood is a common formulation for an explicitly parameterized policy. Under a fixed-variance Gaussian assumption, minimizing it reduces to squared-error regression up to constants and scale. Enlarging a deterministic network does not by itself turn a single Gaussian or a mean into a multimodal conditional distribution. The drawn density is an illustrative marginal of one action coordinate; the actual chunk distribution lives in H times d_a dimensions and must model correlations. Do not claim every imitation method directly optimizes tractable likelihood: ACT uses a variational latent-variable objective with reconstruction and KL terms, while diffusion and flow matching use their own training losses. These methods are within imitation learning, not alternatives to imitation learning as a category. Reinforcement learning changes the training signal toward cumulative reward and will be introduced later.',
        [CS224R, ('MIT · Imitation learning', 'https://underactuated.mit.edu/imitation.html')], 'Imitation learning'))

    # 9. Policy-induced covariate shift and the purpose of expert correction.
    b = text(75, 201, 'An imperfect action changes the next observation on which the policy must act.', 31)
    b += text(75, 250, 'Schematic state / observation space', 25, color=MUTED)
    b += path('M 130 345 C 350 300 725 305 1050 320', '#e8f0f5', 102)
    b += path('M 130 345 C 350 300 725 305 1050 320', BLUE, 3)
    for x, y in [(130,345),(435,315),(740,313),(1045,320)]:
        b += circle(x,y,13,'white',BLUE)
    b += text(575, 286, 'expert demonstrations', 27, 700, color=BLUE, anchor='middle')
    b += path('M 130 345 L 435 374 L 740 458 L 1045 527', WARM, 3)
    for x, y in [(435,374),(740,458),(1045,527)]:
        b += circle(x,y,13,'white',WARM)
    b += text(405, 432, 'small error', 25, color=WARM)
    b += text(780, 575, 'learner visits unfamiliar inputs', 27, 700, color=WARM, anchor='middle')
    b += line(1140, 237, 1140, 610)
    b += text(1190, 271, 'DAgger', 33, 700, color=BLUE)
    b += text(1190, 335, '1. Roll out learner', 27)
    b += lines(1190, 405, ['2. Ask expert for', '    actions at visited states'], 26, gap=38)
    b += lines(1190, 517, ['3. Aggregate labels', '    and retrain'], 27, gap=38)
    b += text(75, 661, 'The policy changes its input distribution:', 29)
    b += mathml(680, 625, 650, 66,
        '<mrow><msub><mi>d</mi><mtext>expert</mtext></msub><mo>(</mo><mi>o</mi><mo>)</mo><mo>≠</mo>'
        '<msub><mi>d</mi><mi>π</mi></msub><mo>(</mo><mi>o</mi><mo>)</mo></mrow>', 35)
    b += text(75, 707, 'Low held-out action loss alone does not establish reliable task completion.', 28)
    b += takeaway('Evaluate in rollouts: the policy’s actions select its future inputs.',
                  'DAgger adds expert labels on visited states; access to the expert and collection cost still matter.')
    out.append(slide('A small action error can change the next input', b,
        'The horizontal drawing is a schematic state/observation-space path, not a physical tool trajectory or a quantitative probability-density plot. The pale band represents the region sampled by expert demonstrations. A learner error can take the system into inputs that were rare or absent there, and further errors can compound. This differs from evaluating a predictor on a fixed, externally sampled test set. Explain DAgger operationally: run the current learner, query the expert for what should have been done at the visited states, add those labeled pairs to all previous data, and retrain. The original algorithm can mix expert and learner control during collection. Querying the expert for a label is distinct from always letting the expert take control; human-gated intervention is a related practical variant. Do not present worst-case horizon-squared bounds without their assumptions or claim that chunking eliminates distribution shift. Episode-level action loss is useful for debugging and selection, while native task success and recovery require closed-loop rollouts.',
        [DAGGER, CS224R], 'Closed-loop distribution shift'))

    # 10. Four explicit frames, so the interaction survives print as its last state.
    b = text(75, 196, 'Prediction horizon H = 6 commands.  Execute h = 2, then use a new observation.', 31)
    b += text(75, 237, 'Superscript = prediction time; commands for the same execution step can change.', 25, color=MUTED)
    scenes = []
    stage_messages = ['1. Observe o₀ and predict six future commands.',
                      '2. Execute the first two commands; obtain observation o₂.',
                      '3. Predict again from o₂; replace the unused suffix.',
                      '4. Repeat at o₄: new observations can change later commands.']
    for stage in range(4):
        s = ''
        for j in range(10):
            s += text(400+j*111+50, 291, str(j), 25, color=MUTED, anchor='middle')
        s += text(75, 291, 'Execution step', 27, 700)
        s += text(75, 354, 'Prediction from o₀', 27, 700)
        s += _action_cells(400, 315, 0, 6, executed=2 if stage >= 1 else 0, replaced=stage >= 2)
        s += text(75, 459, 'Prediction from o₂', 27, 700, color=INK if stage >= 2 else '#a6afb5')
        if stage >= 2:
            s += _action_cells(622, 420, 2, 6, executed=2 if stage >= 3 else 0, replaced=stage >= 3)
        s += text(75, 564, 'Prediction from o₄', 27, 700, color=INK if stage >= 3 else '#a6afb5')
        if stage >= 3:
            s += _action_cells(844, 525, 4, 6)
        s += rect(75, 601, 25, 25, '#e5f0ec', TEAL) + text(112, 622, 'executed', 24, color=TEAL)
        s += rect(295, 601, 25, 25, PALE) + text(332, 622, 'predicted', 24, color=BLUE)
        s += rect(520, 601, 25, 25, '#f1f2f3') + line(523, 622, 542, 605, MUTED, 1.5)
        s += text(557, 622, 'replaced', 24, color=MUTED)
        s += text(75, 659, stage_messages[stage], 27, 700, color=BLUE)
        scenes.append(s)
    b += steps('robot-chunk', scenes, y=675)
    b += takeaway('Prediction horizon H and execution horizon h answer different questions.',
                  'Larger h reduces model calls but delays new visual feedback; rapid joint feedback can remain active.')
    out.append(slide('Predict a chunk, execute a prefix, observe again', b,
        'Advance the four steps. First the model sees o_0 and predicts commands for steps zero through five. It executes only steps zero and one. At step two a fresh observation leads to a new chunk, so the unused commands from the old prediction can be replaced. The command subscript is its intended execution step; the parenthesized superscript is the observation/prediction step, so a_2^(0) and a_2^(2) need not have the same value. At step four the process repeats. Green cells have been executed; pale blue cells are future proposals; crossed gray cells are superseded proposals, not executed commands. H is the prediction length and h the execution prefix. They need not be equal. The drawing is logical timing and omits inference latency, queueing, and asynchronous implementation details discussed earlier. The policy can be visually open-loop during each prefix while its lower-level joint servo remains closed-loop. This is the receding-horizon execution pattern used by Diffusion Policy. ACT also studies action chunking, but its temporal ensembling variant queries at every step and combines overlapping predictions for the same execution time; it is not simply the exact prefix-playback algorithm drawn here. Larger h can lower model-call overhead and preserve temporal consistency while delaying visual correction. It is an engineering tradeoff, not a guarantee that longer or shorter is always better.',
        [DP, ACT], 'Action chunking'))

    # 11. Two different spaces: geometric command routes and one command coordinate.
    b = text(75, 198, 'A diffusion policy refines noise into an observation-conditioned action chunk.', 31)
    b += text(75, 249, 'Several target routes may be valid', 29, 700, color=BLUE)
    b += text(75, 288, 'Illustrative tool-position commands', 24, color=MUTED)
    b += circle(130, 437, 12, 'white', BLUE) + circle(510, 437, 12, 'white', BLUE)
    b += rect(280, 372, 100, 130, '#e7e9eb', MUTED)
    b += text(330, 405, 'obstacle', 24, color=MUTED, anchor='middle')
    b += path('M 143 437 C 190 437 190 327 280 327 L 380 327 C 465 327 455 437 497 437', TEAL, 4)
    b += path('M 143 437 C 190 437 190 548 280 548 L 380 548 C 465 548 455 437 497 437', BLUE, 4)
    b += line(143, 437, 497, 437, WARM, 2, '7 5')
    b += text(135, 586, 'The average can point into the obstacle.', 24, color=WARM)
    b += line(595, 238, 595, 596)
    b += text(655, 249, 'Repeated denoising, conditioned on oₜ', 29, 700, color=BLUE)
    plot_x = [660, 970, 1280]
    vals = [[35,150,62,172,40,136,83], [141,102,130,83,104,53,65], [153,143,121,96,75,61,57]]
    labels = ['Gaussian noise', 'Intermediate', 'Action chunk']
    for x, values, label in zip(plot_x, vals, labels):
        b += text(x+107, 313, label, 26, 700, anchor='middle')
        b += line(x, 520, x+210, 520, MUTED, 1.5)
        b += line(x, 332, x, 520, MUTED, 1.5)
        points = [(x+j*33, 334+v) for j,v in enumerate(values)]
        d = 'M ' + ' L '.join(f'{xx} {yy}' for xx,yy in points)
        b += path(d, BLUE, 3)
        for xx,yy in points:
            b += circle(xx, yy, 3, BLUE, BLUE)
        b += text(x+105, 552, 'command offset j', 23, color=MUTED, anchor='middle')
    b += arrow(880, 424, 950, 424) + arrow(1190, 424, 1260, 424)
    b += text(1090, 594, 'One coordinate of the H × dₐ command array', 25, color=MUTED, anchor='middle')
    b += text(75, 651, 'Train: corrupt demonstrated action chunks and learn to predict the added noise.', 29)
    b += text(75, 704, 'Sample: repeatedly denoise a random chunk while conditioning on the observation.', 28)
    b += takeaway('Diffusion Policy generates commands; a controller and the world determine the outcome.',
                  'Denoising iterations are model computation, not elapsed robot execution time. Diagrams are illustrative.')
    out.append(slide('Generate a continuous action chunk', b,
        'The left sketch motivates modeling a distribution: tool-position command routes can go around either side of an obstacle, whereas averaging coordinates can create an invalid route. It is an illustrative geometric example, not a claim that multimodality explains every gain of generative policies or that the command route will be tracked perfectly. On the right, each plot shows only one coordinate of a proposed H-by-d_a array as a function of future command offset j. Moving from plot to plot is a denoising iteration, not the robot moving forward in physical time. The jagged input is sampled noise, and the final curve is an action sequence; it is not a forecast of measured robot states. The generic diffusion-policy training procedure perturbs demonstrated chunks at sampled noise levels and trains a conditioned network to predict the noise. Sampling applies repeated learned denoising updates, with a schedule and possible stochastic terms. Observation features condition generation rather than being generated as future images. The exact update and noise parameterization vary by sampler, so the schematic curves are not numerical samples from the published system. Diffusion Policy can model temporal correlations and several valid behaviors while using receding-horizon execution. Flow matching will later make another continuous generation objective explicit.',
        [DP, CS224R], 'Generative action policies'))
    assert len(out) == 11
    return out
