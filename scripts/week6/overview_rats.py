"""Three diagram-led RATs slides for the separate Week 6 overview.

The practice diagram and skill record are paraphrased teaching schematics.
The final chart reproduces Table 4 values, not a new robot experiment.
"""
from .common import *

PAPER = ('RATs · paper', 'https://playful-rats.github.io/assets/papers/rats-paper.pdf')
CODE = ('RATs · release code', 'https://github.com/Playful-RATs/RATs')
LIBRARY = ('RATs · skill record', 'https://github.com/Playful-RATs/RATs/blob/main/skill_library/molmospaces_formula_play.json')
SECTION = 'Learning reusable robot programs'


def _practice():
    b = text(75, 210, 'Robotics Agent Teams (RATs): practice before test', 30, 700, color=BLUE)
    b += text(1525, 210, 'Foundation-model weights stay fixed', 27, color=MUTED, anchor='end')

    # Three roles, with a shared persistent store beneath the interaction loop.
    for x, title in ((75, 'Propose a goal'), (595, 'Write and run code'), (1115, 'Check the outcome')):
        b += text(x, 299, title, 32, 700, color=BLUE)
        b += line(x, 319, x + 410, 319)

    # An intentionally schematic scene, not a recorded rollout.
    b += rect(100, 355, 300, 113, '#fafcfd', LINE)
    b += rect(130, 374, 240, 71, PALE, BLUE)
    b += line(209, 410, 291, 410, BLUE, 7)
    b += arrow(280, 480, 195, 480, TEAL)
    b += text(75, 526, '“Pull this drawer open”', 29)
    b += arrow(470, 411, 565, 411)

    b += lines(595, 385, ['Perception → grasp → pull',
                          'Generated Python',
                          'calls robot APIs'], 29, gap=53)
    b += arrow(1010, 411, 1090, 411)

    b += lines(1115, 385, ['Goal + step verification',
                           'Localize the failure',
                           'Revise and retry'], 28, gap=53)
    b += path('M 1330 516 V 557 H 795 V 531', WARM, 2.5)
    b += arrow(795, 531, 795, 512, WARM)

    b += line(75, 602, 1525, 602)
    b += text(125, 652, 'Persistent memory', 31, 700, color=TEAL)
    b += text(460, 652, 'Code skills  +  failure lessons  +  reliability', 31)
    b += path('M 1330 557 V 580 H 1500 V 650', TEAL, 2.5)
    b += arrow(1500, 650, 1450, 650, TEAL)
    b += path('M 112 671 V 708 H 80 V 549', TEAL, 2.5)
    b += arrow(80, 549, 80, 540, TEAL)
    b += text(220, 707, 'Memory guides the next practice goal and generated program', 27, color=TEAL)
    b += takeaway('At test time: frozen library + online planning, verification, and retries.',
                   'The learned library is fixed; the execution agent still makes decisions.')
    notes = (
        'RATs stands for Robotics Agent Teams. This drawing is a paraphrased teaching schematic, '
        'not a recorded drawer experiment. It groups the paper’s proposer, execution, and memory '
        'teams into a readable loop. The proposer uses scene context and accumulated experience '
        'to choose a goal before downstream benchmark instructions arrive. In the released formula '
        'scorer, object–skill novelty is combined with a competence-frontier term, 4c(1−c), plus '
        'retry and recent-failure adjustments. The planner selects skills, the writer generates '
        'Python, and existing perception/control APIs execute it. Goal and per-step verifiers, '
        'failure diagnosis, bounded retries, and sometimes isolated sub-action practice help revise '
        'attempts. Successful behavior can yield parameterized helpers; failures yield lessons and '
        'candidate helpers for later validation. There is no foundation-model gradient update or '
        'RL training in this process. Neural perception/grasp modules and robot controllers remain '
        'part of the supplied execution stack. This is autonomous code/memory improvement, not '
        'evidence that the foundation model recursively improves its own learning algorithm. '
        'At test time the skill library is frozen. The release supports direct helper injection in '
        'native RATS, whereas cross-environment planner-mode transfer can show source code for '
        'the writer to copy or adapt to the destination robot API. Sources: paper §§3.1–3.5; '
        'repository README, agents/curiosity_scoring.py, docs/play.md, and docs/cross-environment.md.'
    )
    return slide('RATs: self-directed practice with code policies', b, notes,
                 (PAPER, CODE), section=SECTION)


def _record():
    b = text(75, 210, 'Paraphrased schematic · released MolmoSpaces skill record', 28, color=MUTED)

    # A single document-like record, rather than a collection of repeated cards.
    b += rect(75, 242, 1450, 465, 'white', LINE)
    b += text(105, 292, 'grasp_and_pull_horizontal_handle', 34, 700, color=BLUE,
              attrs='font-family="ui-monospace,Menlo,Consolas,monospace"')
    b += text(105, 342, '(target_pos, pull_direction, pull_distance = 0.15 m)', 29,
              attrs='font-family="ui-monospace,Menlo,Consolas,monospace"')
    b += line(75, 372, 1525, 372)
    b += line(905, 372, 905, 707)

    b += text(105, 421, 'Function', 29, 700, color=BLUE)
    # Each step represents several lines in the original helper.
    stages = [(105, 'Approach'), (295, 'Grasp'), (485, 'Pull'), (675, 'Release')]
    for i, (x, name) in enumerate(stages):
        b += text(x + 80, 525, name, 29, 700, anchor='middle', color=TEAL)
        if i < 3:
            b += arrow(x + 163, 516, x + 185, 516, TEAL)
    b += lines(105, 610, ['Compose pose and gripper API calls',
                          'into a parameterized Python helper'], 28, gap=45)

    b += text(945, 421, 'Preconditions', 29, 700, color=BLUE)
    b += lines(945, 469, ['Reachable target', 'Horizontal handle'], 28, gap=45)
    b += line(945, 544, 1485, 544)
    b += text(945, 589, 'Recorded evidence', 29, 700, color=BLUE)
    b += text(945, 648, '3 / 3', 44, 700, color=TEAL)
    b += text(1085, 646, 'successful uses · “verified”', 27)
    b += takeaway('A reusable skill includes code, conditions, and evidence about its use.')
    notes = (
        'The function name, default pull distance, preconditions, and 3-of-3 usage record come '
        'from the released MolmoSpaces JSON library. The layout and four-stage behavior summary '
        'are paraphrased; the signature is simplified and the slide is not executable source. '
        'The actual helper takes target_pos and pull_direction arrays, sets the direction’s '
        'vertical component to zero, constructs a horizontal approach frame, opens the gripper, '
        'moves to a pre-grasp pose and the target, closes, pulls, opens, and retreats. The source '
        'preconditions require a reachable target and a horizontal handle suitable for vertical '
        'finger closure. The record identifies learning iteration 43, usage_count=3, '
        'success_count=3, and tier=verified. These are accumulated library statistics, not a '
        'controlled success-rate estimate or proof of robustness on new handles. “Verified” is '
        'the system’s reliability tier; it is not a safety guarantee. Other skill-record fields '
        'include effects, API primitives, dependencies, and provenance. Later uses can update '
        'reliability or deprecate a helper. The paper’s terminology “learning a skill” here '
        'means creating and retaining a code composition; it does not imply fitting new neural '
        'weights for the grasp or vision model. Source: skill_library/molmospaces_formula_play.json '
        'entry grasp_and_pull_horizontal_handle and paper Appendix A.3.'
    )
    return slide('What RATs stores after practice', b, notes,
                 (LIBRARY, PAPER), section=SECTION)


def _ablation():
    b = text(75, 210, 'LIBERO-PRO success rate (%)', 30, 700, color=BLUE)
    b += text(75, 252, '60 simulated tasks', 26, color=MUTED)
    b += rect(735, 189, 27, 27, '#c2cbd1', 'none')
    b += text(775, 212, 'No play', 27)
    b += rect(1008, 189, 27, 27, TEAL, 'none')
    b += text(1048, 212, 'Library from 50 play iterations', 27)

    # Table 4: compare the library within each executor, with the unequal
    # trial count across executor groups stated below the axis.
    baseline, scale = 586, 6.0
    for value in (0, 25, 50):
        y = baseline - value * scale
        b += line(175, y, 1465, y, '#dbe2e6', 1.5)
        b += text(145, y + 9, str(value), 26, anchor='end', color=MUTED)
    b += line(175, baseline, 1465, baseline, MUTED, 1.8)
    bars = [(330, 23.2, '#c2cbd1'), (535, 32.3, TEAL),
            (1010, 36.3, '#c2cbd1'), (1215, 44.3, TEAL)]
    for x, value, fill in bars:
        y = baseline - value * scale
        b += rect(x, y, 125, baseline - y, fill, 'none')
        b += text(x + 62.5, y - 17, f'{value:.1f}', 34, 700, anchor='middle', color=INK)

    b += text(495, 635, 'CaP-Agent0 (baseline)', 30, 700, anchor='middle', color=BLUE)
    b += text(1175, 635, 'RATs execution team', 32, 700, anchor='middle', color=BLUE)
    b += text(495, 681, '10 trials per task', 27, anchor='middle', color=MUTED)
    b += text(1175, 681, '5 trials per task', 27, anchor='middle', color=MUTED)
    b += takeaway('Library gain: +9.1 pp with CaP-Agent0; +8.0 pp with RATs execution.',
                   'pp = percentage points. Compare the pair within each execution system.')
    notes = (
        'The four values are reproduced from Table 4 of Playful Agentic Robot Learning. '
        'All curious-play libraries in this ablation come from 50 play iterations. Hold the '
        'test-time executor fixed within each pair: CaP-Agent0 rises from 23.2% with no play '
        'to 32.3% with curious-play skills, a 9.1 percentage-point increase. RATS execution '
        'rises from 36.3% to 44.3%, an 8.0-point increase. The fuller executor includes '
        'structured planning, verification, diagnosis, and retries, so it changes more than '
        'the available library. Trial counts differ between the executor groups: ten per '
        'task for CaP-Agent0 and five for RATS execution. These bars are descriptive results, '
        'not an equal-latency or equal-token-budget comparison, and no uncertainty intervals '
        'are supplied here. Random-play rows, omitted for legibility, are 25.0% for '
        'CaP-Agent0 and 38.3% for RATS execution. The separate headline Table 1 reports '
        '43.8% for RATS versus 23.2%, yielding +20.6 pp; do not mix that value with the '
        '44.3% Table 4 ablation. The paper’s main evaluation has 60 LIBERO-PRO tasks; '
        'release defaults and model examples differ from some paper settings. These '
        'controlled code-agent comparisons are more informative about the skill library '
        'than a broad claim that code policies outperform all VLAs. Evaluation remains '
        'mainly simulated. The appendix adds physical tasks beyond the two main-table '
        'tasks, but the quantitative real-world set is still small and transferred skills '
        'can hurt individual tasks. Source: paper Table 4 and §§4.4–4.5; limitations on p. 9.'
    )
    return slide('Separate skill reuse from test-time execution', b, notes,
                 (('RATs · Table 4', PAPER[1] + '#page=8'),), section=SECTION)


def get_slides():
    return [_practice(), _record(), _ablation()]
