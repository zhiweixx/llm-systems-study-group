# Robot learning overview — VLM, VLA, WAM, and code-based self-improvement

30-slide tutorial, 26 October 2026. The existing 52-slide Week 6 deck is preserved. Slides 1–14 teach observations, robot action arrays, VLMs, VLAs, π0.5, and world-action models. Slides 15–18 introduce executable policies and persistent improvement. Slides 19–27 examine RAX, RATs, COAP and the 123-round RSI report. Slides 28–30 compare evidence and apply the framework.

Original teaching diagrams use source-linked primary literature. Numeric action arrays and reported results are marked as actual or illustrative. Dataset recordings and paper figures retain attribution. Results across different robot tasks and protocols are not directly comparable. No robot experiments were run for this deck.

Use arrow keys to navigate, Slides for the overview, and Notes for context and sources. Slide 3 has a recorded simulator video. Slide 9 has a step-through action-chunk demonstration. Print / PDF uses a video poster and the final interaction frame. All essential assets are embedded in the HTML.

Design references: [Flex-π reading-group deck](https://docs.google.com/presentation/d/1QmGqrGiUL1zaF2VGfYqd5bAaC1zxKeNoK7UD93NECf4/edit) and [How to Train and Fine-tune VLMs](https://docs.google.com/presentation/d/1-6qBQlqvwpGgWTgoPDTuSEAj17XNFFYD1agJAqMHdzk/edit). Both were read in native form and all slides inspected as rendered PDF pages. The overview follows their figure-led organization and simple academic layout, retaining the study group blue accent and omitting institutional branding.

## 1. Robot learning and code-based agents

**Section: Introduction**

This 30-slide tutorial starts with observable interfaces, then introduces neural action policies, world-action models, and program-based control. Recent RAX, RATs, COAP and the 123-round RSI report are presented as different systems with different evaluation protocols. They are not a single leaderboard. Teaching diagrams are original. The robot image is from Russ Tedrake’s MIT Robotic Manipulation materials. The layout follows the supplied reading-group references: white background, clear title rule, one large diagram or evidence item per page, and short annotations. The existing 52-slide Week 6 deck is retained separately.

- [MIT: pick and place](https://manipulation.mit.edu/pick.html)

## 2. Four interfaces to keep separate

**Section: Introduction**

These are useful interfaces rather than mutually exclusive categories. VLM is broad: image–text alignment models and generative multimodal models both qualify. VLA specifies a learned vision/language-conditioned action policy. WAM is an evolving family that couples action learning with a representation of future world evolution; models differ in whether future prediction is required at inference. Code as Policy describes an executable program as the policy representation and may call a VLM, VLA, planner, or classical controller. RSI describes a process for modifying a system, not a fifth action representation.

- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [UNITAS §3](https://arxiv.org/html/2610.12099v1#S3)
- [Code as Policies](https://code-as-policies.github.io/)

## 3. Watch one observation–action loop

**Section: Foundations**

This is a recorded ALOHA simulated insertion demonstration from the LeRobot dataset. It is not an autonomous rollout of one of the models discussed later. Two robot arms must align a peg and socket. The stored episode has 500 records at nominal 50 Hz; the embedded playback is sampled at 10 Hz. A high-level policy selects desired targets, a lower-level controller tracks them, and new images and robot measurements close the loop. A target is different from the realized next state. Watch for the need to observe alignment before insertion.

- [ALOHA dataset](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)
- [ACT / ALOHA](https://arxiv.org/abs/2304.13705)

## 4. What does a robot action actually contain?

**Section: Robot data and actions**

The first seven coordinates are the six left-arm revolute-joint targets and the left normalized gripper opening. The next seven repeat that ordering for the right arm. sim_env.py documents these as absolute joint position commands, not joint increments, tool coordinates, or torques. constants.py supplies the named joint order. Revolute joints use radians in the simulation. Gripper normalization is an affine mapping relative to calibration endpoints, and it is not clipped by the shown normalization function: the actual first right-gripper target is approximately −0.03124. Thus do not label every coordinate in the array as radians or promise a strict [0,1] bound. The stored target is associated with nominal frame time, not verified wall-clock time of a human action.

- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Action interface](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)
- [ALOHA · Joint order](https://github.com/tonyzhaozh/act/blob/main/constants.py)

## 5. Robot learning starts with aligned observations and actions

**Section: Foundations**

The field shapes are from the actual ALOHA simulation dataset; the code block is a readable schema, not a literal file serialization. Each action is a desired joint/gripper configuration and state records the measured configuration. A training window pairs the current observation with H consecutive command vectors. A language-conditioned model may also receive the task instruction, but original ACT is not language conditioned. Demonstrations can be recorded from teleoperation, where a person moves leader arms to supply follower-arm commands. Behavior cloning optimizes an action prediction objective on these aligned records. At deployment the recorded target is unavailable. Split complete episodes and scenes for evaluation; splitting adjacent frames can leak nearly identical observations.

- [ALOHA dataset](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)
- [ACT / ALOHA](https://arxiv.org/abs/2304.13705)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 6. VLMs can match, describe, and answer

**Section: VLM → VLA**

CLIP learns image and text encoders from paired data, with a contrastive objective that favors matching pairs over alternatives. A generative VLM such as PaliGemma predicts text conditioned on visual embeddings and a prompt. The question and answer are an illustrative teaching example, not a measured prediction. These two interfaces can support retrieval, open-vocabulary perception, or task reasoning. The architecture on the following slide follows a generative VLM, which is one common starting point for a VLA; it does not describe every vision-language model.

- [CLIP](https://arxiv.org/abs/2103.00020)
- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 7. Vision–language–action models learn robot commands

**Section: VLM → VLA**

Vision–language–action models condition a learned action policy on images and language, often also robot proprioception. The diagram is generic: visual features are connected to an LLM-derived backbone, and an action output is trained using robot data. RT-2 represents robot actions as discrete tokens; OpenVLA adapts a pretrained VLM to action prediction. A flow-based VLA adds a continuous action expert. Robot supervision supplies the physical grounding that image–text pretraining alone does not establish: which coordinates, scale, and timing lead to successful motion. The actual robot controller still tracks the generated targets.

- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)

## 8. Actions can be generated as tokens or continuous arrays

**Section: VLM → VLA**

The token IDs and continuous arrays are illustrative, not model outputs. Autoregression is a valid robot-policy design: discrete actions are trained with a token prediction objective. FAST compresses action chunks before tokenization. In flow matching, training pairs a demonstrated normalized action array A with noise ε. An interpolation Xτ=(1−τ)ε+τA provides the input, and A−ε is a supervised direction. At inference a Transformer predicts successive refinements starting from noise, conditioned on the observation. The refinement index τ is not robot time: each array already contains a sequence of future commands. Noise and intermediate arrays are not sent to motors. Flow can represent multiple possible actions, while simple squared-error regression may average them. Autoregressive models can also represent multimodality; flow is not the only valid choice.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [FAST](https://arxiv.org/abs/2501.09747)

## 9. Predict a chunk, execute a prefix, observe again

**Section: Robot data and actions**

This interaction is a toy receding-horizon schedule using the verified fourteen-dimensional robot interface. The model predicts H=6 commands, then the runtime executes h=2 commands at 50 Hz, covering forty milliseconds of playback. Every cell is an entire fourteen-dimensional target, not a single scalar or a model token. Rows distinguish predictions made from different observations; columns share the same robot execution index. Crossed-out gray cells are old predictions that are discarded and replaced by a new row. Green means executed. A controller can still use rapid feedback while following each target. The diagram omits inference and communication latency; real deployments must schedule targets with those delays in mind. H and h here are pedagogical choices, not the original ACT execution algorithm or its temporal-ensembling rule.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)

## 10. π0.5 connects a semantic subtask to a motor action chunk

**Section: VLM → VLA**

π0.5 extends the π0 family with heterogeneous co-training and hierarchical inference. Given observations and a broad instruction, it produces a subtask in language, then conditions its continuous action expert on that subtask and robot observations/state. The published system outputs 50-step low-level chunks, corresponding to one second at 50 Hz. For the mobile platforms these include arm, gripper and torso targets plus base velocities, not just joint positions. The full model supports both discrete text generation and flow matching. Pretraining uses FAST-encoded action tokens; post-training adds the continuous action expert. Its main motivation is generalization to previously unseen homes; it does not imply that arbitrary new physical skills appear without robot data. The diagram omits detailed attention masks and action-expert internals.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [π0.5 explanation](https://www.pi.website/blog/pi05)

## 11. π0.5 learns from more than robot action labels

**Section: VLM → VLA**

The paper mixes data from different robots/environments with high-level task annotations and web vision-language tasks. These are distinct supervision sources, not evidence that unlabelled internet video alone supplies exact joint targets. The diagram groups the mixture into three teaching categories; the original recipe has finer categories, including verbal-instruction data. Images are credited crops of π0.5 Figure 4. Joint training lets shared representations learn semantics and physical behaviors, but the embodiment-specific action interface must still be supplied.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 12. WAM: learn actions together with future world changes

**Section: World-action models**

World-action model is a family name rather than a single standardized architecture. A policy learns to choose actions; a world model learns how observations or states evolve, often conditioned on an action. A WAM couples these capabilities in training or generation. This picture shows a conceptual shared model and is not a claim that every WAM factorizes the same way. Future prediction can supply a physical learning signal or support planning, but not every deployment samples a future or searches action candidates. Some models retain action-only inference. A visually plausible future is not by itself evidence of accurate contact dynamics.

- [UNITAS §3](https://arxiv.org/html/2610.12099v1#S3)
- [MIT: pick and place](https://manipulation.mit.edu/pick.html)

## 13. A concrete WAM: predict motion in 3D

**Section: World-action models**

UNITAS couples a point-dynamics expert and an action expert. The point-dynamics branch represents gripper and scene displacement trajectories in a metric frame; the action branch outputs executable end-effector commands. Scene prediction conditions on a supplied or generated gripper trajectory. Policy-only inference can omit both future point-flow branches. Therefore this model should not be drawn as necessarily doing imagined-rollout search before each robot action. The central panel of Figure 1 is displayed with the side panels clipped, without modifying its content. Credit: Wang et al., UNITAS, 2026, CC BY 4.0. The project offers an interactive 3D viewer; its link is in the footer.

- [UNITAS §3](https://arxiv.org/html/2610.12099v1#S3)
- [Interactive 3D demo](https://unitas-wam.github.io/#explore)

## 14. Reasoning and motor control run at different timescales

**Section: Control architectures**

System 1/2 and A/B are informal architecture labels whose exact mapping varies. Here slow means less frequent semantic or planning updates, while fast means robot action generation and control feedback. A VLM may choose a goal for a VLA; generated code may instead select geometric targets for a conventional controller. The measured robot state closes the fast loop, and task progress can close a slower loop. One model can support both roles, as π0.5 illustrates. A system can also use separate models or machines. The slide intentionally avoids universal frequency claims.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [Code as Policies](https://code-as-policies.github.io/)

## 15. Code as Policies makes a program the policy

**Section: Code as Policy**

The photo is from the original Code as Policies project; the short program is authored pseudocode, not a claim about its exact API. The model writes code that composes provided perception/control operations, arithmetic, conditions and loops. A program can re-observe, branch and recover rather than simply replay commands. What the model generates depends strongly on the available API: pick_and_place hides more robotics than move_pose. CaP systems can use learned perception and pre-existing manipulation primitives. Separate the code-writing call from execution: some systems call the model repeatedly online, while others freeze and run a completed program.

- [Code as Policies](https://code-as-policies.github.io/)

## 16. A program can produce precise numeric actions

**Section: Code as Policy**

The LLM does not need to memorize all joint coordinates. It can write a computation that obtains a measured object position, applies a geometric offset, solves inverse kinematics and passes the result to a robot API. The toy code uses metres in the robot-base frame and assumes a chosen orientation R. Actual perception may use RGB-D, calibration, known geometry or a learned model. IK may have multiple solutions or no feasible solution, so practical execution adds collision, joint-limit and reachability checks. The low-level controller converts desired positions into actuator behavior. This is an explanatory program, not runnable robot code.

- [MIT: pick and place](https://manipulation.mit.edu/pick.html)
- [Code as Policies](https://code-as-policies.github.io/)

## 17. What is stored in a skill library?

**Section: Code as Policy**

The function is illustrative and deliberately exposes a feedback check after grasping. A stored skill may contain source code, parameters, preconditions, effects, dependencies, demonstrations, run logs and reliability estimates. Different papers use skill for different artifacts: a neural policy, a callable program, or a prompt/instruction document. This deck labels the artifact explicitly. RAX and RATs can retain program knowledge across attempts, but reusable source may still need adaptation to a new robot API or scene. A single successful execution does not establish a universal precondition.

- [RAX](https://arxiv.org/abs/2610.22496)
- [RATs](https://playful-rats.github.io/)

## 18. RSI claims need an explicit improvement loop

**Section: Code and self-improvement**

In recent robotics work, self-improvement often means repeated program synthesis, execution, feedback and retention of useful artifacts while the coding model stays fixed. Calling this recursive does not establish that the foundation model improves its own weights or that its own learning procedure is modified. Distinguish within-episode feedback control from between-episode persistent adaptation. Also distinguish improvement of a task program from changing the harness that judges and accepts it. The next case studies locate these boundaries and examine evidence rather than assuming a shared meaning of RSI.

- [RAX](https://arxiv.org/abs/2610.22496)
- [RATs](https://playful-rats.github.io/)
- [Wang: 123 rounds](https://arxiv.org/abs/2609.31760)

## 19. RAX: explore, test, and revise a robot program

**Section: Robot agents and program revision**

Read the upper arrows first. The coding agent writes a Python program that calls the same robot tools available during interactive exploration. The green return path is the program’s sensor feedback; the blue return path is the slower software revision process. These are different loops. A frozen main LLM can still produce an improving system because source code and external memory change. In RAX, compilation means turning successful decisions into executable software, not neural-network distillation. The general workflow begins with interactive tool use, then tests a program on new layouts and edits its perception, action logic, checks, and recovery. Task-specific programs belong to the learned skill library; instruction skills describe how the agent should use the interface and record experience. Tool APIs are the executable boundary underneath both. Programs may invoke a VLM where needed, so running without the main coding agent does not universally mean no model calls. Real-robot tests require operator scene resets; intervention counts as debugging rather than unattended success. Sources: repository overview, https://github.com/robot-agent-exploration/rax; autonomous workflow, https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/agent_skills/task/autonomous-control/SKILL.md. The simulator’s offline container refers to network isolation, not offline learning: robot interaction and model access remain available through relays. Architecture: https://github.com/robot-agent-exploration/rax/blob/9d03ba60cee50f5a39c0031774b337dc1ff75245/docs/ARCHITECTURE.md. Teaching interpretation: this is program improvement from experience. It does not demonstrate recursive self-improvement of the foundation-model weights.

- [RAX · workflow](https://github.com/robot-agent-exploration/rax)
- [Autonomous control](https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/agent_skills/task/autonomous-control/SKILL.md)

## 20. Air fryer: geometry becomes numeric actions

**Section: Robot agents and program revision**

The image is the authors’ Figure 3, reproduced from https://arxiv.org/html/2610.22496v1/articulation.png; RAX / Huang et al., 2026, CC BY 4.0. It depicts the fitted articulated object and planned hand poses, not a learned video world model. The left panel labels the fixed body, drawer, and pull axis; the right panel shows grasp, pull, release, and close poses. Explain the numeric pathway on the right. geom.py reads depth, intrinsics K, and camera-to-base transform T, back-projects valid depth pixels, and transforms their points into the robot base frame. Depth is converted to metres. Source: https://github.com/robot-agent-exploration/skills-lib/blob/f1845de5c915b9e16590b86fbb64e957f020f5a1/skills/air-fry-pan/code/air_fry_pan/geom.py. The released perception code uses color/height filters and geometric fits for this specific white fryer, dark handle, and yellow banana. This task’s autonomous program does not require a runtime learned detector. Source: https://github.com/robot-agent-exploration/skills-lib/blob/f1845de5c915b9e16590b86fbb64e957f020f5a1/skills/air-fry-pan/code/air_fry_pan/perception.py. The controller searches inverse-kinematics candidates and rejects insufficient joint margin or clearance. Its pulling motion advances along the measured handle direction in 3-cm increments. Here d is drawn as a horizontal unit vector embedded in the base frame, and p is the hand target in metres; this equation explains the implementation rather than quoting a measured pose. The 0.025-m width and 50-N force are actual handle-grasp settings, not a claim that the measured grasp width equals its target. Source: https://github.com/robot-agent-exploration/skills-lib/blob/f1845de5c915b9e16590b86fbb64e957f020f5a1/skills/air-fry-pan/code/air_fry_pan/controller.py. Motion wrappers send position/quaternion or joint targets and gripper commands through MCP. The vendor controller generates physical motion; the LLM does not emit motor torques. FR3 uses metres and [x,y,z,w] quaternions, targeting the point between the fingers. Interface: https://github.com/robot-agent-exploration/rax/blob/8858d7f3e34eff382abae738cf0578d2726a9a0a/agent_skills/control/franka/robot-interface/SKILL.md. Generalization is bounded: this was one rig/day with specific colors, dimensions and tested placements. Four later runs were clean unattended successes, while earlier runs included a camera collision and interventions. The published skill entry remains marked draft; transferring it requires calibration and fresh checks. Evidence and limits: https://github.com/robot-agent-exploration/skills-lib/blob/f1845de5c915b9e16590b86fbb64e957f020f5a1/skills/air-fry-pan/SKILL.md.

- [RAX · Fig. 3 (CC BY 4.0)](https://arxiv.org/html/2610.22496v1#S3.F3)
- [Released air-fryer code](https://github.com/robot-agent-exploration/skills-lib/tree/f1845de5c915b9e16590b86fbb64e957f020f5a1/skills/air-fry-pan)

## 21. RoboDojo: explore, freeze, then evaluate

**Section: Robot agents and program revision**

This is one controlled task comparison from RAX Table 2, not the full RoboDojo benchmark. Both runs use the same task interface and differ in exploration budget. Plug in charger has 50 evaluation layouts and binary scoring: 26% versus 94% after budgets of four versus eight exploration episodes. Rehearsals consume that budget. The code bundle is frozen and run without the coding agent, edits, or retries on unseen layouts; evaluation results are not returned to the explorer. The explorer receives RGB-D, calibration, the task rubric, a completion image, and tool access. These are paper-reported results, not our reproduction. Do not infer uncertainty intervals or generalize monotonic improvement to every task. Other paper results use different protocols: BEHAVIOR is mostly agent-in-the-loop, including cross-test review reports on three tasks; HumanoidArena’s main setting supplies three action demonstrations. The +37.4 headline averages percentage-point gains across different benchmarks. The 241× air-fryer claim concerns inference overhead, excluding robot execution time. Sources: RAX §§5.2.1, 5.3.3, 10.3, Table 2 and Table 13; https://arxiv.org/html/2610.22496v1. The current usage guide exposes observation profile, demonstration context, rubric access, budget, layout split, review feedback, and evaluation mode as configuration choices. Treat these as experimental conditions; current defaults are not a full reconstruction of the paper’s historical runs. The formal code mode and formal agent mode are distinct. Source: https://github.com/robot-agent-exploration/rax/blob/9d03ba60cee50f5a39c0031774b337dc1ff75245/docs/USAGE.md. Teaching point: more interaction can help an agent construct a more reliable program, but the experiment must separate acquisition experience from evaluation and report the observation/action interface. This task-level plot avoids comparing heterogeneous system scores as though they measured one common capability.

- [RAX · Table 2 + protocol](https://arxiv.org/html/2610.22496v1#S5.SS2.SSS1)
- [Evaluation modes](https://github.com/robot-agent-exploration/rax/blob/9d03ba60cee50f5a39c0031774b337dc1ff75245/docs/USAGE.md#4-evaluation)

## 22. RATs: self-directed practice with code policies

**Section: Learning reusable robot programs**

RATs stands for Robotics Agent Teams. This drawing is a paraphrased teaching schematic, not a recorded drawer experiment. It groups the paper’s proposer, execution, and memory teams into a readable loop. The proposer uses scene context and accumulated experience to choose a goal before downstream benchmark instructions arrive. In the released formula scorer, object–skill novelty is combined with a competence-frontier term, 4c(1−c), plus retry and recent-failure adjustments. The planner selects skills, the writer generates Python, and existing perception/control APIs execute it. Goal and per-step verifiers, failure diagnosis, bounded retries, and sometimes isolated sub-action practice help revise attempts. Successful behavior can yield parameterized helpers; failures yield lessons and candidate helpers for later validation. There is no foundation-model gradient update or RL training in this process. Neural perception/grasp modules and robot controllers remain part of the supplied execution stack. This is autonomous code/memory improvement, not evidence that the foundation model recursively improves its own learning algorithm. At test time the skill library is frozen. The release supports direct helper injection in native RATS, whereas cross-environment planner-mode transfer can show source code for the writer to copy or adapt to the destination robot API. Sources: paper §§3.1–3.5; repository README, agents/curiosity_scoring.py, docs/play.md, and docs/cross-environment.md.

- [RATs · paper](https://playful-rats.github.io/assets/papers/rats-paper.pdf)
- [RATs · release code](https://github.com/Playful-RATs/RATs)

## 23. What RATs stores after practice

**Section: Learning reusable robot programs**

The function name, default pull distance, preconditions, and 3-of-3 usage record come from the released MolmoSpaces JSON library. The layout and four-stage behavior summary are paraphrased; the signature is simplified and the slide is not executable source. The actual helper takes target_pos and pull_direction arrays, sets the direction’s vertical component to zero, constructs a horizontal approach frame, opens the gripper, moves to a pre-grasp pose and the target, closes, pulls, opens, and retreats. The source preconditions require a reachable target and a horizontal handle suitable for vertical finger closure. The record identifies learning iteration 43, usage_count=3, success_count=3, and tier=verified. These are accumulated library statistics, not a controlled success-rate estimate or proof of robustness on new handles. “Verified” is the system’s reliability tier; it is not a safety guarantee. Other skill-record fields include effects, API primitives, dependencies, and provenance. Later uses can update reliability or deprecate a helper. The paper’s terminology “learning a skill” here means creating and retaining a code composition; it does not imply fitting new neural weights for the grasp or vision model. Source: skill_library/molmospaces_formula_play.json entry grasp_and_pull_horizontal_handle and paper Appendix A.3.

- [RATs · skill record](https://github.com/Playful-RATs/RATs/blob/main/skill_library/molmospaces_formula_play.json)
- [RATs · paper](https://playful-rats.github.io/assets/papers/rats-paper.pdf)

## 24. Separate skill reuse from test-time execution

**Section: Learning reusable robot programs**

The four values are reproduced from Table 4 of Playful Agentic Robot Learning. All curious-play libraries in this ablation come from 50 play iterations. Hold the test-time executor fixed within each pair: CaP-Agent0 rises from 23.2% with no play to 32.3% with curious-play skills, a 9.1 percentage-point increase. RATS execution rises from 36.3% to 44.3%, an 8.0-point increase. The fuller executor includes structured planning, verification, diagnosis, and retries, so it changes more than the available library. Trial counts differ between the executor groups: ten per task for CaP-Agent0 and five for RATS execution. These bars are descriptive results, not an equal-latency or equal-token-budget comparison, and no uncertainty intervals are supplied here. Random-play rows, omitted for legibility, are 25.0% for CaP-Agent0 and 38.3% for RATS execution. The separate headline Table 1 reports 43.8% for RATS versus 23.2%, yielding +20.6 pp; do not mix that value with the 44.3% Table 4 ablation. The paper’s main evaluation has 60 LIBERO-PRO tasks; release defaults and model examples differ from some paper settings. These controlled code-agent comparisons are more informative about the skill library than a broad claim that code policies outperform all VLAs. Evaluation remains mainly simulated. The appendix adds physical tasks beyond the two main-table tasks, but the quantitative real-world set is still small and transferred skills can hurt individual tasks. Source: paper Table 4 and §§4.4–4.5; limitations on p. 9.

- [RATs · Table 4](https://playful-rats.github.io/assets/papers/rats-paper.pdf#page=8)

## 25. COAP: Code-Only-as-Policy

**Section: Code, execution, and improvement**

COAP is Code-Only-as-Policy, the Ropedia A0 paper at arXiv:2610.12369. It is distinct from the earlier Code as Policies method. This authored diagram summarizes §§3.1–3.4 and Figure 3. The paper places coding agents in development, where they inspect logged failures and revise a shared library. Once an episode starts, source code θ stays fixed. The evolving state x_t and current observations o_t determine both the next state and trajectory segment through (x_{t+1}, a_t) = F_θ(x_t, o_t). A fixed program therefore does not mean replaying a fixed action sequence. Its conditions, measurements, recovery branches, and retry counts can produce different movements in different episodes.

The observation interface includes three RGB views, from the head and both wrists, joint angles, end-effector poses, and gripper command echoes. Explicit state records object measurements and their confidence/source/time, robot configuration, and task progress or memory. NumPy/OpenCV perception fits known object geometry to image regions. Motion code uses kinematics and collision checks to choose feasible trajectories, and the simulator controller executes commands. The simplified diagram groups some routines for readability. It does not show a neural policy hidden in the geometry box.

The paper reports no LLM, VLM, or VLA service during execution. An online GPT controller places model calls on the observation-to-decision path instead. That is an execution boundary, not an intrinsic ranking of the two approaches. COAP uses benchmark-provided camera calibration, robot meshes, known object shapes, and table height as constants. It measures episode-specific object states from RGB. Development agents can inspect simulator ground truth, while the deployed policy does not receive oracle object state. Do not generalize its perception assumptions to arbitrary unseen household objects or compare its scores directly with the following study. Source: https://arxiv.org/abs/2610.12369, §§3.1–3.4.

- [COAP · §§3.1–3.4](https://arxiv.org/abs/2610.12369)

## 26. A robot software improvement loop

**Section: Code, execution, and improvement**

This is Wang’s technical report What Stops Recursive Self-Improvement in Robotics? Lessons from 123 Rounds of Agentic Skill Discovery, arXiv:2609.31760v1. The diagram adapts its Figure 2. The simulated PandaOmron mobile manipulator receives head and wrist RGB-D images and joint readings. Its task program calls typed skills for localization, finding fixture regions, approach, grasp, and motion. Skills call external providers such as segmentation, depth, grasp generation, motion planning, and active-view selection. Unlike COAP, this runtime includes external model providers. The GPT coding agents themselves are never called by the deployed robot.

Three agent roles operate between trials. The orchestrator writes task programs. The proposer sees diagnoses, execution traces, past attempts, and camera keyframes, then proposes a skill edit or research request. A research agent finds, installs, wraps, tests, and registers external components. Routing is refitted from skill-call logs. Accepted changes go live in a later round. The report describes code/library changes and provider acquisition, not gradient-based training of an action policy. Humans chose the coding model and reasoning setting: gpt-5.6-sol for rounds 1–75, then gpt-6-astra. Those model-selection changes were not made by the loop.

Declared step effects are judged using privileged simulator state. Exact failure replay checks an observed case, while paired old/new tests use fresh states and a held-out cabinet-to-counter task. Sealed failures and checks on other affected skills were added later. This protocol evolved during the run. Four scene seeds per task were repeatedly used for collection in a single kitchen layout. Cabinet and Sink were added after the initial Fridge phase. The held-out task participates in adaptive acceptance, so it is not an untouched final benchmark. From round 74, containers hid the run repository and held-out task from agents.

The warm boundary is central: humans were prohibited from writing skills, programs, or model wrappers, but could repair the evaluator, tests, memory, and actuator description. They made 113 such repairs and suggested the later VLM grounding class. The harness therefore remained a substantial human-maintained component. The term self-improvement describes a bounded software loop, not a robot changing every part of its own learning process. Source: https://arxiv.org/html/2609.31760v1, §2, Table 1, and Appendices A–B.

- [123-round study · §2](https://arxiv.org/html/2609.31760v1#S2)

## 27. 57 accepted changes, zero fridge completions

**Section: Code, execution, and improvement**

The two headline counts come from Appendix B, Tables 5–6. There were 57 accepted library changes, of which 51 concerned perception, and no Fridge completion among 486 completed collection rollouts. The other completed-rollout totals were Cabinet 0/256 and Sink 14/230. Nine Sink successes had failed intermediate program steps. The narrative explains that many succeeded when an object happened to fall into the sink after the intended placement sequence failed. Native task success and successful execution of the declared chain are different measurements. The slide uses cumulative collection outcomes, not a final held-out test success estimate.

The full run contained 847 proposal attempts, 993 attempted and 972 completed rollouts, and 113 human harness repairs. Routine routing refits are excluded from the 57 accepted changes. Eighty-six provider registration events correspond to only 20 identities, because repaired providers could be registered again. Eight identities were later used by deployed skills. All eleven identities introduced during rounds 114–121, including new vision-language grounding components, had zero calls in subsequent task collection. The lower diagram represents this late subset. It does not claim that all providers were unused. The arrow compares two observed stages of integration, not a successful transition.

Three mechanisms help explain the outcome. The segmentation pipeline repeatedly confused relational targets such as the top shelf. Upstream failures prevented rollouts from reaching later skills: 57% of fridge rollouts in the three-task phase failed at step 1, and only 2.7% passed four steps. Finally, evaluator errors and incorrectly recorded research requests distorted selection and memory. A grasp change in round 118 won reproducible fresh-state and held-out comparisons, yet the next 60 task rollouts had no completion. A local improvement can be real and still make little measurable difference to a long task.

This is one long run in one simulator and kitchen layout. It has no independent replications, matched-budget baselines, or controlled ablations. Agents, providers, and the human-maintained harness changed together. It supports specific failure diagnoses, not a universal claim that robot self-improvement cannot work or that a stronger agent cannot help. Its setup differs substantially from COAP, so the headline outcomes are not a head-to-head comparison. Source: https://arxiv.org/html/2609.31760v1, §§5–7 and 10, Tables 5–7.

- [123-round study · Tables 5–7](https://arxiv.org/html/2609.31760v1)

## 28. How would we know that self-improvement helped?

**Section: Synthesis**

This is a proposed experimental design, not an additional result. Use paired or otherwise matched test episodes and retain all relevant budgets, tools and controller settings. To attribute improvement to the library, hold the execution agent fixed. A second experiment can vary the test-time agent. Use an untouched final evaluation when model/program selection repeatedly consults validation results. Report exploration cost, human interventions, inference overhead and physical task time separately. Record failures as well as successes and show whether new code is actually exercised. RATs provides a useful library/execution ablation; the RSI report illustrates how local acceptance and end-to-end performance can diverge.

- [RATs](https://playful-rats.github.io/)
- [Wang: 123 rounds](https://arxiv.org/abs/2609.31760)

## 29. Compare interfaces and what gets updated

**Section: Synthesis**

This table describes common cases, not hard category boundaries. Neural weights are usually updated during training and frozen for a deployment rollout. Code systems may call these trained models and may or may not revise code online. WAM future generation can be optional at inference. A hybrid might use a VLM for target selection, a WAM for prediction, a VLA for manipulation, and code to coordinate them. RSI is an update process applied to a chosen part of that system. Compare components only after specifying observations, action interface, training data and evaluation budget.

- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [UNITAS §3](https://arxiv.org/html/2610.12099v1#S3)
- [Code as Policies](https://code-as-policies.github.io/)

## 30. Apply the framework to one robot task

**Section: Discussion**

Suggested discussion: first specify camera/depth and robot state, then joint or tool-pose commands and the controller. Decide whether the central challenge is semantic grounding, dexterous behavior, consequence prediction, or task-level coordination. A VLA needs appropriate physical action supervision; a WAM needs future-state or trajectory targets; a code agent needs usable sensing and control interfaces. Propose a failure record and a held-out evaluation before choosing an RSI loop. Core readings are linked on their corresponding slides and in these notes. Style references supplied by the user were the Flex-π reading-group deck and How to Train and Fine-tune VLMs. Their organization inspired the figure-led teaching approach; no institutional branding is copied.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [UNITAS §3](https://arxiv.org/html/2610.12099v1#S3)
- [RAX](https://arxiv.org/abs/2610.22496)
- [RATs](https://playful-rats.github.io/)
- [Wang: 123 rounds](https://arxiv.org/abs/2609.31760)
