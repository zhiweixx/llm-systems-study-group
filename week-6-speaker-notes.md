# Week 6 — Robot learning: from demonstrations to VLAs and LLM agents

30 slides for researchers familiar with LLMs and statistics who are new to robotics. Slides 1–11 establish control interfaces, recorded data, imitation learning, and temporal feedback. Slides 12–22 connect VLMs, VLAs, continuous action generation, and the π0.5 case study. Slides 23–30 compare LLM robotics interfaces and recent reported evidence, ending with a research question and solution.

Read alongside MIT Robotic Manipulation, Stanford CS224R (2026) imitation learning, and the primary papers linked on each slide. Teaching diagrams are authored illustrations. Dataset frames and paper plots retain source attribution. Reported results are not new experiments run for this deck; evaluations with different tasks and interfaces are not directly comparable.

Use Previous step / Next step within the action-chunk and π0.5-evidence slides. Arrow keys navigate slides; Notes opens the explanation and references. The HTML embeds its images and works offline. Print / PDF shows the final interactive frame. Recent sources checked 24 October 2026.

## 1. From images and language to robot motion

**Section: Opening**

Use the cup-to-drawer instruction as an illustrative running task, not as an experimental claim or the task in the ALOHA dataset example. Ask what exactly leaves the model: a skill name, a tool pose, a vector of joint targets, or a sequence of commands. Every choice requires an execution interface. The robot changes its world, so subsequent inputs depend on past decisions. This is the conceptual bridge from familiar statistical prediction to embodied feedback. The first eleven pages establish interfaces and imitation learning before adding pretrained visual/language representations. No numerical benchmark is implied by this diagram.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 2. A robot acts inside a feedback loop

**Section: Robot interfaces**

Read the outer loop first. Cameras and joint sensors supply observations to a policy or a planner. The output is a desired motion, while a controller uses measurements to track it through actuation. The inner green loop remains meaningful even when visual decisions are slower. Proprioception means robot-self measurements, such as encoders and velocity estimates; it is not another camera view. The world state also includes object pose, contact, friction, and other quantities that are not directly observed. A current image plus joint positions need not be Markov; histories or state estimation may help. “Policy” and “controller” can overlap in robotics terminology; here we separate a learned decision interface from its lower-level tracking implementation. A skill-call interface adds a skill executor, and Cartesian targets may add inverse kinematics or planning, before reaching low-level control. A torque policy would move the boundary, so the diagram is a position-target example rather than a universal stack.

- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [MIT · Output feedback](https://underactuated.mit.edu/output_feedback.html)

## 3. What does an action vector mean?

**Section: Robot interfaces**

The arm is a planar teaching schematic, not ALOHA geometry or a complete three-dimensional frame drawing. A real frame has three orthogonal axes; the two displayed axes emphasize that tool and world axes can differ. Joint configuration describes the robot’s internal coordinates. End-effector pose describes tool position and orientation relative to a reference frame. Forward kinematics maps configuration to pose; inverse kinematics seeks configurations achieving a desired pose and may have zero, one, or multiple solutions. A valid endpoint is different from a feasible, collision-free trajectory to it. The starred q is a desired configuration. Units and joint order must be supplied by the interface; revolute coordinates commonly use radians, whereas prismatic coordinates use length. Cartesian orientation can use a rotation matrix, quaternion, axis-angle, or another convention. Absolute, relative-to-current, and sequential-delta commands also differ. A gripper coordinate might be width, normalized opening, velocity, or force. None of these conventions should be inferred from a vector dimension alone.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [Modern Robotics · Transforms](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/)

## 4. The model and the motors run on different clocks

**Section: Timing**

This is an authored asynchronous timing example, not measured performance from ACT, UMI, or a VLA. Camera captures are approximately every 33 milliseconds. Two model calls start 100 milliseconds apart and each takes 80 milliseconds; a call uses the latest image at its start. New commands begin when results are available and are played at 20-millisecond intervals. A previous command or a hold behavior covers the initial computation period. Servo feedback remains active throughout, including during model computation; its drawn ticks are compressed and do not each represent one millisecond. Under the simplified zero-transport-delay assumption, the first result uses an 80-millisecond-old image. Real systems also incur sensor, communication, actuator, and scheduling delays. UMI makes this concrete by aligning measurements to image capture times and matching action execution times to measured latency. A simulator that pauses during inference would have different physical-time behavior. Lower model latency and faster actuator command playback address different bottlenecks.

- [UMI · Data and timing](https://arxiv.org/html/2402.10329)
- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 5. Open one demonstration episode

**Section: A real dataset record**

This is the public simulated ALOHA insertion dataset, not a physical-robot episode. The two images are actual frames 0 and 100 of episode 0. The numeric table deliberately compares rows 0 and 1 instead, to expose the distinction between a measured position and its stored target. Episode 0 contains 500 records. Each state and action vector has left six arm joints, left gripper, right six arm joints, and right gripper. The simulation interface defines absolute desired joint coordinates, with revolute arm angles in radians and normalized gripper opening nominally zero closed and one open; recorded normalized measurements may slightly exceed that nominal range. The inspected historical converter copies qpos, action, and images at the same index and constructs timestamp = frame_index / fps. These are nominal episode times, not measured sensor timestamps or proof that a human issued a command at that exact wall-clock instant. The ACT simulation loader pairs qpos/images at t with actions starting at t. The task string in current dataset metadata is “Insert the peg into the socket.” This is a task label, not per-frame speech, and original ACT does not consume language. ACT simulation interface: https://github.com/tonyzhaozh/act/blob/main/sim_env.py ; original loader: https://github.com/tonyzhaozh/act/blob/main/utils.py . The converter sets next.done on the final record, so that field does not by itself certify success.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)

## 6. Turn an episode into training examples

**Section: Training data**

This diagram is a pedagogical H=4 training window, not the reported ACT chunk length. d_a is the dimension of one command, fourteen in the shown ALOHA interface, while H is the number of commands predicted together. Inputs may include a history rather than only one observation. The action at the same stored index is the first target under the verified simulation loader convention; another dataset may use a different alignment, so inspect acquisition and conversion code before applying a shift. Sample windows within episodes and avoid train/validation leakage from neighboring frames. End windows can be omitted or padded with an explicit mask; padding must not be mistaken for new valid control labels. ACT uses a conditional latent-variable model during training, but that architecture is secondary to the input/target relationship here. Original ACT conditions on images and joint measurements, without a language input. A later language-conditioned policy may add an instruction to the input; the dataset’s task metadata alone does not make ACT language-conditioned. An offline inspector can check these windows and shapes, but cannot establish closed-loop success, recovery, or real-world transfer.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [ACT · Data loader](https://github.com/tonyzhaozh/act/blob/main/utils.py)
- [LeRobot · Dataset format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 7. More robot data also means more interfaces

**Section: Data at scale**

These are collection and aggregation examples, not three interchangeable datasets. DROID collects demonstrations on a common Franka platform in many environments. UMI uses a handheld gripper with visual tracking and carefully aligns observations/actions for deployment. Open X-Embodiment combines datasets from many embodiments. A common RLDS or LeRobot container does not by itself align physical meanings. A fourteen-dimensional joint target differs from a seven-dimensional tool/gripper command, even after scaling each coordinate to unit variance. The illustrated point-frame transform has both rotation and translation; transformation rules for rotations, relative displacements, twists, and forces require their own conventions. Numerical normalization changes coordinate scale, while geometric calibration relates frames. UMI’s relative trajectory representation reduces some embodiment-specific variation but still needs calibration, timing, and an execution interface. Some Open X-Embodiment training adaptations use per-dataset normalization without geometrically aligning all frames. Preserve metadata rather than treating a float array as self-describing. LeRobot’s numeric tables, videos, and episode metadata are storage/access machinery, not a universal robot action schema.

- [DROID · Dataset](https://arxiv.org/html/2403.12945v2)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)
- [Open X-Embodiment · §III–IV](https://arxiv.org/html/2310.08864v6)

## 8. Behavioral cloning learns an action distribution

**Section: Imitation learning**

For this audience, begin from conditional density estimation. Demonstrations supply observations and action labels, including whole chunks if that is the chosen output. The expectation is over stored demonstration pairs; a language-conditioned extension adds the instruction after the conditioning bar. Negative log likelihood is a common formulation for an explicitly parameterized policy. Under a fixed-variance Gaussian assumption, minimizing it reduces to squared-error regression up to constants and scale. Enlarging a deterministic network does not by itself turn a single Gaussian or a mean into a multimodal conditional distribution. The drawn density is an illustrative marginal of one action coordinate; the actual chunk distribution lives in H times d_a dimensions and must model correlations. Do not claim every imitation method directly optimizes tractable likelihood: ACT uses a variational latent-variable objective with reconstruction and KL terms, while diffusion and flow matching use their own training losses. These methods are within imitation learning, not alternatives to imitation learning as a category. Reinforcement learning changes the training signal toward cumulative reward and will be introduced later.

- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)
- [MIT · Imitation learning](https://underactuated.mit.edu/imitation.html)

## 9. A small action error can change the next input

**Section: Closed-loop distribution shift**

The horizontal drawing is a schematic state/observation-space path, not a physical tool trajectory or a quantitative probability-density plot. The pale band represents the region sampled by expert demonstrations. A learner error can take the system into inputs that were rare or absent there, and further errors can compound. This differs from evaluating a predictor on a fixed, externally sampled test set. Explain DAgger operationally: run the current learner, query the expert for what should have been done at the visited states, add those labeled pairs to all previous data, and retrain. The original algorithm can mix expert and learner control during collection. Querying the expert for a label is distinct from always letting the expert take control; human-gated intervention is a related practical variant. Do not present worst-case horizon-squared bounds without their assumptions or claim that chunking eliminates distribution shift. Episode-level action loss is useful for debugging and selection, while native task success and recovery require closed-loop rollouts.

- [DAgger · Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 10. Predict a chunk, execute a prefix, observe again

**Section: Action chunking**

Advance the four steps. First the model sees o_0 and predicts commands for steps zero through five. It executes only steps zero and one. At step two a fresh observation leads to a new chunk, so the unused commands from the old prediction can be replaced. The command subscript is its intended execution step; the parenthesized superscript is the observation/prediction step, so a_2^(0) and a_2^(2) need not have the same value. At step four the process repeats. Green cells have been executed; pale blue cells are future proposals; crossed gray cells are superseded proposals, not executed commands. H is the prediction length and h the execution prefix. They need not be equal. The drawing is logical timing and omits inference latency, queueing, and asynchronous implementation details discussed earlier. The policy can be visually open-loop during each prefix while its lower-level joint servo remains closed-loop. This is the receding-horizon execution pattern used by Diffusion Policy. ACT also studies action chunking, but its temporal ensembling variant queries at every step and combines overlapping predictions for the same execution time; it is not simply the exact prefix-playback algorithm drawn here. Larger h can lower model-call overhead and preserve temporal consistency while delaying visual correction. It is an engineering tradeoff, not a guarantee that longer or shorter is always better.

- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)
- [ACT · §IV](https://arxiv.org/html/2304.13705v1)

## 11. Generate a continuous action chunk

**Section: Generative action policies**

The left sketch motivates modeling a distribution: tool-position command routes can go around either side of an obstacle, whereas averaging coordinates can create an invalid route. It is an illustrative geometric example, not a claim that multimodality explains every gain of generative policies or that the command route will be tracked perfectly. On the right, each plot shows only one coordinate of a proposed H-by-d_a array as a function of future command offset j. Moving from plot to plot is a denoising iteration, not the robot moving forward in physical time. The jagged input is sampled noise, and the final curve is an action sequence; it is not a forecast of measured robot states. The generic diffusion-policy training procedure perturbs demonstrated chunks at sampled noise levels and trains a conditioned network to predict the noise. Sampling applies repeated learned denoising updates, with a schedule and possible stochastic terms. Observation features condition generation rather than being generated as future images. The exact update and noise parameterization vary by sampler, so the schematic curves are not numerical samples from the published system. Diffusion Policy can model temporal correlations and several valid behaviors while using receding-horizon execution. Flow matching will later make another continuous generation objective explicit.

- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 12. A VLM connects images to language representations

**Section: VLMs and robot actions**

This is an editable architectural illustration, not a model-generated example. PaliGemma combines the SigLIP ViT-So400m vision encoder, a linear projection, and Gemma-2B. The projection aligns feature dimensions, rather than converting image patches into vocabulary IDs. Its image and prompt prefix uses bidirectional attention, while the generated answer suffix uses autoregressive attention. The model can emit natural language and serialized structured outputs such as boxes. Its base pretraining is not chat instruction tuning. Recognition and semantic representations do not by themselves specify a robot action coordinate system, scale, or execution controller. OpenVLA uses a different VLM backbone: Prismatic, with DINOv2 and SigLIP features plus Llama 2. The schematic introduces the reusable three-part VLM structure, not a claim that every VLA uses PaliGemma. Source: PaliGemma §3.1 and Figures 1–2.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 13. A VLA turns those representations into robot actions

**Section: VLMs and robot actions**

The top row is a deliberately authored quantization example, not an observed OpenVLA training value. Mapping +0.0200 m from [−0.10,+0.10] to [−1,1] gives +0.20. With 256 uniform bins, floor(256×0.60)=153, and decoding to that bin center gives approximately +0.01992 m. A token index is not a physical displacement. Original OpenVLA fine-tunes Prismatic-7B on 970k demonstrations, predicts one 7D end-effector/gripper action from one image and an instruction, and computes next-token cross-entropy on action tokens only. It uses 256 bins per coordinate between the training dataset’s first and 99th percentiles and repurposes 256 vocabulary entries. It does not have the chunking, proprioceptive input, or multiple-camera features of later derivatives. RT-2 co-fine-tunes on robot data and web vision-language tasks, uses an eight-value encoding including termination, and constrains output vocabulary for action prompts. Both generate discrete action codes autoregressively. A robot-specific decoder and controller still execute the resulting numeric command. Sources: OpenVLA §§3.1–3.3 and RT-2 §§3.2–3.3.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)

## 14. Which capability comes from which experience?

**Section: VLMs and robot actions**

Read across each row rather than reading downward as a leaderboard or a strict family tree. RT-1 scales language-conditioned robot imitation and uses a pretrained visual encoder and language embeddings, but it is not a conversational VLM adapted into a policy. Its final action dimensions are predicted non-autoregressively. RT-2 and OpenVLA adapt VLM representations to action prediction. RT-2 reports semantic reuse of motor skills supported by robot data, not arbitrary new motor skills learned solely from web text. ACT and Diffusion Policy are behavioral-cloning methods with expressive temporal/action representations; original ACT is task-specific and has no language conditioning. Their chunking and generative modeling choices can coexist with pretrained VLMs, as π0 will demonstrate. ACT’s CVAE is a conditional variational autoencoder; the present lesson focuses on chunk prediction and temporal ensembling rather than deriving its latent-variable objective. The table is an authored synthesis from RT-1 §5 and Appendix D.4, RT-2 §§3–5, OpenVLA §3, ACT §IV, and Diffusion Policy §3. Additional primary links: https://arxiv.org/html/2304.13705v1 and https://arxiv.org/html/2303.04137v5.

- [RT-1 §5](https://robotics-transformer1.github.io/assets/rt1.pdf)
- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 15. π0 combines a VLM with a continuous action expert

**Section: Continuous VLA actions**

π0 initializes its visual-language weights from PaliGemma and adds a randomly initialized 300M-parameter action expert, giving roughly 3.3B parameters in total. The robot configuration q_t and action slots use robotics-specific expert weights. Image and text slots use the VLM weights, and attention provides the conditioning connections. This is a simplified computational diagram, not a separate large-language planner calling a black-box motion module. In the original π0, q_t is projected continuously. The original π0.5 changes proprioception to discretized text input, which will be distinguished later. The flow model predicts a vector field on an H=50 action chunk; ten Euler steps generate a chunk in the reported experiments. These ten iterations are generation computation, not ten physical servo cycles. During inference, prefix attention keys/values can be cached while the suffix is updated. Prefix representations are fixed within this sampling call, not permanently frozen model weights. π0 trains on diverse robot demonstrations and then uses curated downstream post-training; some long tasks use an additional high-level VLM. Sources: π0 v4 §IV, Figure 3, Appendices B/D, and CS224R lecture 17 p.6.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 17](https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf)

## 16. Flow matching: learn a path from noise to actions

**Section: Continuous VLA actions**

A is a demonstrated H×d_a action chunk and ε is an independent Gaussian tensor of the same shape. Sample τ from the training schedule, interpolate X_τ=(1−τ)ε+τA, and regress the network’s vector field against A−ε. The expectation is over demonstrated context/chunk pairs, independent noise, and generation time. Under squared error, the optimal field at a noisy point is the conditional mean of these local velocity targets; integrating this field transforms an entire noise distribution into an action distribution. This does not require the output action distribution to be Gaussian or to equal a single conditional mean action. Training can sample one τ per example rather than unrolling the full inference trajectory. At inference no demonstrated A is available: sample X_0 and repeatedly apply the learned field with observation/instruction fixed during the call. π0 v4 uses ten Euler steps and a training τ distribution emphasizing noisier times. The equations consistently use noise-to-action orientation. Older π0/π0.5 texts contain an inconsistent target sign, so the teaching equations follow the corrected π0 v4 §IV. The field has units of normalized action per unit generative time, not necessarily physical end-effector velocity. CS224R lecture 2 pp.11–14 gives the corresponding teaching derivation.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 17. What must transfer to a new home?

**Section: π0.5 case study**

This schematic isolates semantic recognition, subtask choice, and motor execution. A new environment can change background appearance, object instances, layout, and local geometry simultaneously. The diagram is an authored cup-to-drawer illustration and is not a reconstruction of a training or evaluation home. π0.5’s mobile training subset contains about 400 hours across about 100 home environments, supplemented with other robots, laboratory tasks, high-level labels, web tasks, and later verbal instruction demonstrations. Its central claim is transfer of skills to new settings. The official explanation explicitly distinguishes that from learning wholly new physical skills without relevant robot experience. The later evaluation slide separates real-home task progress from controlled mock-environment ablations. Sources: original π0.5 Introduction, §IV-C, §V, and official project explanation.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [π0.5 explanation](https://www.pi.website/blog/pi05)

## 18. π0.5 learns from several kinds of supervision

**Section: π0.5 case study**

Read each row as an input/target pair. MM is mobile manipulation in homes. ME collects demonstrations from easier-to-transport non-mobile robots in diverse homes. The laboratory cross-embodiment category includes other robot configurations and Open X-Embodiment. HL manually labels robot observations with semantic subtask descriptions and relevant object boxes. Web data supplies captioning, VQA, and localization supervision. VI is collected after an initial low-level policy exists: an expert guides it by selecting verbal subtask commands, producing demonstrations of high-level decisions. These are not additional teleoperated joint-action labels from speech alone. The mobile subset is roughly 400 hours in roughly 100 environments. The paper reports that 97.6% of first-phase examples come from sources other than the mobile household subset, but does not disclose a complete numeric mixing table. The boxes/targets here are schematic categories rather than copied training records. Sources: original π0.5 Figure 4 on PDF p.6 and §§IV-C/D.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 19. Separate the training recipe from runtime

**Section: π0.5 case study**

This diagram follows the original 19-page π0.5 paper rather than treating current OpenPI code as an exact reproduction. First, pretrain for 280k gradient steps with next-token prediction over text, boxes, and compressed FAST action tokens. This phase uses mobile manipulation, multi-environment non-mobile data, laboratory cross-embodiment data including OXE, high-level labels, and web data. Second, add a randomly initialized action expert and train another 80k steps with joint cross-entropy and flow matching, using flow-loss coefficient 10.0. Post-training action data use MM/ME successful episodes filtered by a length threshold. It retains web and relevant high-level data, adds verbal instruction examples, and omits laboratory cross-embodiment data. FAST discrete action outputs and flow action outputs are masked from attending to one another. FAST compresses a chunk; it is not identical to OpenVLA’s independent 256-bin coordinates. Original π0.5 discretizes proprioceptive state into text input, unlike π0’s continuous state projection. At inference, the system uses autoregressive subtask text followed by flow sampling. Current OpenPI README describes π0.5 trained with knowledge insulation and only supports the flow-matching head for π0.5 training/inference; do not assume downloading it reproduces this entire household hierarchy. Sources: π0.5 Figure 3 (p.4), Figure 4 (p.6), §§IV-A–D, and OpenPI README.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [OpenPI release](https://github.com/Physical-Intelligence/openpi)

## 20. Decide the subtask, then generate the motion

**Section: π0.5 case study**

The equation is the paper’s model factorization, not the unconstrained probability chain rule: its action distribution depends on the inferred subtask c_t and current observation, but not directly on the overall task prompt ℓ. The same model supports autoregressive semantic output and continuous flow-based action output with different expert weights. In the actual system high-level inference uses four camera views, while low-level inference uses the wrist and forward cameras; o_t denotes available observation context rather than asserting identical camera subsets. Proprioception includes joints, grippers, lift and base velocity. The mobile platforms have 18 or 19 action dimensions. They issue arm/gripper/lift targets and base velocities at 50 Hz with action chunking; simple PD controllers track the targets. The paper reports no additional trajectory planner or collision detector in this execution path. Ten flow denoising steps generate the 50-step chunk. A 50 Hz command stream does not mean the entire VLM generates a semantic subtask 50 times per second. The official explanation describes a one-second chunk. Distinguish high-level demonstration labels used in training from inferred subtasks used at deployment. Sources: π0.5 §§IV-A/B/E and official explanation.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [π0.5 explanation](https://www.pi.website/blog/pi05)

## 21. What did the π0.5 experiments establish?

**Section: π0.5 case study**

Frame 1 reproduces Figure 7(b). §V-A explicitly says three held-out real homes with bedroom and kitchen tasks in each. Thus the caption’s three kitchens plus three bedrooms describe six room environments in three homes. The plot compares task progress, averaged over ten trials per task/environment, with a rubric roughly reflecting completed steps. It is not a universal task-completion success rate. Many quantitative trials last 2–5 minutes; longer 10–15-minute demonstrations in the introduction are separate evidence. Appendix B notes exclusions of some canceled episodes due to robot failures, time limits, or other causes, so do not infer a larger exact total trial count from the caption alone. Frame 2 reproduces Figure 13 in controlled mock environments. The full model uses semantic inference. The implicit-HL variant keeps high-level training examples but omits high-level inference at runtime. No-HL removes both. The strong implicit-HL result shows that training supervision accounts for a substantial part of the gain; the result should not be reduced to “explicit reasoning is essential.” The figure also contains no-web, no-verbal-instruction, GPT-4 high-level, and human high-level comparisons, which are defined in §V-E. These share the low-level inference process, and the main paper discusses the full model outperforming the chosen human high-level baseline. This does not establish a broad human-ability comparison. §VI describes failure on difficult unfamiliar handles, occlusion, repeated subtask choices, simple-prompt limitations, and restricted memory. Source: original π0.5 §§V-A/E, VI, and evaluation appendix.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 22. Demonstrations provide a starting point for improvement

**Section: Policy improvement**

Behavioral cloning learns from demonstrated actions. Reinforcement learning optimizes expected cumulative reward and can exploit feedback about what happened after actions. A pretrained VLA supplies useful exploration and representations but generally does not supply a trained critic/value function. This diagram is an authored overview, not the exact algorithm of π0.5 and not a claim that π0.5 itself used this RL stage. CS224R lecture 17 distinguishes online updates from iterated offline learning over large rollout batches, describes value-guided supervised policy updates, and explores latent steering and residual/edit policies. These methods make different compromises; we do not teach them as interchangeable or guaranteed improvements. Human interventions can supply useful imitation data and coexist with RL. Large models incur substantial gradient-update costs, physical resets and rollouts can dominate collection time, and diffusion/flow action generators plus chunking complicate direct use of algorithms built around tractable per-step policy probabilities. Avoid transferring the lecture’s informal “80% plateau” into a universal reliability law or reading π0.5’s task-progress metric as success probability. Sources: CS224R lecture 17 pp.5–15 and 20–29, lecture 2 on online interventions.

- [CS224R, lecture 17](https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 23. Language enters the robot stack at several levels

**Section: Language models and robot control**

SayCan multiplies language-model usefulness with a learned value function estimating skill feasibility, then selects an existing skill. Code as Policies writes programs that process perception outputs and call control APIs, including reactive loops. VoxPoser generates code that constructs spatial value maps from observations and language; planners use those maps as objectives. Its cached code can be re-evaluated with fresh observations. These rows describe outputs and executors, not mutually exclusive model families. The diagrams simplify the papers rather than reproducing one shared implementation.

- [SayCan](https://say-can.github.io/)
- [Code as Policies](https://code-as-policies.github.io/)
- [VoxPoser](https://voxposer.github.io/)

## 24. What does a general model send to the robot?

**Section: Language models and robot control**

Read each row as the output of the language-related component followed by its executor. A numeric target requests a configuration or pose rather than a motor torque. A generated program can close feedback loops through API calls. In policy review, learned robot behavior supplies candidate actions and the general model can retain or replace them. Actual systems can combine rows. This taxonomy extends the original control loop and separates action representation from which parameters were trained.

- [Code as Policies](https://code-as-policies.github.io/)
- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)
- [Astra policy study §3](https://arxiv.org/html/2609.38537v1)

## 25. RoboDojo exposes uneven task performance

**Section: Language models and robot control**

The model sees three RGB views, proprioception, text history and the latest two image turns. Wiki-derived scoring descriptions differ from baseline inputs. A non-learned stack executes targets at 25 Hz. Macro SR averages categories, not pooled trials. The repository discloses replacement of two task cells with targeted reruns and a 170-call budget, while the paper specifies 100 by default. Treat this as the paper-reported campaign rather than an independently verified current leaderboard rank. The category pattern does not identify a causal cognitive mechanism.

- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)
- [Released run notes](https://github.com/RoboProbe/RoboProbe/blob/3c0bc68383a75bf3fa2d7bfc5ab1d1241c7e171e/results/l3_inspect_eef_official_2100/README.md)

## 26. A fast playback loop can hide slow decisions

**Section: Language models and robot control**

The reported 18.56-second mean call latency uses successful episodes only and includes network delay. It is distinct from the 15 Hz execution rate. Physics pauses while inference runs. Astranomy uses task-specific action-step limits, without wall-time or retry limits. Direct retains a short observation window, whereas the agentic system retains persistent history; their success difference cannot be assigned to compute alone. Orchestration supervises Cosmos3-Nano-Policy proposals. Five trials per task give limited precision for small differences. The drawn segment widths are schematic, not a reconstructed episode. Successful-episode cost averages must not be relabeled as unconditional cost per attempt.

- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)

## 27. Reasoning can revise a learned policy’s proposal

**Section: Language models and robot control**

The RoboDojo hybrid uses released task-finetuned π0.5 weights. Astra either accepts a prefix or substitutes an end-effector correction; the two action branches are exclusive. Direct and Hybrid share task instances, seeds, descriptions, reasoning setting and EEF interface. Ten tasks were stratified by published policy success, with more drawn from low-success intervals. Hybrid completes 24/50 versus Direct 13/50. Classification tasks can favor Direct. Published standalone-policy references were reweighted, not rerun in the paired experiment. The diagram makes the cooperative mechanism explicit without treating this subset as the full benchmark.

- [Astra policy study §3](https://arxiv.org/html/2609.38537v1)

## 28. A robot result includes its evaluation protocol

**Section: Language models and robot control**

This is an authored design for a controlled comparison, not a claim that all cited studies share one interface. Use identical held-out cases and declare what each agent can observe, call, remember and reset. Freeze any development before measuring held-out performance. Embodied First Exam separates standard mode from standard-realtime: in realtime, physics advances during thinking and holds the last target between commands. Its default reset budgets also differ, so the mode label alone is insufficient. A second, useful engineering question compares the best complete systems with their own interfaces; report those access and budget differences rather than calling that a model-only experiment.

- [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md)
- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)
- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)

## 29. Question: what makes this comparison informative?

**Section: Research exercise**

Give the audience a minute before revealing the next page. Ask for a distinction between a matched-interface experiment and an engineering comparison of complete systems. The drawer displacement is an authored disturbance, not a claim about a published task. Useful answers identify pretraining and task-adaptation differences, observation timestamps and calibration, stale actions during inference, reset rules, and native physical completion. An action count alone does not capture the consequences of a real-time disturbance. No fabricated benchmark result or interview attribution accompanies this exercise.


## 30. Solution: matched trials and a budget curve

**Section: Research exercise**

This is a proposed experimental design and an invented curve, not a numerical prediction. Evaluate repeated matched cases and report uncertainty appropriate to the task and scene sampling. Time includes model computation, tools and execution. State whether physics continues during computation; use the same perturbation schedule in a controlled comparison. Record training and adaptation evidence even when it cannot be equalized. A matched action interface asks about a controlled component change, while allowing each system its preferred executor asks about engineering performance. End by linking this experiment back to the observation, action, feedback and data concepts introduced at the start of the lesson.

- [Embodied First Exam](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md)
