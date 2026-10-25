# Week 6 — Robot learning: from demonstrations to VLAs and LLM agents

60 slides for researchers familiar with LLMs and statistics who are new to robotics. Slides 1–24 establish control interfaces, recorded data, imitation learning, and temporal feedback. Slides 25–46 connect VLMs, VLAs, continuous action generation, and the π0.5 case study. Slides 47–60 compare LLM robotics interfaces and recent reported evidence, ending with a research question and solution. The original 30-slide presentation is preserved separately as a reference.

Read alongside MIT Robotic Manipulation, Stanford CS224R (2026) imitation learning, and the primary papers linked on each slide. Teaching diagrams are authored illustrations. Dataset frames and paper plots retain source attribution. Reported results are not new experiments run for this deck; evaluations with different tasks and interfaces are not directly comparable.

Play the embedded ALOHA demonstration and use Previous step / Next step within the teaching examples. Arrow keys navigate slides; Notes opens the explanation and references. The HTML embeds its figures and robot demonstration and works offline. Print / PDF shows the final interactive frame and a static video poster. Recent sources checked 24 October 2026.

## 1. From images and language to robot motion

**Section: Opening**

The source illustration is the pick-and-place setup from MIT’s manipulation course. Start with the concrete task of moving an object to a desired pose. The central question is what numerical or executable output actually moves the machine. The lesson first follows sensing, geometry, data, and imitation learning, then adds pretrained vision/language models and LLM-based control interfaces. This figure is a teaching setup, not an experimental result from the ALOHA demonstration used later.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)

## 2. A robot acts inside a feedback loop

**Section: Foundations**

The policy chooses a command, such as a desired joint configuration. A lower-level controller uses measurements to track that command through actuation. The robot and its environment produce new observations. Proprioception means measurements of the robot itself, including joint positions and velocities. A policy may require observation history because measurements reveal only part of the physical state. “Controller” and “policy” overlap in some robotics usage; the separation here is a useful position-command example, not a universal taxonomy. Learned modules, classical planning, and feedback control can coexist in one stack.

- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)

## 3. A task goal becomes a sequence of geometric goals

**Section: Foundations**

MIT’s pick-and-place chapter deliberately begins with ideal object-pose measurement to isolate geometry. The four gripper poses show why even a seemingly simple task needs an approach pose, a grasp, transfer, and release. Real perception must estimate object and tool poses from measurements, and contact/friction may remain uncertain. The idealized chapter assumes the object stays fixed in the world while ungrasped and fixed relative to the gripper while held. These are teaching assumptions, not general laws of manipulation. The picture introduces the need for target poses before discussing learned policies.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)

## 4. Joint angles and tool pose describe different things

**Section: Foundations**

A configuration q is the set of the robot’s joint coordinates. The tool pose is its position and orientation relative to a reference frame. Forward kinematics computes pose from configuration; inverse kinematics seeks a configuration achieving a target pose. IK may have no solution or multiple solutions, and a valid endpoint does not establish a collision-free trajectory. The planar arm is an authored schematic, not the ALOHA robot. Revolute joint coordinates commonly use radians; prismatic coordinates use length. A tool-position target and a vector of desired joint angles are different action interfaces.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)

## 5. A vector needs a coordinate frame

**Section: Foundations**

The MIT figure shows frames attached to bodies. The right-hand illustration is a two-dimensional slice with world and tool axes sharing an origin for clarity; in general origins differ as well. A displacement expressed in one frame changes coordinates through rotation, while a point-position conversion generally includes translation. A numeric target must identify its reference frame, orientation convention, units, and whether it is absolute or relative. Scaling data by a mean and standard deviation is numerical normalization, not geometric frame conversion. Do not infer those conventions from the length of an array.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [Modern Robotics · Transforms](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-3-1-homogeneous-transformation-matrices/)

## 6. One slow decision can produce many fast commands

**Section: Foundations**

All rates and latencies on this page are invented for teaching. The camera icon marks an image available at the beginning of the first model call. Its result arrives 80 milliseconds later; during that interval the system executes an earlier target or a hold behavior. The result is then played as commands at 50 Hz. The joint servo continues to use rapid sensor feedback, even while model computation is in progress. Servo ticks are one millisecond apart. Actual systems must measure sensor, transport, computation, scheduling, and actuation delays. UMI explicitly aligns captured observations and future execution timestamps. Some simulators pause during inference, which changes the physical-time interpretation.

- [UMI · Data and timing](https://arxiv.org/html/2402.10329)
- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)

## 7. Watch one recorded demonstration

**Section: Robot demonstration**

Play the actual simulated ALOHA insertion episode before interpreting the data. This is episode 0 of lerobot/aloha_sim_insertion_human, not a rollout generated by our lesson or a physical-robot video. The original records are nominally 50 Hz. The embedded offline video samples every fifth source frame, giving ten displayed frames per second and covering nominal episode times 0 through 9.9 seconds. The encoded container lasts about 10.24 seconds; do not infer sensor timestamps from playback time. Ask which observations reveal approach, alignment, grasp, and insertion. The native controls allow pause and replay, while print uses a poster image.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)

## 8. One frame is more than an image

**Section: Robot data**

The image is actual episode 0 frame 0. The LeRobot record contains observation.state and action as float32 arrays of length fourteen: six left-arm coordinates, left gripper, six right-arm coordinates, right gripper. RGB storage is height 480 by width 640 by three channels; a loader may return channel-first tensors. Episode 0 has 500 records at nominal 50 Hz. The current task metadata string is “Insert the peg into the socket.” This is a dataset label, not per-frame speech or evidence that original ACT consumes language. A state field here describes measured robot variables, not the entire physical world. LeRobot is storage/access machinery rather than a universal action schema.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [LeRobot · Format](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 9. The command is not the outcome

**Section: Robot data**

These are rounded values from the first two stored rows, not synthetic dynamics. The connected segments only join sampled values; they do not establish the continuous response between samples. The target is an absolute desired revolute-joint coordinate in radians. The converter copies image, qpos, and action at the same index and constructs nominal time as frame index divided by FPS. The original ACT simulation loader starts the target chunk at that index. Therefore call it a stored target at nominal frame time, not a documented human wall-clock command. The grippers use a separate normalized opening convention, nominally zero closed and one open. A command does not guarantee the next measured position.

- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)
- [ACT · Simulation API](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)

## 10. Turn a trajectory into training pairs

**Section: Robot data**

The green observation at t is paired with the next H stored commands, beginning at t under the verified simulation alignment convention. H=4 is illustrative. The action dimension d_a is independent of H; it is fourteen for the ALOHA example. Input history or optional language can be added when the selected method uses them. Original ACT uses images and joint observations, without language. A different acquisition pipeline may need a different action/observation shift. Split evaluation data by whole episodes and relevant scenes or tasks to avoid neighboring-frame leakage. At episode ends, either drop an incomplete window or pad it with a validity mask; do not silently continue into the next episode. The final-record next.done flag is not itself a success label.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [ALOHA · Converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)

## 11. How does a person provide a demonstration?

**Section: Demonstrations**

The ALOHA photograph is a crop of the original ACT setup figure. The person backdrives leader arms; follower arms mirror their joint configuration, and leader positions provide command labels while follower positions provide measurements. The UMI photograph is cropped from its hardware figure and shows a handheld gripper with camera/IMU rather than a leader robot. UMI estimates gripper trajectories and handles latency/alignment when transferring them to a robot. Neither raw human video nor a shared file format automatically supplies executable robot labels. DROID is another collection route, using a common real robot across diverse environments; Open X-Embodiment aggregates many datasets and embodiments. Their action units, frames, absolute/relative conventions, and timing must remain explicit. Original source assets are preserved; only the displayed viewports are cropped.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)
- [Open X-Embodiment](https://arxiv.org/html/2310.08864v6)

## 12. Behavioral cloning is supervised learning of actions

**Section: Imitation learning**

Behavioral cloning treats demonstrated observation/action pairs as supervised data. The policy can represent a conditional action distribution rather than only a point prediction. Negative log likelihood is a common explicit-density formulation; under a fixed-variance Gaussian assumption it reduces to squared-error regression up to constants and scale. The same idea applies to an action chunk A rather than a single action a. Diffusion and flow-matching policies learn conditional distributions with their respective denoising or vector-field objectives; this equation is not a claim that every policy directly evaluates tractable likelihood. No reward function is needed for this imitation objective. Reinforcement learning will later change the learning signal to cumulative reward.

- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 13. Two valid actions can have an invalid average

**Section: Action distributions**

Pause on the first frame and ask for the squared-error optimum. The second frame draws a pointwise mean between two equally weighted geometric command routes. That average intersects the obstacle although each demonstrated route avoids it. This is an authored illustrative command-space example, not a measured robot rollout. It motivates representing more than a conditional mean. It does not establish that all real datasets are strongly multimodal or that multimodality explains every empirical gain of generative policies. Temporal correlation, optimization, robustness, and action representations also matter. A larger deterministic network alone does not change the distribution family.

- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)
- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 14. Learn to turn noise into an action

**Section: Action distributions**

This is the broad generative-policy intuition used by the Stanford lecture: start from a distribution that is easy to sample and learn a conditioned transformation toward demonstrated actions. The drawn curves are illustrative one-dimensional marginals, not the full high-dimensional chunk distribution. The observation is held fixed while generating a candidate action. Different noise samples can lead to different valid candidates. This page intentionally precedes the particular diffusion algorithm. Diffusion and flow matching differ in their training objectives and sampling formulations; neither is merely training a deterministic regression from arbitrary independent noise directly to an arbitrary demonstration.

- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)
- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 15. The learner changes its own input distribution

**Section: Closed-loop learning**

The pale region represents observations or states encountered by demonstrations; this is not a physical tool path or a quantitative density contour. The learned policy can leave that region after an imperfect action. Its next input is therefore sampled from the distribution induced by the learner, which may differ from the expert distribution. Errors can compound. The key contrast with ordinary held-out prediction is that the policy’s outputs affect its future inputs. Do not quote a worst-case horizon-squared bound without its assumptions, or claim that chunking removes distribution shift. Offline action loss is useful, but task completion and recovery require rollouts.

- [DAgger · Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 16. Ask the expert about the learner’s mistakes

**Section: Closed-loop learning**

DAgger iterates between running the current policy, querying an expert for actions at the visited states, aggregating those labels with prior data, and retraining. The green correction arrow is an expert action label at a learner-visited state; it is not a guarantee that the current learner already executed the correction. The original algorithm can mix expert and learner control during collection. Human-gated intervention is a related practical variant in which a person takes over and supplies corrective demonstrations. Labeling and takeover are different interfaces. Expert availability and the cost of collecting new trajectories remain part of the method’s assumptions.

- [DAgger · Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 17. Predict a sequence of commands in one call

**Section: Action chunking**

H is the number of future commands and d_a is the dimension of one command. The illustrated output has six commands and fourteen coordinates each; those numbers are pedagogical and use the ALOHA action dimension. Green dots mark gripper coordinates to reinforce that coordinates can have different units and meanings. A chunk is a structured command array, not six model conversations and not six predicted physical states. The observation can include one or more cameras, proprioception, and sometimes a history or language instruction. Predicting a chunk does not yet determine how much of it will be executed before observing again.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 18. Execute a prefix, then update the plan

**Section: Action chunking**

The first frame predicts six commands. The second executes only two, which turn green. A new observation at step two supports a fresh chunk; unused predictions from the first chunk are crossed out. At step four the process repeats. Subscripts identify execution steps and parenthesized superscripts identify prediction/observation steps, so two proposals for the same execution step can have different values. The drawing is logical timing and omits inference latency. H is the prediction horizon and h the executed prefix, with h at most H. This receding-horizon pattern is used in Diffusion Policy. Larger h can reduce model calls but delay visual correction. Rapid joint feedback may remain active inside the prefix.

- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 19. Overlapping chunks can be combined

**Section: Action chunking**

This is ACT’s temporal-ensembling idea, distinct from simply executing a prefix and discarding the remainder. Querying at successive observation steps creates overlapping chunks. At execution step two, the highlighted column contains predictions made from observations zero, one, and two. ACT combines predictions aligned to that same execution time using a weighted average. It does not average adjacent execution times as an ordinary temporal smoothing filter would. This adds inference-time computation and controls how quickly new observations influence the command. The schematic uses short chunks for readability, not the original reported horizon. It makes no claim that model inference is instantaneous.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)

## 20. ACT: Action Chunking with Transformers

**Section: Action policies**

Original Action Chunking with Transformers learns from ALOHA demonstrations. In the real system its observation includes four RGB camera views and fourteen follower joint/gripper measurements. The output is a chunk of desired joint positions. ACT trains a conditional VAE: an encoder sees demonstrated actions and joint observations to infer a latent style variable, while the transformer decoder predicts the action sequence. Reconstruction and KL terms train the model. At inference the encoder is removed and z is set to the prior mean, zero. The displayed architecture is a simplified inference path; it deliberately does not draw the full training encoder. Original ACT has no language input. The simulated example earlier can use fewer camera views, so do not infer its schema from this real-system architecture.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)

## 21. Diffusion Policy generates an action chunk

**Section: Action policies**

Diffusion Policy models a conditional distribution of action sequences given recent observations. Observations are conditioning inputs, not part of a jointly generated future image/state sequence. After predicting a horizon, the system executes a prefix and replans from new observations. The horizon sizes in this diagram are illustrative. The method can use visual observations or suitable low-dimensional observations, and its action coordinates depend on the robot interface. It need not contain an explicit world model. The next page opens the repeated denoising computation inside the circle.

- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)

## 22. Refine a candidate command sequence

**Section: Action generation**

Each panel shows one coordinate across a future action chunk, not the whole H-by-d_a array. The panels are successive generative refinements conditioned on the same observation. Their horizontal axes are future command offsets; moving from panel to panel is a model-computation iteration, not the robot advancing through those commands. The plotted values are authored illustrations, not samples from a trained policy. During diffusion training, demonstrated chunks are corrupted at sampled noise levels and a conditioned network learns to predict the added noise. During sampling, a schedule of learned denoising updates transforms noise toward an action sequence. Exact samplers may include stochastic terms and differ in parameterization. Flow matching, introduced later, uses a related but distinct vector-field objective.

- [Diffusion Policy · §2](https://arxiv.org/html/2303.04137v5)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 23. A generated target still needs feedback control

**Section: Execution**

The diagram isolates a joint-position servo: compare a desired configuration q* with the measured q, use the error in a control law, and apply motor effort. The minus sign represents subtracting measured position from desired position; real controllers can also use velocity, integral terms, feedforward dynamics, torque limits, and other signals. We do not claim every robot has the same PID implementation. This lower-level feedback can remain active while a learned policy executes an open-loop visual prefix or computes its next chunk. Physical motion depends on both control and dynamics, including contact and disturbance, so generated command arrays are not guaranteed outcomes. A tool-pose command may additionally require IK or another Cartesian controller.

- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
- [ACT · §IV](https://arxiv.org/html/2304.13705v1)

## 24. What would you need to execute this array?

**Section: Transition to VLAs**

Use this as a short audience check before adding vision-language models. An array dimension is insufficient. Ask for coordinate meanings and ordering, units, reference frame, absolute versus relative convention, gripper semantics, intended execution times, limits, and the controller or skill executor consuming it. The numbers are invented and do not correspond to an actual robot command. A VLA can learn the mapping into this interface, while an LLM might instead select a skill or produce an executable program. None of those labels eliminate the need to define sensing, feedback, and the action contract. The lesson now turns to what pretrained visual and language representations add.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 25. What does a vision–language model predict?

**Section: Visual language to action**

A VLM accepts visual inputs and text prompts and produces predictions such as a description, an answer, or serialized boxes. The prompt shown here is an authored illustration, not a model run or a measured result. The photograph is a mock kitchen evaluation environment from π0.5 Figure 6. This page establishes the interface before showing the encoder and Transformer. PaliGemma is a representative generative VLM; it is not the only architecture called a VLM. Calibrated robot commands require an additional action representation and robot experience.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 26. Images become input tokens

**Section: Visual language to action**

This is an editable architectural illustration, not a model-generated example. PaliGemma combines the SigLIP ViT-So400m vision encoder, a linear projection, and Gemma-2B. The projection aligns feature dimensions, rather than converting image patches into vocabulary IDs. Its image and prompt prefix uses bidirectional attention, while the generated answer suffix uses autoregressive attention. The model can emit natural language and serialized structured outputs such as boxes. Its base pretraining is not chat instruction tuning. Recognition and semantic representations do not by themselves specify a robot action coordinate system, scale, or execution controller. OpenVLA uses a different VLM backbone: Prismatic, with DINOv2 and SigLIP features plus Llama 2. The schematic introduces the reusable three-part VLM structure, not a claim that every VLA uses PaliGemma. Source: PaliGemma §3.1 and Figures 1–2.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)

## 27. A numerical action can become a token

**Section: Discrete action policies**

The top row is a deliberately authored quantization example, not an observed OpenVLA training value. Mapping +0.0200 m from [−0.10,+0.10] to [−1,1] gives +0.20. With 256 uniform bins, floor(256×0.60)=153, and decoding to that bin center gives approximately +0.01992 m. A token index is not a physical displacement. Original OpenVLA fine-tunes Prismatic-7B on 970k demonstrations, predicts one 7D end-effector/gripper action from one image and an instruction, and computes next-token cross-entropy on action tokens only. It uses 256 bins per coordinate between the training dataset’s first and 99th percentiles and repurposes 256 vocabulary entries. It does not have the chunking, proprioceptive input, or multiple-camera features of later derivatives. RT-2 co-fine-tunes on robot data and web vision-language tasks, uses an eight-value encoding including termination, and constrains output vocabulary for action prompts. Both generate discrete action codes autoregressively. A robot-specific decoder and controller still execute the resulting numeric command. Sources: OpenVLA §§3.1–3.3 and RT-2 §§3.2–3.3. The tick marks shown are a sparse schematic of the quantization interval, not all 256 bin boundaries. The dataset defines frame, units, bounds and gripper conventions. This slide isolates conversion of one coordinate before learning and deployment.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)

## 28. Train the VLM on demonstrated action tokens

**Section: Discrete action policies**

This equation is next-token cross-entropy on an action-token sequence z. I is the image and ℓ is the instruction; previous action-coordinate tokens condition later tokens within the same command. Original OpenVLA evaluates its cross-entropy loss on action tokens only and fine-tunes a Prismatic-7B VLM. The diagram’s token values are illustrative rather than an actual robot record; g denotes the gripper coordinate’s token. RT-2 additionally co-fine-tunes on original web vision-language examples. It constrains decoding to action tokens when prompted for actions, while retaining language outputs for language tasks. Do not attribute RT-2’s precise data mixture to OpenVLA or imply that original OpenVLA has chunking or proprioceptive input.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)

## 29. Deploy a vision–language–action (VLA) model

**Section: Discrete action policies**

The top row is a deliberately authored quantization example, not an observed OpenVLA training value. Mapping +0.0200 m from [−0.10,+0.10] to [−1,1] gives +0.20. With 256 uniform bins, floor(256×0.60)=153, and decoding to that bin center gives approximately +0.01992 m. A token index is not a physical displacement. Original OpenVLA fine-tunes Prismatic-7B on 970k demonstrations, predicts one 7D end-effector/gripper action from one image and an instruction, and computes next-token cross-entropy on action tokens only. It uses 256 bins per coordinate between the training dataset’s first and 99th percentiles and repurposes 256 vocabulary entries. It does not have the chunking, proprioceptive input, or multiple-camera features of later derivatives. RT-2 co-fine-tunes on robot data and web vision-language tasks, uses an eight-value encoding including termination, and constrains output vocabulary for action prompts. Both generate discrete action codes autoregressively. A robot-specific decoder and controller still execute the resulting numeric command. Sources: OpenVLA §§3.1–3.3 and RT-2 §§3.2–3.3.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 30. Web knowledge can select a familiar motor skill

**Section: Visual language to action**

This photograph is cropped from RT-2 Figure 2, which shows semantic generalization examples. The task uses a familiar physical manipulation skill with a target indicated through a semantic concept. The research distinction is between richer interpretation/selection and arbitrary new physical competence. RT-2 explicitly limits its motor skill distribution to robot experience even when web knowledge changes which object or destination it chooses. These particular illustrative demonstrations are not a common leaderboard against OpenVLA or later policies. RT-1 is a useful predecessor that scaled language-conditioned robot imitation; it was not a conversational VLM fine-tuned into a robot. Source: RT-2 Figure 2, §§3–5.

- [RT-2 §3](https://robotics-transformer2.github.io/assets/rt2.pdf)
- [RT-1 §5](https://robotics-transformer1.github.io/assets/rt1.pdf)

## 31. π0 adds a continuous action expert

**Section: Continuous action policies**

π0 initializes its visual-language weights from PaliGemma and adds a randomly initialized 300M-parameter action expert, giving roughly 3.3B parameters in total. The robot configuration q_t and action slots use robotics-specific expert weights. Image and text slots use the VLM weights, and attention provides the conditioning connections. This is a simplified computational diagram, not a separate large-language planner calling a black-box motion module. In the original π0, q_t is projected continuously. The original π0.5 changes proprioception to discretized text input, which will be distinguished later. The flow model predicts a vector field on an H=50 action chunk; ten Euler steps generate a chunk in the reported experiments. These ten iterations are generation computation, not ten physical servo cycles. During inference, prefix attention keys/values can be cached while the suffix is updated. Prefix representations are fixed within this sampling call, not permanently frozen model weights. π0 trains on diverse robot demonstrations and then uses curated downstream post-training; some long tasks use an additional high-level VLM. Sources: π0 v4 §IV, Figure 3, Appendices B/D, and CS224R lecture 17 p.6.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 17](https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf)

## 32. One sample is a whole action chunk

**Section: Continuous action policies**

The matrix schematically shows action-coordinate rows and physical-time columns. A_t is mathematically stored as H×d_a in this deck; the display transposes it for a horizontal time axis. Entries are intentionally blank, rather than fabricated robot measurements. Each column corresponds to one physical command. π0 uses H=50 and bidirectional attention across continuous action slots. Every flow-generation iteration updates the entire chunk; the action slots are not generated as 50 independent language-model conversations. Prediction horizon, executed prefix, physical control rate and the number of flow iterations are distinct. The next three slides explain generation time τ separately.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)

## 33. Construct a flow-matching training example

**Section: Flow matching**

The line is a two-dimensional schematic of a path in the much higher-dimensional action-chunk space. Sample a demonstrated chunk A and an independent Gaussian ε with the same shape, then draw τ according to the training schedule. Interpolate X_τ=(1−τ)ε+τA. The derivative of this particular linear path is A−ε, which is known during training. We can therefore supervise a local velocity without simulating a robot or unrolling a generated trajectory. This is the noise-to-action convention in corrected π0 v4. The drawing’s position τ≈0.42 is illustrative. Its geometry is not an end-effector path in the room. CS224R lecture 2 pp.12–14 motivates the same construction.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 34. Fit the vector field, not a single mean action

**Section: Flow matching**

A is a demonstrated H×d_a action chunk and ε is an independent Gaussian tensor of the same shape. Sample τ from the training schedule, interpolate X_τ=(1−τ)ε+τA, and regress the network’s vector field against A−ε. The expectation is over demonstrated context/chunk pairs, independent noise, and generation time. Under squared error, the optimal field at a noisy point is the conditional mean of these local velocity targets; integrating this field transforms an entire noise distribution into an action distribution. This does not require the output action distribution to be Gaussian or to equal a single conditional mean action. Training can sample one τ per example rather than unrolling the full inference trajectory. At inference no demonstrated A is available: sample X_0 and repeatedly apply the learned field with observation/instruction fixed during the call. π0 v4 uses ten Euler steps and a training τ distribution emphasizing noisier times. The equations consistently use noise-to-action orientation. Older π0/π0.5 texts contain an inconsistent target sign, so the teaching equations follow the corrected π0 v4 §IV. The field has units of normalized action per unit generative time, not necessarily physical end-effector velocity. CS224R lecture 2 pp.11–14 gives the corresponding teaching derivation. The statistical distinction is important: squared-error regression here estimates the conditional mean of local velocity targets at X_τ, τ, and context. It is not direct regression from the observation to one mean action. The expectation averages over data pairs, independent Gaussian noise and generation time. Expressiveness of the resulting sampled action law follows from transforming a noise distribution. It does not imply generative methods always beat robust direct regression in every task or dataset.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 35. Follow samples through the flow

**Section: Flow matching**

Use Next step to advance generation time while retaining the same initial noise samples. This is an analytic demonstration, not a trained model, robot trajectory, empirical benchmark, or visualization of π0 hidden states. Conditional on a fixed observation, let the data action A equal −2 or +2 with equal probability and ε~N(0,1). The interpolating law is an equal mixture of Gaussians with means ±2τ and standard deviation 1−τ. Its conditional flow is v(x,τ)=[2 tanh(2τx/(1−τ)^2)−x]/(1−τ). This follows from E[A|X_τ=x] and ε=(x−τA)/(1−τ). We integrate this known field using Euler δ=0.01 to τ=0.99. The paths illustrate how initial Gaussian variation separates into two modes. The action coordinate uses arbitrary units; horizontal τ is generation time and does not represent robot execution time. A learned policy approximates a conditional field from demonstrations, and a real action chunk has many coordinates. π0’s reported sampler uses ten Euler steps; the finer analytic demonstration is chosen for a clear teaching picture. The discrete endpoint law makes the field singular exactly at τ=1, so the display stops at 0.99. The control is a discrete step-through, not live numerical inference.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 36. A household task extends beyond one grasp

**Section: π0.5: heterogeneous supervision**

The photograph strip is π0.5 Figure 2, cropped to the photographs while retaining their order. It shows a robot cleaning a previously unseen kitchen. The brief labels paraphrase tasks shown in the original caption. The research challenge combines object recognition, selecting useful subtasks, and executing motor behavior under changed geometry. This is demonstration evidence, not a measurement that every task succeeds in every home. The task-progress plot is presented separately after the training recipe. The official explanation distinguishes transferring learned skills to new settings from learning arbitrary new physical skills without relevant robot experience. Sources: original π0.5 Introduction, §V, Figure 2, and official project explanation.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [π0.5 explanation](https://www.pi.website/blog/pi05)

## 37. Robot data broaden the physical experience

**Section: π0.5: heterogeneous supervision**

These are cropped examples from π0.5 Figure 4, not a newly collected dataset. MM supplies mobile household demonstrations, around 400 hours in about 100 home environments. ME supplies demonstrations from non-mobile robots in diverse homes; these robots are easier to transport but differ from the target mobile platforms. Laboratory cross-embodiment data include single/dual arms and static/mobile robots, with public Open X-Embodiment data as an additional component. The pictured lab examples are only part of this category. Action targets require each dataset’s control-mode, units, normalization and embodiment conventions. These physical sources complement semantic supervision rather than duplicating it. The full numeric mixing ratios are not disclosed; 97.6% of first-phase examples come from sources other than the target mobile household subset. Source: π0.5 §IV-C and Figure 4.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 38. Semantic data teach what to do next

**Section: π0.5: heterogeneous supervision**

The three images are crops from π0.5 Figure 4. The bedroom illustrates high-level supervision: manual subtask labels and relevant bounding boxes accompany robot observations. The pie illustrates a web visual-question-answering example; the broader web mixture also includes captioning and localization. The human beside the robot illustrates verbal instruction demonstrations collected after a low-level policy exists. An expert chooses useful next subtask commands during execution, yielding labels for high-level decisions. These are not newly inferred joint commands extracted from speech alone. VI enters post-training; high-level labels and web examples participate in both phases. The short input/output annotations paraphrase the data roles rather than claiming the displayed images constitute newly run model tests. Source: π0.5 §§IV-C/D and Figure 4.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 39. π0.5 first learns with discrete tokens

**Section: π0.5: training**

This diagram follows the original 19-page π0.5 paper rather than treating current OpenPI code as an exact reproduction. First, pretrain for 280k gradient steps with next-token prediction over text, boxes, and compressed FAST action tokens. This phase uses mobile manipulation, multi-environment non-mobile data, laboratory cross-embodiment data including OXE, high-level labels, and web data. Second, add a randomly initialized action expert and train another 80k steps with joint cross-entropy and flow matching, using flow-loss coefficient 10.0. Post-training action data use MM/ME successful episodes filtered by a length threshold. It retains web and relevant high-level data, adds verbal instruction examples, and omits laboratory cross-embodiment data. FAST discrete action outputs and flow action outputs are masked from attending to one another. FAST compresses a chunk; it is not identical to OpenVLA’s independent 256-bin coordinates. Original π0.5 discretizes proprioceptive state into text input, unlike π0’s continuous state projection. At inference, the system uses autoregressive subtask text followed by flow sampling. Current OpenPI README describes π0.5 trained with knowledge insulation and only supports the flow-matching head for π0.5 training/inference; do not assume downloading it reproduces this entire household hierarchy. Sources: π0.5 Figure 3 (p.4), Figure 4 (p.6), §§IV-A–D, and OpenPI README.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 40. Then add efficient continuous action generation

**Section: π0.5: training**

This diagram follows the original 19-page π0.5 paper rather than treating current OpenPI code as an exact reproduction. First, pretrain for 280k gradient steps with next-token prediction over text, boxes, and compressed FAST action tokens. This phase uses mobile manipulation, multi-environment non-mobile data, laboratory cross-embodiment data including OXE, high-level labels, and web data. Second, add a randomly initialized action expert and train another 80k steps with joint cross-entropy and flow matching, using flow-loss coefficient 10.0. Post-training action data use MM/ME successful episodes filtered by a length threshold. It retains web and relevant high-level data, adds verbal instruction examples, and omits laboratory cross-embodiment data. FAST discrete action outputs and flow action outputs are masked from attending to one another. FAST compresses a chunk; it is not identical to OpenVLA’s independent 256-bin coordinates. Original π0.5 discretizes proprioceptive state into text input, unlike π0’s continuous state projection. At inference, the system uses autoregressive subtask text followed by flow sampling. Current OpenPI README describes π0.5 trained with knowledge insulation and only supports the flow-matching head for π0.5 training/inference; do not assume downloading it reproduces this entire household hierarchy. Sources: π0.5 Figure 3 (p.4), Figure 4 (p.6), §§IV-A–D, and OpenPI README.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 41. Choose a subtask, then generate its motion

**Section: π0.5: inference**

The equation is the paper’s model factorization, not the unconstrained probability chain rule: its action distribution depends on the inferred subtask c_t and current observation, but not directly on the overall task prompt ℓ. The same model supports autoregressive semantic output and continuous flow-based action output with different expert weights. In the actual system high-level inference uses four camera views, while low-level inference uses the wrist and forward cameras; o_t denotes available observation context rather than asserting identical camera subsets. Proprioception includes joints, grippers, lift and base velocity. The mobile platforms have 18 or 19 action dimensions. They issue arm/gripper/lift targets and base velocities at 50 Hz with action chunking; simple PD controllers track the targets. The paper reports no additional trajectory planner or collision detector in this execution path. Ten flow denoising steps generate the 50-step chunk. A 50 Hz command stream does not mean the entire VLM generates a semantic subtask 50 times per second. The official explanation describes a one-second chunk. Distinguish high-level demonstration labels used in training from inferred subtasks used at deployment. Sources: π0.5 §§IV-A/B/E and official explanation.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [π0.5 explanation](https://www.pi.website/blog/pi05)

## 42. The hierarchy separates two conditional decisions

**Section: π0.5: inference**

The equation is the paper’s model factorization, not the unconstrained probability chain rule: its action distribution depends on the inferred subtask c_t and current observation, but not directly on the overall task prompt ℓ. The same model supports autoregressive semantic output and continuous flow-based action output with different expert weights. In the actual system high-level inference uses four camera views, while low-level inference uses the wrist and forward cameras; o_t denotes available observation context rather than asserting identical camera subsets. Proprioception includes joints, grippers, lift and base velocity. The mobile platforms have 18 or 19 action dimensions. They issue arm/gripper/lift targets and base velocities at 50 Hz with action chunking; simple PD controllers track the targets. The paper reports no additional trajectory planner or collision detector in this execution path. Ten flow denoising steps generate the 50-step chunk. A 50 Hz command stream does not mean the entire VLM generates a semantic subtask 50 times per second. The official explanation describes a one-second chunk. Distinguish high-level demonstration labels used in training from inferred subtasks used at deployment. Sources: π0.5 §§IV-A/B/E and official explanation.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 43. How much of each task did the robot complete?

**Section: π0.5: evidence**

This reproduces original π0.5 Figure 7(b). §V-A specifies three held-out real homes, with kitchen and bedroom tasks in each. Thus the caption’s three kitchens plus three bedrooms refer to six room environments within three homes. The y-axis is average task progress, not binary episode success. The rubric roughly credits completed task steps; moving half the dishes can receive partial credit. The caption reports ten trials per task/environment. Appendix B describes canceled-episode exclusions, so do not derive a larger exact total from the bar count. Many quantitative trials last 2–5 minutes; longer 10–15-minute demonstrations are separate evidence. This is an environment-generalization result, not a measurement of arbitrary unseen motor skills. Source: π0.5 §V-A, Figure 7 and evaluation appendix.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 44. Training labels help even without an explicit hierarchy

**Section: π0.5: evidence**

This reproduces original π0.5 Figure 13 in controlled mock environments. Full π0.5 uses high-level labels and explicit subtask inference. The implicit-HL variant keeps the high-level training examples but omits that inference at runtime; no-HL removes both. The full-versus-implicit comparison is labeled p=0.144, so the observed mean gap should not be presented as a statistically established advantage at conventional thresholds. High-level supervision remains useful even without explicit high-level inference. The entire plot retains no-web, no-verbal-instruction, GPT-4 high-level, and human high-level ablations, with uncertainty and reported p-values. Their definitions are in §V-E. The comparison uses the full low-level inference process with high-level variants. The full model outperforming this particular human high-level baseline does not establish general human-level ability. Metric: task progress. Source: π0.5 §V-E and Figure 13.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 45. Generalization still has a physical scope

**Section: π0.5: scope**

The photographs show real evaluation rooms from π0.5 Figure 6; they are not labeled failure episodes. The failure classes come from the paper’s discussion: difficult or unfamiliar handles, partial observability such as occluded spills, and distracted high-level choices such as repeated drawer opening and closing. Longer-context memory and complex user preferences remain limitations. The paper and official explanation concern transferring learned motor skills into unfamiliar environments, not arbitrary new physical skills. The current OpenPI π0.5 release is trained with knowledge insulation and supports only the flow-matching head for training and inference. It should not be equated with downloading a complete reproduction of the paper’s hierarchical mobile household system.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [OpenPI release](https://github.com/Physical-Intelligence/openpi)

## 46. Experience can improve an imitation-trained policy

**Section: Beyond behavioral cloning**

Behavioral cloning learns from demonstrated actions. Reinforcement learning optimizes expected cumulative reward and can exploit feedback about what happened after actions. A pretrained VLA supplies useful exploration and representations but generally does not supply a trained critic/value function. This diagram is an authored overview, not the exact algorithm of π0.5 and not a claim that π0.5 itself used this RL stage. CS224R lecture 17 distinguishes online updates from iterated offline learning over large rollout batches, describes value-guided supervised policy updates, and explores latent steering and residual/edit policies. These methods make different compromises; we do not teach them as interchangeable or guaranteed improvements. Human interventions can supply useful imitation data and coexist with RL. Large models incur substantial gradient-update costs, physical resets and rollouts can dominate collection time, and diffusion/flow action generators plus chunking complicate direct use of algorithms built around tractable per-step policy probabilities. Avoid transferring the lecture’s informal “80% plateau” into a universal reliability law or reading π0.5’s task-progress metric as success probability. Sources: CS224R lecture 17 pp.5–15 and 20–29, lecture 2 on online interventions.

- [CS224R, lecture 17](https://cs224r.stanford.edu/slides/17_cs224r_rl_vlas_2026.pdf)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 47. SayCan: choose a useful, feasible skill

**Section: Language models and robot control**

SayCan combines language-model usefulness with learned value-function feasibility and selects from a predefined repertoire of language-conditioned robot skills. The diagram is conceptual, without invented numerical scores. The selected skill has learned low-level control. Scene grounding enters through the value functions; do not imply the original language prompt receives a rich visual feedback stream. The spill instruction and candidate skill paraphrase the paper’s motivating example. The original study includes physical mobile manipulation, unlike the simulation results later in this section.

- [SayCan · 2022](https://say-can.github.io/)

## 48. Code as Policies: let code close the loop

**Section: Language models and robot control**

The code shown here is authored pseudocode, not a verbatim model output or a promise that a specific API exists. Code as Policies uses language-model-generated programs to combine perception and control APIs; these programs can contain loops, geometry computations, and feedback. Running a generated program is distinct from querying an LLM for every servo command. The physical tabletop image is a crop of Figure 2a from the paper, rendered from PDF page 2. It illustrates the original system’s setting and is not a test of the pseudocode on this slide. The full project provides demonstrations and prompts.

- [Code as Policies · 2022](https://code-as-policies.github.io/)

## 49. VoxPoser: turn language into spatial objectives

**Section: Language models and robot control**

VoxPoser uses language models and perception tools to generate code that composes three-dimensional value maps. Affordance maps encourage desired interactions and constraint maps discourage forbidden regions; motion planning and low-level control execute the resulting objectives. Cached code can be re-evaluated with new observations. The original method figure is cropped to the map and planning panels, omitting its dense code panel. This slide does not claim that the LLM directly outputs motor torques, or that all perception and control components were trained from scratch or without robot experience. RGB-D denotes color plus depth.

- [VoxPoser · 2023](https://voxposer.github.io/)

## 50. A numeric target still needs a robot controller

**Section: Language models and robot control**

This conceptual loop describes direct end-effector control in the recent RoboDojo study. The model requests Cartesian gripper pose and opening; the surrounding stack validates actions, transforms the target, solves inverse kinematics and plans a joint trajectory. The physical controller or simulator then executes the joint path and returns new sensor information. A target is not a torque. Inverse kinematics maps an end-effector pose to robot joint configurations. The reported RoboDojo interface provides move_eef and give_up, not a semantic pick-and-place skill library. No learned motor policy supplies the direct method’s motion.

- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)

## 51. RoboDojo: images are part of the policy interface

**Section: Language models and robot control**

This image is the paper’s three-view grounding illustration for a make-kong task, not a physical robot photo or a representative sample of all benchmark tasks. The undated gpt-6-astra alias is frozen and used at medium reasoning effort. Inputs include three RGB cameras, proprioception, wiki-derived goal and scoring descriptions, full text history, and the latest two image turns. The direct policy does not receive depth, privileged object poses, or a reward oracle. These observation and instruction choices are part of the measured system and can differ from published learned-policy baselines. See §§3.2–3.3 of the primary paper.

- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)

## 52. RoboDojo performance varies sharply by task type

**Section: Language models and robot control**

These are Table 1 native full-task success rates: 30.50, 4.00, 8.25, 38.67 and 31.00 percent, with five-category macro average 22.48 percent. The separate score metric is 28.97 and must not be relabeled as success rate. The paper evaluates 42 tasks with 50 trials each and one seed. Read the graph as a descriptive category pattern, not proof of a causal reasoning or memory mechanism. The pinned release README discloses two targeted task-cell replacements and a 170-call budget; the paper states 100 by default. Those campaign details remain unreconciled, so this is explicitly a paper-reported result, not an independently reproduced leaderboard rank. Input access differs from policy baselines, which are intentionally omitted here.

- [RoboDojo study §3–4](https://arxiv.org/html/2609.24170v1)
- [Released run notes](https://github.com/RoboProbe/RoboProbe/blob/3c0bc68383a75bf3fa2d7bfc5ab1d1241c7e171e/results/l3_inspect_eef_official_2100/README.md)

## 53. Astranomy: the harness changes the control system

**Section: Language models and robot control**

Astranomy’s Direct EEF system chooses numeric targets using a short observation window. The agentic Codex system can maintain persistent history and use code and tools. The orchestration system supervises Cosmos3-Nano-Policy proposals. These are systems with different histories and harnesses, not three controlled doses of computation applied to an otherwise identical policy. All use the report’s 120-task simulation suite. A controller remains below each high-level interface. The study’s public report is retrieved and supports these claims; the linked implementation endpoint was not publicly retrievable in the audit. Model labels are those used in the report, without assuming an undisclosed dated checkpoint.

- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)

## 54. Astranomy reports system-level success rates

**Section: Language models and robot control**

The report evaluates Direct EEF at 66.3%, the persistent agent at 74.8%, and policy orchestration at 66.5% full-task success, over 120 tasks with five episodes each, seeds 0–4. It uses three camera views at 384×216 and medium effort. Physics is paused during inference, with 15 Hz action execution and task-specific action-step limits; there is no wall-time or retry cap. These system-level rates are not evidence that computation alone caused a gain. Five episodes per task also limit precision for small differences. Do not compare this suite’s rates directly with the separate RoboDojo 42-task campaign or the selected ten-task hybrid experiment. No error bars are invented.

- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)

## 55. A paused simulator does not test real-time reactions

**Section: Language models and robot control**

Step through this schematic timeline. Astranomy pauses physics while a model call is in flight, so an object cannot drift during that interval. Once a command arrives, action playback runs at 15 Hz. The reported 18.56-second mean Direct EEF call latency is computed over successful episodes only and includes network delay; it is not a control-loop period or an unconditional latency estimate. Timeline widths are schematic and do not reconstruct an episode. A physical experiment, or a simulation where physics continues while the model thinks, tests a different timing regime. Embodied First Exam explicitly distinguishes paused standard mode from standard-realtime, though its other protocol defaults also need to be matched.

- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)
- [Embodied First Exam protocol](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md)

## 56. Hybrid control: accept a policy prefix or correct it

**Section: Language models and robot control**

In Systematically Exploring, released RoboDojo-task-finetuned π0.5 weights propose a 50-step joint-action sequence. Astra observes the images and proposal, then either accepts the first 1–15 policy steps or substitutes a 1–5-step end-effector correction. These are mutually exclusive branches, not two commands executed consecutively. Both pass through the robot execution stack and receive observations after execution. The direct baseline also outputs 1–5 end-effector steps. The explicit policy-training qualification matters: frozen general-model weights do not make the entire hybrid system free of robot-specific training. Here a target step refers to the paper’s direct action interface, not an unmediated torque.

- [Astra policy study §3](https://arxiv.org/html/2609.38537v1)

## 57. Hybrid control helps on this selected task subset

**Section: Language models and robot control**

The paper reports 13 of 50 full-task completions for Direct (26%) and 24 of 50 for Hybrid (48%). Each group of rectangles is an aggregate count, with filled cells placed together for legibility; rectangle positions do not depict matched per-instance outcomes. The study uses ten selected tasks and five paired initializations, with task instances, seeds, descriptions, reasoning settings, and end-effector interface held fixed. The selection is stratified by published π0.5 success and emphasizes low-success tasks: six from the lowest stratum, two from the next, one each from the remaining two. The gain is not uniform across tasks; some classification tasks favor Direct. The standalone π0.5 reference is a reweighted published number, not a locally rerun paired baseline, so it is omitted.

- [Astra policy study §3](https://arxiv.org/html/2609.38537v1)

## 58. The evaluation protocol is part of the result

**Section: Language models and robot control**

This is an authored controlled-experiment diagram, not a description shared by all cited benchmarks. Hold test cases and the action interface fixed when asking about a decision component. Explicitly declare sensor access, task descriptions, tools, memory, prior trials, code generation, resets, and adaptation. Freeze development before held-out evaluation. Report the physics clock: paused during inference or advancing while thinking. Measure elapsed time alongside native completion. A separate engineering question may legitimately allow each complete system its own interface, but that is a best-system comparison and cannot isolate the decision model alone. Embodied First Exam illustrates why mode names are insufficient: standard and standard-realtime differ in clock and can have different reset defaults.

- [Embodied First Exam protocol](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md)
- [Astranomy §2–4](https://research.nvidia.com/labs/srl/projects/astranomy/index.html)

## 59. Question: how would you compare these systems?

**Section: Research exercise**

Pause here for audience discussion. This is an authored research exercise and diagram, not a real trial or a reported interview question. The drawer moves during the model’s inference delay. Ask learners to identify information access, timing, task adaptation, and outcome variables before choosing a winner. A VLA action chunk and a general-model-plus-planner stack can differ in both learned knowledge and action interface. The question should elicit the distinction between a controlled component comparison and an engineering comparison of complete systems. Stale observations and commands matter when physics continues during computation.


## 60. Solution: evaluate the operating curve

**Section: Research exercise**

The curves are authored and schematic, with no numeric predictions or measured points. Compare repeated matched cases over meaningful resource budgets and report uncertainty at the task or scene level. Elapsed time includes model computation, tool use, and execution. State and match whether physics advances during computation, and use the same disturbance schedule. Declare training, demonstrations, prior exploration, reusable code, and resets even when equalizing them is impossible. A shared action interface isolates a decision-component difference. A best-system experiment instead lets each method use its preferred executor and must expose those design differences. Connect the discussion back to observation, action, feedback, and data from the beginning of the lecture.

- [Embodied First Exam protocol](https://github.com/mll-lab-nu/embodied-first-exam/blob/main/docs/protocol/README.md)
