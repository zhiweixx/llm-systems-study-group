# Week 6 — Robot learning: from demonstrations to VLAs and LLM agents

52 slides for researchers familiar with LLMs and statistics who are new to robotics. Slides 1–14 explain robot observations, concrete action arrays, demonstration records, and imitation learning. Slides 15–38 teach VLMs, VLAs, autoregressive and flow-based action generation, and π0.5 training and execution. Slides 39–52 explain Code as Policies, System 1 / System 2 architectures, and general models controlling or supervising robots. The original 30-slide presentation is preserved separately as a reference.

Read alongside MIT Robotic Manipulation, Stanford CS224R (2026) imitation learning, and the primary papers linked on each slide. Teaching diagrams are authored illustrations. Dataset frames and paper plots retain source attribution. Reported results are not new experiments run for this deck; evaluations with different tasks and interfaces are not directly comparable.

Play the embedded ALOHA demonstration and use Previous step / Next step within the teaching examples. Arrow keys navigate slides; Notes opens the explanation and references. The HTML embeds its figures and robot demonstration and works offline. Print / PDF shows the final interactive frame and a static video poster. Sources checked 25 October 2026.

## 1. From images and language to robot actions

**Section: Opening**

The goal is to understand the interfaces, not to memorize a list of methods. We will inspect an actual stored episode, decode the robot action, create supervised training pairs, and follow a learned prediction into execution. Later sections connect this interface to visual-language representations, generative action models, code-based control, and hierarchical systems. The cover is a frame from the simulated ALOHA insertion dataset, not a physical-robot rollout or a result from a model introduced in this lesson.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 2. The policy commands; the controller tracks

**Section: Robot data and actions**

This is the joint-position interface used in our running example. The policy takes observations and selects desired joint/gripper coordinates. A lower-level feedback controller tracks those targets through motor actuation. The robot and environment then produce the next observations. Proprioception is measurement of the robot itself, such as joint position or velocity; it does not include the entire scene. A controller uses measured errors, so the commanded target need not equal the next measured position. Some robotics literature uses policy and controller interchangeably, and some learned policies directly output torques; the distinction here describes this concrete position-control stack. The displayed command begins with actual episode-zero action values rounded for readability.

- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)
- [ALOHA · Action interface](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)

## 3. Watch one recorded demonstration

**Section: Recorded demonstration**

This is episode 0 of lerobot/aloha_sim_insertion_human, a recorded simulation. The original dataset has a nominal 50-Hz record rate. The local playback samples every fifth source frame, giving ten displayed frames per second and covering nominal times 0 through 9.9 seconds. The encoded container lasts about 10.24 seconds, so playback time is not a sensor timestamp. Pause while the arms approach, grasp, and align. Ask what must be inferred from the image and which robot measurements are also useful. The original task is two-arm peg insertion; this is not evidence that a language model controls this rollout.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)

## 4. One frame is more than an image

**Section: Robot data and actions**

These are the actual dataset fields and shapes. The image shown is episode 0 frame 0. RGB storage uses height 480, width 640, and three channels; a model loader may transpose it to channel-first tensors. Both observation.state and action contain fourteen float32 numbers, but one records measured configuration and the other a desired configuration. The joint and gripper ordering is unpacked on the next page. LeRobot organizes storage, timestamps, and access; it does not make all datasets share the same action meaning.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Action interface](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)

## 5. Decode the complete 14D ALOHA command

**Section: Robot data and actions**

The first seven coordinates are the six left-arm revolute-joint targets and the left normalized gripper opening. The next seven repeat that ordering for the right arm. sim_env.py documents these as absolute joint position commands, not joint increments, tool coordinates, or torques. constants.py supplies the named joint order. Revolute joints use radians in the simulation. Gripper normalization is an affine mapping relative to calibration endpoints, and it is not clipped by the shown normalization function: the actual first right-gripper target is approximately −0.03124. Thus do not label every coordinate in the array as radians or promise a strict [0,1] bound. The stored target is associated with nominal frame time, not verified wall-clock time of a human action.

- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Action interface](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)
- [ALOHA · Joint order](https://github.com/tonyzhaozh/act/blob/main/constants.py)

## 6. A different robot interface can use a 7D tool command

**Section: Robot data and actions**

This vector is a declared teaching example, not an ALOHA record or a measured DROID sample. The interface is [Δx, Δy, Δz, Δr_x, Δr_y, Δr_z, g]. Translation is in meters in the robot base frame. The three rotation coordinates form an incremental rotation vector expressed in that same frame; its direction is the rotation axis and its norm is the angle in radians. It is not an elementwise addition to Euler angles. Here only the z coordinate is nonzero, so the requested orientation increment is 0.05 radians about base z. The gripper convention is explicitly 1=open, 0=closed. Action dimension depends on the interface: a seven-joint arm with a gripper might instead receive eight joint/gripper targets. Other conventions use velocities, quaternions, absolute poses, or normalized commands.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)

## 7. How does a tool target become motor motion?

**Section: Robot data and actions**

The arithmetic shown is only the translation part of the previous declared interface. Position coordinates are all expressed in the base frame. For the rotation-vector convention, orientation is updated by composing the current rotation with the incremental rotation, on the side specified by the spatial/base-frame convention. We do not fabricate numerical joint targets without a robot model. In this illustrative stack, inverse kinematics searches for joint configurations satisfying the target tool pose, and motion planning considers a feasible path and limits. An endpoint IK solution alone is not a collision-free trajectory. The joint controller then tracks the requested joint targets with measurement feedback. Some robot interfaces combine these steps in Cartesian impedance or velocity control; this page presents one concrete route, not a required universal architecture.

- [MIT · Pick and place](https://manipulation.mit.edu/pick.html)
- [Modern Robotics · Control](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-1-control-system-overview/)

## 8. An episode is an aligned sequence of records

**Section: Robot data and actions**

The actual episode contains records 0 through 499. This page shows two real images, frame 0 and frame 100, with omitted intervening records indicated by ellipses. The first-row shoulder values are actual left-shoulder measurements and targets; frame 100 only shows verified field shapes, not invented values. The converter copies source qpos, images, and actions using the same index, then creates nominal timestamps as index/FPS. The simulation training loader begins its action target at that index. This establishes a dataset alignment convention, not a claim about exact hardware capture or command-issue timestamps. The episode lasts 500 command slots at 50 Hz; its final nominal sample is 9.98 seconds. Episode identity must be retained when records are concatenated. The converter marks the final record done; that flag alone is not a task-success label.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [Episode 0 · Stored rows](https://datasets-server.huggingface.co/rows?dataset=lerobot%2Faloha_sim_insertion_human&config=default&split=train&offset=0&length=2)
- [ALOHA · Converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)

## 9. Where do the action labels come from?

**Section: Robot data and actions**

ALOHA teleoperation maps human-moved leader-arm joint positions to follower-arm targets and records robot observations alongside those commands. The displayed photograph is a physical ALOHA setup from the ACT paper; it is not a photograph of the simulated episode. UMI records human manipulation through a portable camera-equipped gripper, reconstructs the gripper trajectory with visual-inertial SLAM, and measures gripper width. Its data and deployment interfaces align observations and future actions, and account for timing and coordinate conventions. The two systems do not produce interchangeable arrays without conversion. We use these images to show where supervision comes from; no learner-generated action labels or per-frame language annotations are implied.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)
- [ALOHA · Action interface](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)

## 10. Turn an episode into input–target training pairs

**Section: Robot data and actions**

The current observation is paired with the next H stored commands, beginning at the same index under this simulation alignment convention. Here H=4 is illustrative, and every action cell is a complete fourteen-dimensional command for both arms. The observation includes the image and measured joint/gripper configuration. Moving t generates more training examples from the same episode. Other tasks may use image history or language as additional inputs. The input does not include the future observations drawn for temporal context. At an episode boundary, either drop incomplete windows or use padding with a validity mask rather than spill into another episode. Evaluation splits should use whole episodes and suitable scene/task separation to avoid near-duplicate neighboring frames leaking across splits.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [ALOHA · Converter](https://github.com/huggingface/lerobot/blob/3c0a209/lerobot/common/datasets/push_dataset_to_hub/aloha_hdf5_format.py)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 11. Action dimension and chunk length are different axes

**Section: Robot data and actions**

A single action is a vector with d coordinates; an action chunk is an H-by-d matrix. With this robot interface, d=14. Choosing H=50 gives 700 scalar outputs, not a robot with 700 joints or degrees of freedom. At 50-Hz execution, fifty slots cover one second of playback; their nominal start times span 0 through 0.98 seconds relative to the first slot. This H is a teaching example and should not be read as a universal hyperparameter for ACT, diffusion policies, or any other method. Predicting a chunk does not require committing to all its rows, which motivates the prefix-execution example next.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)

## 12. Behavior cloning learns the recorded input–action mapping

**Section: Robot data and actions**

Behavior cloning uses demonstrations as supervised training data. The input is the observation at the start of a window; the target is the demonstrated action or action chunk. A simple regression policy could minimize an L1 or squared error between predicted and target actions; a probabilistic or generative policy uses an objective appropriate to its output model. Numerical normalization is usually performed per action coordinate and undone before execution. The essential point is the training/deployment distinction: the recorded target is available to compute training loss, but it is not an input the deployed model can consult. The trained weights produce commands from fresh observations, and the controller executes them. Original ACT does not take a language instruction; adding language later is an extension of this input interface.

- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)
- [ACT · §IV](https://arxiv.org/html/2304.13705v1)

## 13. Execute a short prefix, then predict again

**Section: Robot data and actions**

This interaction is a toy receding-horizon schedule using the verified fourteen-dimensional robot interface. The model predicts H=6 commands, then the runtime executes h=2 commands at 50 Hz, covering forty milliseconds of playback. Every cell is an entire fourteen-dimensional target, not a single scalar or a model token. Rows distinguish predictions made from different observations; columns share the same robot execution index. Crossed-out gray cells are old predictions that are discarded and replaced by a new row. Green means executed. A controller can still use rapid feedback while following each target. The diagram omits inference and communication latency; real deployments must schedule targets with those delays in mind. H and h here are pedagogical choices, not the original ACT execution algorithm or its temporal-ensembling rule.

- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [UMI · Data and timing](https://arxiv.org/html/2402.10329)

## 14. Add language while keeping an executable action interface

**Section: Bridge to vision and language**

The next section adds language and pretrained visual representations to the interface we have established. The instruction on this page is the dataset task metadata string. It is not a per-frame spoken annotation, and it was not an input to the original ACT policy. This diagram illustrates a language-conditioned extension: images, robot measurements, and a task instruction condition a policy whose output still must be decoded into the declared robot commands. A familiar visual-language model may produce textual descriptions or answers; producing useful physical actions requires an action interface and suitable training or control machinery. Later pages distinguish direct vision-language-action policies from systems that generate code or call existing skills.

- [ALOHA · Metadata](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human/blob/main/meta/info.json)
- [ACT · §IV](https://arxiv.org/html/2304.13705v1)
- [CS224R · Imitation](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 15. A VLM turns an image and a question into an answer

**Section: Learning from images and language**

Start with the interface, not the model taxonomy. The photograph is the mock blue kitchen from original π0.5 Figure 6. The question and answer are authored teaching examples grounded in the visible cabinet color; no VLM was run for this slide. I denotes the image, Q the question, and y the answer tokens on later pages. PaliGemma is one representative generative VLM. Image-text training can also supervise captions or serialized boxes. The next slide opens the box.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 16. Image patches become vectors the language model can read

**Section: Learning from images and language**

PaliGemma §3.1 and Figure 2: a SigLIP vision encoder produces a sequence of image features; a linear projection maps them to the Gemma embedding dimension. The decoder receives projected image vectors and text embeddings in one prefix. These visual vectors are continuous embeddings, not discrete words or quantized action IDs. Image and question tokens use full prefix attention; generated answer tokens use causal attention. At 224-pixel resolution PaliGemma has 256 visual tokens, but the diagram shows only three representative slots. The drawing is conceptual; it does not imply only three patches exist.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)

## 17. VLM training predicts the answer, with the prompt loss masked

**Section: Learning from images and language**

This is an illustrative supervised record, not a claim that this photograph is in PaliGemma training. The objective sums negative log probabilities only over target answer positions, conditioned on image I, question Q, and preceding target tokens. Teacher forcing makes all target positions available in parallel during training while causal attention prevents looking at future answer tokens. Token labels are readable word-level stand-ins for subword tokens. EOS is end-of-sequence. PaliGemma §3.1 and §5.2 explain prefix attention and suffix-only supervision. A loss mask differs from an attention mask: the prefix contributes context even though it is not the prediction target.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)

## 18. A sentence answers the question; it does not move the robot

**Section: Learning from images and language**

At inference the answer suffix is not supplied: the VLM generates it autoregressively from the image, question, and its own earlier outputs. A semantic answer alone does not specify a robot command in the action interface defined in the preceding chapter. An application could translate language through separately engineered tools or skills; a VLA instead learns an action output from robot demonstrations. This page establishes the missing target supervision before introducing VLA architecture.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 19. A VLA predicts an action in a defined robot interface

**Section: Learning robot actions**

The seven coordinates are [Δx, Δy, Δz, Δr_x, Δr_y, Δr_z, g]. Translation is expressed in the robot base frame in meters. The three rotation coordinates form an incremental rotation vector in radians, not additive Euler angles. The toy convention is g=1 open and g=0 closed. The example is [0.02,0,−0.01,0,0,0.05,1]; it illustrates an action interface and is not an inferred, calibrated, or demonstrated command for the pictured kitchen. The photograph shows a richer mobile platform; the seven-coordinate interface is an instructional single-arm example. Original OpenVLA predicts a seven-dimensional end-effector/gripper action; exact dataset rotation and gripper conventions must be respected rather than presumed universal.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 20. Train the VLM to predict the demonstrated action tokens

**Section: Learning robot actions**

OpenVLA §3.2 represents each action coordinate by one discrete token and applies next-token cross-entropy only to action-token positions. Here z_j denotes the token associated with coordinate j and ℓ is the instruction. The seven z labels stand for codebook entries; they are not literal strings of an actual tokenizer. The displayed numeric values are the same illustrative command as the previous page, rounded to readable precision. During dataset preparation a known demonstration action is encoded into target tokens. A dataset-specific decoder later maps these tokens to approximate coordinate values and restores units. The schematic does not pretend arbitrary numbers have universal token IDs. Original OpenVLA uses 256 bins per coordinate over the 1st–99th percentile interval, but the number-line construction is not needed to understand the learning objective.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 21. Decode the tokens, execute the command, then observe again

**Section: Learning robot actions**

This closes the supervised-learning interface around a physical robot. The model predicts tokens without receiving the demonstration action. Decoding restores the action representation, normalization, and units expected by the robot controller. Execution changes the scene, so the policy must be called with new observations. Original OpenVLA predicts one action per call; later pages deliberately move from that single-step example to chunk generation. The controller and action adapter are essential parts of deployment, even when the policy is trained end to end.

- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 22. VLM and VLA training differ in their target records

**Section: Learning robot actions**

The two rows are deliberately authored examples over the same scene. Web image-text data can supply visual and semantic supervision. Robot demonstrations supply actions synchronized with observations and task instructions. A pretrained VLM transfers useful representations, but those representations do not replace the need for suitable action labels and an action interface. This table describes a distinction in record types rather than an exclusive partition: systems such as RT-2 and π0.5 co-train on both robot and non-robot records. Original OpenVLA trains the robot stage on demonstrations with an action-only loss.

- [PaliGemma §3](https://arxiv.org/html/2407.07726v1#S3)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 23. An action chunk is a matrix of future commands

**Section: Generating an action chunk**

This three-by-seven matrix is a toy continuation of the single-command example, not a measured rollout. H is the number of future commands and d_a the action dimension. The storage convention throughout this chapter is H rows by d_a columns. The controller may execute only a prefix before the policy replans from new observations. A chunk describes physical command time; it does not specify how the neural network generates the matrix. That distinction motivates the next slide. Real π0/π0.5 chunks contain 50 commands; the original paper sometimes uses an inclusive time index with H=49, while this teaching deck consistently uses H to count commands.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 24. A Transformer can generate actions autoregressively or by flow

**Section: Generating an action chunk**

Architecture, output representation, generation objective, and action chunking are distinct design choices. An autoregressive decoder predicts discrete action tokens with a causal output mask. A flow model can also use Transformer layers; π0’s continuous action block uses bidirectional attention among the action slots and predicts a vector field. In this diagram the box denotes an architectural family, not a claim that the two choices have identical weights or masks. Context can include robot state for models that use it. Chunking works with either route. Both can model multiple valid action choices, unlike a single deterministic mean-action predictor under squared error.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [FAST §§V–VI](https://arxiv.org/html/2501.09747v1#S5)

## 25. Autoregressive chunk generation reuses next-token prediction

**Section: Generating an action chunk**

c denotes the fixed conditioning context: current observation and instruction, plus state when used by the model. An action tokenizer encodes the full demonstrated matrix A into m tokens. During inference the model predicts tokens in sequence and then decodes them to the chunk. During teacher-forced training, causal masking still permits all target positions to be evaluated in parallel. The 700-token count is an illustrative 50×14 scalar-binning scheme, not a required length for every autoregressive VLA. FAST exists precisely because encoding every adjacent scalar separately can create long, highly redundant sequences. This comparison concerns serial dependencies, not an inference-time speed ratio.

- [FAST §§V–VI](https://arxiv.org/html/2501.09747v1#S5)
- [OpenVLA §3](https://arxiv.org/html/2406.09246v3#S3)

## 26. FAST compresses temporal redundancy before token prediction

**Section: Generating an action chunk**

FAST §V and Figure 4 apply a discrete cosine transform independently along time for each normalized action coordinate, scale and round the coefficients, flatten them in frequency-first order, and use byte-pair encoding to pack frequent coefficient patterns. The transform itself is a change of basis; quantization introduces reconstruction error, while BPE compresses the quantized symbols losslessly. The schematic does not claim that this specific 3×2 matrix always becomes two tokens. Token count varies with the chunk, precision setting, and tokenizer. The VLA learns to predict the resulting tokens, and inverse decoding reconstructs an approximate action matrix. FAST §VI evaluates viable autoregressive policies; compressed token prediction is not intrinsically incapable of dexterous behavior.

- [FAST §§V–VI](https://arxiv.org/html/2501.09747v1#S5)

## 27. A noisy chunk is just a numeric mixture of a chunk and noise

**Section: Learning a continuous chunk generator**

All matrices are explicit toy values in normalized action space, with rows indexing future physical commands and columns indexing two displayed coordinates. Noise ε has the same shape as A; real training samples its entries from a standard Gaussian. The shown noise values are illustrative. For τ=0.5, Xτ=(A+ε)/2 gives exactly the displayed matrix. τ is generation time from zero to one; it is not physical robot time. No robot should execute Xτ during the generative procedure. A is available because this is construction of a training example. This removes the need to understand denoising terminology before seeing the arithmetic.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 28. Flow training supervises a direction at the noisy input

**Section: Learning a continuous chunk generator**

For the straight interpolation path Xτ=(1−τ)ε+τA, its derivative is A−ε. Subtracting the matrices on the previous page gives the exact target displayed here. The action expert predicts a vector of the same shape as the chunk. The squared loss compares predicted and known directions, averaged over demonstrations, noise, and sampled τ. No generated rollout or environment reward is needed to construct this supervised target. Statistically, the optimal regressor is the conditional expectation of velocity given noisy input and context, not a direct conditional mean of final actions. We use the noise-to-action sign in corrected π0 v4. The original π0.5 text has an inconsistent sign convention; this teaching derivation is internally consistent.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 29. Flow inference starts from noise and repeatedly updates the chunk

**Section: Learning a continuous chunk generator**

Only the current context, sampled initial noise, and learned vector field are available at deployment. The network is not given A or A−ε. Starting at τ=0, a numerical integrator repeatedly evaluates the field and updates the entire action matrix until τ reaches one. π0 v4 §IV reports forward Euler with δ=0.1 for ten steps and caches the observation-prefix attention keys and values. Different samplers and step counts are possible. Intermediate matrices are internal generation states; the final matrix is decoded to the physical action convention and sent to the controller. This diagram is a computation graph, not a new trained policy or robot experiment.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)

## 30. Generation time and robot time are different axes

**Section: Learning a continuous chunk generator**

Each row represents one full action chunk at a different generation-time value τ. Each column is a future robot command time. Advancing τ changes every entry of the candidate chunk; it does not move the robot through those candidate values. Once generation finishes, the controller executes commands along the physical-time axis, usually only a prefix before observing and replanning. The τ=0.5 row is schematic and does not claim that inference follows the straight path to a particular known demonstration. This distinction applies to iterative diffusion and flow chunk generators.

- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)
- [CS224R, lecture 2](https://cs224r.stanford.edu/slides/02_cs224r_imitation_2026.pdf)

## 31. Compare the training and inference procedures

**Section: Learning a continuous chunk generator**

These are alternative ways of representing a conditional distribution, not an expressive versus inexpressive distinction. An autoregressive categorical model can represent multimodal action sequences. A flow field can transform noise into a multimodal continuous distribution. The averaging limitation of direct squared-error action prediction does not apply to autoregressive likelihood training. FAST §VI-E reports comparable performance for FAST and flow on some evaluated tasks, with faster training for FAST in some settings and faster inference for the smaller flow expert in that implementation. Such results depend on tokenizer, backbone, model size, sampler, task, and hardware. Do not read a serial token count divided by a flow step count as a wall-clock speedup.

- [FAST §§V–VI](https://arxiv.org/html/2501.09747v1#S5)
- [π0 v4 §IV](https://arxiv.org/html/2410.24164v4#S4)

## 32. π0.5 combines household decisions with learned robot motion

**Section: π0.5 as a worked example**

The photographs are from original π0.5 Figure 2; they depict a broader kitchen-cleaning demonstration. The four-box trace is our illustrative decomposition of the concrete dishes-in-sink task, not a frame-by-frame annotation or a measured rollout. This objective requires both choosing a useful next subtask and physically carrying it out in a new arrangement. The coming pages explain the records, training stages, and runtime that produce these two capabilities. The focus is the original research system, not a claim that a public checkpoint alone reproduces its full deployment.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 33. Different records supervise actions, subtasks, and visual answers

**Section: π0.5 as a worked example**

These are illustrative input/target records matching the original paper’s data roles, not reproductions of particular training entries. π0.5 §IV-C combines mobile household robot demonstrations, non-mobile robots in diverse homes, laboratory robots and Open X-Embodiment data, manually labeled high-level subtasks and boxes, and web visual tasks. The target mobile household subset is about 400 hours in about 100 home environments; it is not the entire dataset. Robot observations include images and proprioceptive state. Subtask labels condition low-level action training; high-level training predicts those labels from the overall task and current observation. Web records include captions, visual answers, and localization targets. The full numeric data mixing ratios are not disclosed.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 34. π0.5 first learns all record types with next-token prediction

**Section: π0.5 as a worked example**

Original π0.5 §IV-C first trains a web-pretrained VLM autoregressively on discrete targets, including FAST-encoded action chunks. Textual records naturally supply text tokens; localization supplies serialized location tokens. The loss is target-token cross-entropy, not flow matching at this phase. This reuses the VLM learning setup rather than introducing a separate architecture for each supervision type. The paper reports 280k gradient steps. Data include home and laboratory robot actions, high-level labels, and web tasks. Control-mode markers distinguish joint versus end-effector pose targets; actions are normalized with dataset-specific quantiles and padded for differing embodiments. Those implementation details belong in the training records, not in an assumed universal action convention.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [FAST §§V–VI](https://arxiv.org/html/2501.09747v1#S5)

## 35. Post-training adds flow while keeping token supervision

**Section: π0.5 as a worked example**

Original π0.5 §IV-B/D jointly optimizes target-token cross-entropy, including FAST action tokens, and flow matching with weight 10 for 80k additional steps. The action expert starts from random weights; the model retains its text-prediction capability. The attention mask prevents discrete and continuous action representations from attending to each other, so the flow head cannot copy the FAST action target. Post-training uses successful, length-filtered mobile and other-home robot episodes, corresponding high-level labels, web records, and new verbal-instruction examples. In those examples an expert directs the learned low-level policy with subtask commands, supplying high-level choices. Laboratory cross-embodiment data do not remain in this phase. This is supervised post-training, not reinforcement learning.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 36. At runtime, predict a subtask and then generate its motion

**Section: π0.5 as a worked example**

Original π0.5 §IV-A/B separates p(s|o,ℓ), a high-level subtask distribution, from p(A|o,s), the low-level action-chunk distribution. The low-level factor uses inferred subtask s instead of the overall instruction ℓ. Both are represented by the same model, used with different inference roles. High-level inference uses autoregressive text decoding; low-level inference uses ten flow integration steps. The model predicts 50-command chunks for the full 18–19 dimensional mobile platform, not the seven-coordinate teaching interface. Low-level actions use joint/gripper and torso targets plus base velocity. The diagram omits scheduling rates: choosing a new semantic subtask need not occur once per individual control tick.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 37. The π0.5 architecture now has a role for every input

**Section: π0.5 as a worked example**

Original π0.5 §IV-A and Appendix E use PaliGemma for image-language processing and a separate 300M-parameter action expert. Proprioceptive state is discretized and supplied as text tokens in π0.5, unlike the continuous state projection used by original π0. Image features and task/state text form contextual tokens. The action expert receives the candidate noisy action chunk and generation-time embedding and attends to that context. Within the continuous action block, tokens use bidirectional attention. Text generation remains causal. The two output roles are invoked with the conditioning appropriate to high-level or low-level inference; the diagram is not claiming that one unchanged prompt simultaneously produces both outputs on every call. Training also retains a FAST discrete-action output, as the preceding page showed.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 38. The paper measures task progress in unseen homes

**Section: π0.5 as a worked example**

Original π0.5 §V-A and Figure 7 evaluate three held-out real homes, each with kitchen and bedroom tasks: three kitchens and three bedrooms, not six independent homes. Panel b reports average task progress over ten trials per task/environment. The plotted task families are items in drawer, dishes in sink, and laundry in basket; not every task is present in every home. Progress rubrics award partial completion, so these bars are not binary full-task success rates. Mock environments are also plotted. Quantitative episodes last approximately two to five minutes; longer published demonstrations do not establish an equivalent measured success rate. The figure supports transfer of learned household skills to these test environments. It does not establish unrestricted household autonomy or performance on arbitrary homes and tasks.

- [π0.5 paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 39. Code as Policies: write a program for the task

**Section: Programs, hierarchy, and robot execution**

Start by identifying the blue block and green bowl in the actual project photographs. These images are from Figure 2a; the instruction and program on this slide are authored teaching material, not a reproduced model output or an executed robot test. The toy uses vector addition and a fixed vertical gripper orientation. Its object anchors and 12-cm clearance are assumed suitable for this scene. The program chooses a sequence of perception and motion calls. Following slides define the external functions and show their execution. Do not describe this particular program as a general pick-and-place implementation: grasp geometry, calibration, motion planning and grasp-status checking belong to its provided interface.

- [Code as Policies](https://code-as-policies.github.io/)

## 40. Define the API before reading the generated program

**Section: Programs, hierarchy, and robot execution**

These names and return types define an authored toy interface. detect returns the identifier of one uniquely matching object; get_position returns a NumPy three-vector for its calibrated anchor in metres in the robot base frame. Therefore adding a three-element list in the previous program performs numerical vector addition, not Python list concatenation. grasped is a supplied status predicate. move_to tracks a Cartesian target at fixed gripper orientation and returns when its motion operation completes; this does not itself verify task success. The gripper call opens or closes the fingers. The LLM composes these functions rather than implementing the detector, calibration, kinematics, or servo here. The broader Code as Policies method can also generate control logic and helper functions.

- [CaP · methods + APIs](https://arxiv.org/html/2209.07753v4)

## 41. Run the program one operation at a time

**Section: Programs, hierarchy, and robot execution**

Advance through four authored snapshots. The first demonstrates a perception return value. The second requests the grasp position and closure. The third summarizes the lift-and-transfer part of the program; the full program includes a lift before lateral transfer. The fourth releases and receives a new image. The values are illustrative and use the toy base-frame convention, not the different world-frame convention of the later RoboDojo interface. The runtime, perception stack, planner, controller and physical world produce these intermediate values and effects. The diagram is a teaching animation rather than a physics simulation or execution of the displayed Python. A returned motion status and an observed task outcome are different checks.

- [Code as Policies](https://code-as-policies.github.io/)

## 42. A feedback loop does not require an LLM call at every step

**Section: Programs, hierarchy, and robot execution**

This authored bounded loop retries a grasp using newly queried positions and a supplied grasp-status check. It illustrates a runtime distinction: the LLM produces source code, and ordinary program execution repeats its body. A system may separately ask an LLM to revise code after failure, but that is an additional loop and is not required for each sensor query. The loop uses the previous slide’s toy API, including blocking moves and a fixed orientation. It omits recovery after exhausting attempts because the teaching point is where repeated feedback occurs. Code as Policies supports such perception-conditioned logic, but the exact program here is not a quoted experimental implementation.

- [CaP · methods + APIs](https://arxiv.org/html/2209.07753v4)

## 43. Change the model output, not the physical task

**Section: Programs, hierarchy, and robot execution**

The upper row uses original OpenVLA as a concrete numeric-action example: visual and language inputs are mapped to action tokens, decoded into a robot command, and executed by a controller. The lower row describes the authored program example. Its language model uses code knowledge and API examples to generate logic; its execution stack supplies physical capabilities. This does not mean a code-based robot needs no robot experience. Its detector, calibration, learned skills or controller may have been built or trained using embodied data. Conversely, a VLA may also be wrapped in tools or code. The rows distinguish the emitted object and the location of learned motor behavior, not two mutually exclusive species of robot.

- [OpenVLA · §3](https://arxiv.org/html/2406.09246v3#S3)
- [CaP · methods + APIs](https://arxiv.org/html/2209.07753v4)

## 44. A task contains decisions at different scales

**Section: Programs, hierarchy, and robot execution**

This authored clean-table example separates choosing a useful behavior from producing its motor commands. Putting a cup in a tray and wiping a spill require different motion patterns, and each behavior can involve many feedback updates. The diagram depicts possible subtasks, not an instruction to execute both branches simultaneously. Their order should depend on the actual scene and task. Hierarchical designs can reuse a motor behavior under several broader goals and let local feedback proceed between semantic updates. A hierarchy does not guarantee success: an incorrect subtask, a poor motion policy, or an unsuitable execution interface can each prevent completion.

- [π0.5 · original paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 45. Separate semantic context from local action generation

**Section: Programs, hierarchy, and robot execution**

Use these labels operationally. Some designs separate visual-language interpretation from a motor-action network, allowing different update schedules. Semantic context can be a latent feature representation, not an explicit verbal plan or chain of thought. The motor module still uses a robot controller beneath its action output. There is no universal model size, frequency, message format, or training objective implied by the names. System A/B labels are paper-specific and are not treated here as standard aliases. GR00T N1, for example, connects VLM token features to a flow-based action module and trains the modules together. A modular learned policy and a separate-model deployment are different design decisions.

- [GR00T N1 · §2.1](https://arxiv.org/html/2503.14734v2#S2.SS1)

## 46. A message can be a subtask or learned features

**Section: Programs, hierarchy, and robot execution**

The top row illustrates an explicit semantic-subtask interface, as used in the original π0.5 model factorization. The lower row illustrates feature conditioning; the colored bars are symbolic dimensions, not measured neural activations. GR00T N1 uses a sequence of VLM features with cross-attention, while original Helix uses one continuous latent vector. Those are related feature interfaces but not identical implementations. Neither learned features nor a System 2 label establish that a model produced a natural-language reasoning trace. An interface must be specified before deciding how it can be inspected, cached, updated, or reused.

- [π0.5 · original paper](https://www.physicalintelligence.company/download/pi05.pdf)
- [GR00T N1 · §2.1](https://arxiv.org/html/2503.14734v2#S2.SS1)

## 47. Helix sends latent intent to a fast visuomotor policy

**Section: Programs, hierarchy, and robot execution**

This is the original February 2025 Helix, not a later Helix version. Figure reports a 7B VLM feeding one continuous latent vector to an 80M visuomotor Transformer. The action module also sees fresh images and robot state. Both are trained jointly with regression; inference uses separate onboard processes and a shared latent. The reported 7–9 Hz and 200 Hz rates describe this implementation, not universal definitions. Approximately 500 hours of teleoperation receive hindsight language labels. The outputs are upper-body targets, not direct evidence of torque-level LLM control. No explicit textual reasoning trace is required by this architecture.

- [Helix · 20 Feb 2025](https://www.figure.ai/news/helix)

## 48. π0.5 uses an explicit semantic subtask

**Section: Programs, hierarchy, and robot execution**

Map the original π0.5 factorization onto the preceding interface distinction: p(A,c|o,ℓ)=p(c|o,ℓ)p(A|o,c). The low-level factor conditions on the inferred subtask and observation rather than directly on the overall task prompt. The semantic and continuous outputs belong to the paper’s unified model with different expert weights. This drawing does not assert a particular GPU allocation, number of OS processes, or asynchronous schedule. High-level labels supervise semantic decisions, while action demonstrations ground motor generation. The outer model boundary distinguishes internal computation roles from a system assembled by independently deploying two unrelated pretrained models.

- [π0.5 · original paper](https://www.physicalintelligence.company/download/pi05.pdf)

## 49. Keep the intent; refresh the local feedback

**Section: Programs, hierarchy, and robot execution**

This is an authored logical schedule, not a measured timing trace, fixed frequency ratio, or description of every hierarchical robot. The semantic message stays unchanged while the motor policy repeatedly receives fresh observations. Each displayed action box summarizes a behavior stage for readability; a real stage usually contains many numerical updates. The distinction is between refreshing sensor-conditioned local control and recomputing semantic context. A system could implement this with separate asynchronous processes or another scheduler. A reused message can also become stale, so the design needs an update rule appropriate to its task. These generic possibilities are not a claim that the original π0.5 deployment uses the exact Helix schedule.


## 50. A tool call can request a numeric robot target

**Section: Programs, hierarchy, and robot execution**

The call uses the move_eef schema from An Unexpected Robot Policy, Appendix A. The target value and note are authored illustrations. The published interface uses absolute world-frame grasp-point coordinates, permits subsets of dimensions, and retains observed values for unspecified dimensions. Non-learned software validates a request and attempts joint-path planning; accepted paths execute before a new observation returns. IK means inverse kinematics. A target request is different from generating a reusable Python program and from selecting a learned policy proposal. The model-facing motion tool does not supply semantic pick/place primitives. This is a simulation interface example, not a reported successful motion.

- [Astra direct interface · App. A](https://arxiv.org/html/2609.24170v1)

## 51. A general model can supervise a learned policy

**Section: Programs, hierarchy, and robot execution**

This follows the RoboDojo hybrid in Systematically Exploring. A released task-finetuned π0.5 policy proposes 50 joint-action steps. Astra either accepts a prefix of 1–15 steps or replaces it with 1–5 end-effector-target steps. The branches are mutually exclusive. The highlighted n=3, joint-coordinate values and two z targets are authored examples, not values extracted from a trial. Joint commands and Cartesian corrections use different action representations and their associated executors. The general model supplies review and correction while the robot policy supplies learned motion proposals. Frozen general-model weights do not remove the policy’s robot-specific training. No benchmark ranking or new robot test is asserted.

- [Astra hybrid · RoboDojo](https://arxiv.org/html/2609.38537v1)

## 52. Which component actually supplies robot actions?

**Section: Programs, hierarchy, and robot execution**

Use this table to answer four concrete questions, rather than to rank model families. A general VLM output alone is not a calibrated executable robot command. A VLA learns an action mapping from robot supervision and still needs a controller. A generated program relies on the meaning and implementation of its APIs, which may contain learned robot skills. A dual-system architecture divides semantic context and motor generation; it may itself be a VLA and can also expose tools. The row labels overlap. The robot-grounding column identifies where physical action competence must be supplied, not a universal claim that each component was trained from scratch or that programs cannot contain learned control.

- [OpenVLA · §3](https://arxiv.org/html/2406.09246v3#S3)
- [CaP · methods + APIs](https://arxiv.org/html/2209.07753v4)
- [GR00T N1 · §2.1](https://arxiv.org/html/2503.14734v2#S2.SS1)
