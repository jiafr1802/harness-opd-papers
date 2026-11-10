# SWE-bench 相关经典工作的训练环境与 Action Space

## 1. 结论

SWE-bench 本身主要是 benchmark 与 Docker evaluation harness，不规定训练时必须使用哪一种 agent scaffold。围绕它形成的训练路线至少分成四类：

1. **SWE-agent / SWE-smith：** Docker repository environment + Bash + structured editor + submit；
2. **SWE-Gym / OpenHands：** Docker environment + CodeActAgent event/tool protocol；
3. **R2E-Gym：** Gym-style Docker environment + 显式 function-call tools，并兼容多种 scaffold；
4. **SWE-RL：** 不做交互式环境 rollout，只生成结构化 patch 并用 oracle-patch similarity 给 reward。

若要为 OPRD 补一个公开、可复现的训练 coding environment，最直接的两个选择是：

- **首选 R2E-Gym**：action schema 最清楚，原生支持在线 trajectory collection 和 executable reward；
- **首选工程复现 SWE-smith + SWE-agent**：Docker、任务、成功轨迹、SFT 流程和三工具 interface 都公开，生态更成熟。

## 2. 对照表

| 工作 | Repository sandbox | Agent scaffold | Action / Tool space | Verifier | 是否用于训练 rollout |
| --- | --- | --- | --- | --- | --- |
| SWE-agent | Docker / SWE-ReX | SWE-agent ACI | 当前框架支持 Bash + tool bundles；SWE-smith 配置使用 `bash`、`str_replace_editor`、`submit` | SWE-bench Docker test harness | 原工作偏 agent/evaluation；现已支持大规模轨迹采集 |
| SWE-Gym | 每实例 Docker image | pinned OpenHands `CodeActAgent`；另有 MoatlessTools | OpenHands action/event protocol，主要是 shell、文件浏览/编辑和 finish | repository unit tests + SWE-bench-style evaluation | 是；成功轨迹用于 policy SFT，也训练 verifier |
| R2E-Gym | `DockerRuntime`，也支持 Kubernetes | `r2egym` / `openhands` / `sweagent` | R2E：`search`、`file_editor`、`execute_bash`、`finish`；兼容模式：`str_replace_editor`、`execute_bash`、`submit` | `/run_tests.sh`、expected test output、SWE-bench 或 F2P/P2P | 是；原生并行收集训练 trajectories |
| SWE-smith | 每 repo/commit Docker environment，可供多个任务复用 | SWE-agent | `bash`、`str_replace_editor`、`submit` | 原始测试、bug patch 后测试、F2P/P2P | 是；公开 52k+ tasks 与 26k trajectories |
| SWE-RL | 无交互式 Docker rollout | 训练时无完整 agent loop；推理用 Agentless Mini | `<solution>` 内的 SEARCH/REPLACE patch text | 格式 reward + predicted/oracle patch similarity | 是，但属于 text patch RL，不是 environment OPD |

## 3. SWE-agent：经典 ACI 路线

论文：[SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)  
代码：[SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent)

SWE-agent 的核心贡献是 Agent-Computer Interface。环境负责：

- 在隔离 runtime 中 checkout/reset repository；
- 将模型 action 解析为 command 或 structured tool call；
- 执行命令并把 stdout/stderr 返回为 observation；
- 在 `submit` 后提取 patch，再交给 SWE-bench harness 做最终测试。

当前实现通过 SWE-ReX 管理 runtime，默认支持 Docker deployment。`ToolConfig` 明确提供内建 Bash，并允许通过 command bundles 增加编辑、查看、提交等工具；也支持 function-calling parser。工具层还会屏蔽 `vim`、`nano`、交互式 shell 等不适合自动 rollout 的命令。

需要注意版本差异：原始 SWE-agent 论文强调 `open`、search、edit 等 ACI 命令；当前主分支已经演化为 Bash + bundles + function calling 的可配置框架。引用 action space 时应绑定具体配置，而不是只写“SWE-agent tools”。

## 4. SWE-Gym：OpenHands CodeAct 训练环境

论文：[Training Software Engineering Agents and Verifiers with SWE-Gym](https://arxiv.org/abs/2412.21139)  
代码：[SWE-Gym/SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)  
环境 fork：[SWE-Gym/OpenHands](https://github.com/SWE-Gym/OpenHands)

SWE-Gym 提供 2,438 个真实 Python 软件任务，每题包含 repository、可执行 runtime、unit tests 和自然语言任务。主实验使用固定版本的 OpenHands `CodeActAgent`：

```text
task
  -> OpenHands CodeActAgent
  -> shell / browse / edit actions
  -> Docker or RemoteRuntime
  -> observation event stream
  -> repository tests
```

官方复现流程明确分成 trajectory rollout、executable evaluation、policy training 和 verifier training。成功轨迹转换为 SFT 数据。这里的 action space 属于 pinned OpenHands fork 的 event/tool protocol；直接使用最新 OpenHands 主分支可能得到不同结果。

## 5. R2E-Gym：最明确的公开 Function-call Action Space

论文：[R2E-Gym: Procedural Environment Generation and Hybrid Verifiers for Scaling Open-Weights SWE Agents](https://arxiv.org/abs/2504.07164)  
代码：[R2E-Gym/R2E-Gym](https://github.com/R2E-Gym/R2E-Gym)

`RepoEnv` 是 Gym-style environment，底层 `DockerRuntime` 在通常为 `/testbed` 的 repository path 中执行操作。Agent 代码直接公开三套兼容 scaffold：

```text
r2egym:
  search
  file_editor
  execute_bash
  finish

openhands / sweagent compatibility:
  str_replace_editor
  execute_bash
  submit
```

支持 API-native function calling，也支持文本形式：

```text
<function=tool_name>...</function>
```

Environment 将 action 解析成 `Action(function_name, parameters)`，执行后返回 observation、reward、done 和 info。Verifier 可运行 `/run_tests.sh`，并按环境类型读取 expected test output、SWE-bench resolution status 或 SWE-smith 的 FAIL_TO_PASS / PASS_TO_PASS 测试。

这使 R2E-Gym 特别适合复现 OPRD：Teacher 可以在完全相同的 function-call token prefix 上重新评分，而只有 Student rollout branch 真正执行 action。

## 6. SWE-smith：最完整的公开训练数据生产线

论文：[SWE-smith: Scaling Data for Software Engineering Agents](https://arxiv.org/abs/2504.21798)  
代码：[SWE-bench/SWE-smith](https://github.com/SWE-bench/SWE-smith)

SWE-smith 将 GitHub repository 构造成可复用 Docker environment，再自动制造会破坏至少一个现有测试的任务。其 expert trajectories 使用 SWE-agent，公开的主要接口是：

- `bash`：执行 command；
- `str_replace_editor`：查看、创建和编辑文件；
- `submit`：结束任务并提交 patch。

Bash 使有效 action space 仍然开放，可以运行 `grep`、`find`、`python`、`pytest`、build command 等。验证时记录 FAIL_TO_PASS 与 PASS_TO_PASS，用于区分修复目标和 regression。

SWE-smith 的优势是完整公开 tasks、Docker assets、trajectories 和 SFT 流程；缺点是生成任务的测试在训练时可见，适合作为训练基础设施，不应直接替代带隐藏测试的最终 benchmark。

## 7. SWE-RL：一个重要反例

论文：[SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](https://arxiv.org/abs/2502.18449)  
代码：[facebookresearch/swe-rl](https://github.com/facebookresearch/swe-rl)

SWE-RL 使用 issue、代码上下文和 oracle patch 训练模型输出 SEARCH/REPLACE patch。训练 reward 主要来自格式合法性和 predicted patch 与 oracle patch 的 similarity，并不在每次 RL rollout 中启动 Docker、执行 shell 或运行 unit tests。

因此 SWE-RL 虽然是 SWE-bench 相关训练工作，却不能作为 OPRD 所需的 interactive coding environment。它适合作为“无环境执行的 patch-policy baseline”。

## 8. 对 OPRD 复现的建议

一个公开 OPRD prototype 可以采用：

```text
Task pool: SWE-smith 或 R2E-Gym
Sandbox: DockerRuntime
Action schema: search + file_editor + execute_bash + finish
Rollout: reminder-free Student
Outcome: executable tests / F2P + P2P
Teacher: same-prefix logits + Global/Turn-local Reminder
Training: success NLL + failure Top-100 reverse KL
```

相比直接依赖 Claude Code，这套组合的 tool schema、executor、trajectory format 和 verifier 都能审计，也更容易严格证明 Teacher 与 Student 使用同一个 action space。

## 9. 主要来源

- [SWE-agent repository](https://github.com/SWE-agent/SWE-agent)
- [SWE-bench evaluation harness](https://github.com/SWE-bench/SWE-bench)
- [SWE-Gym repository](https://github.com/SWE-Gym/SWE-Gym)
- [SWE-Gym OpenHands reproduction guide](https://github.com/SWE-Gym/SWE-Gym/blob/main/docs/OpenHands.md)
- [R2E-Gym repository](https://github.com/R2E-Gym/R2E-Gym)
- [SWE-smith repository](https://github.com/SWE-bench/SWE-smith)
- [SWE-RL repository](https://github.com/facebookresearch/swe-rl)