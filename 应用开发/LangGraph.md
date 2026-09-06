

LangGraph是编排LLM工作流的图框架：节点(Node)是执行单元，边(Edge)是流转方向，状态(State)是节点间传递的数据。  

它的定位是“底层、可控”，比LangChain的Chain更灵活，适合做多步推理、人机协作、循环重试的Agent流程。  

LangGraph 是一个低级编排框架和运行时环境，用于构建、管理和部署长期运行的有状
态智能体（agents）。核心理念是将 Agent 工作流建模为图（Graph），其中：
 节点（Nodes）：对智能体逻辑进行编码的函数。他们接收当前状态作为输入，执行一些计算或副作用，并返回更新后的状态。代表计算单元，可以是 LLM 调用、工具执行或任何自定义逻辑
 边（Edges）：根据当前状态确定下一个要执行的Node的函数。它们可以是条件分支或固定转换。定义节点之间的转换逻辑，决定执行流程
 状态（State）：一种共享数据结构，用于表示应用程序的当前快照。它可以是任何数据类型，但通常使用共享状态模式来定义。在整个图执行过程中共享和传递的数据。


LangGraph 提供了构建生产级智能体应用的核心能力：
 持久化执行：构建能够从故障中恢复并长时间运行的智能体
 人机协作：在任何时刻检查和修改智能体状态
 记忆管理：支持短期工作记忆和跨会话的长期记忆
 流式处理：专为流式工作流设计
 生产级部署：为有状态、长期运行的工作流提供可扩展的基础设施
LangGraph 可独立使用，也可与 LangChain、LangSmith 无缝集成。
LangGraph 与 LangChain 的高级抽象不同，它提供了更细粒度的控制，让开发者能够精
确控制智能体的执行流程，适合需要定制化编排的复杂应用场景。


#### 为什么需要LangGraph

随着应用复杂度的提升，传统的 Agent 框架往往面临着状态管理混乱、执行流程不可
控、错误恢复困难等挑战。大语言模型的使用不仅仅是作为执行工具，而更多作为推理引擎
的需求在日益增长。这种转变带来的是更多的重复（循环）和复杂条件的交互需求，这就导
致基于 LCEL 的线性序列构建方式在构建更复杂、更智能的系统时显示出了明显的局限性。
LangGraph 作为 LangChain 团队推出的新一代 Agent 框架，通过引入图计算模型和状
态机理念，为构建生产级 AI Agent 提供了全新的解决方案。


#### LangGraph 架构设计

LangGraph 的运行时基于 Google 的 Pregel 算法，这是一种用于大规模并行图计算的
模型。执行过程分为三个阶段：

- Plan（规划）：确定本轮要执行的节点
- Execution（执行）：并行执行所有选中的节点
- Update（更新）：将节点输出更新到通道（channels）

每个执行轮次称为一个”超步（super-step）”，系统会持续迭代直到没有节点需要执行。  

### State

定义图时，首先要做的就是定义图的State。State由图的schema以及reducer函数组成，其中reducer函数指定了如何对状态进行更新。 

。State的schema将作为图中所有Nodes和Edges的输入模式，它可以是 TypedDict 或 Pydantic 模型。所有 Nodes 都会发出对 State 的更新，
然后使用指定的 reducer 函数应用这些更新。



LangGraph中state_schema、input_scehma、output_schema，这三个概念用于管理图状态的不同方面：   

- state_schema: 这是图的完整内部状态，包含了所有节点可能读写的字段，必须指定，不能为空。 
它里面又包含input_schema和output_schema.通过这样来达到一些状态的隔离机制。 

- input_schema: 定义图接受什么输入，是state_schema的的子集或相等，如果不指定默认就等于state_schema。 
- output_schema: 定义图返回什么输出，是state_schema的子集或相等，如果不指定默认就等于state_schema。 


#### Reducer

reducer 是理解节点更新如何应用于 State 的关键，State 中的每个键都有其独立的 reducer
函数。每个 node 的返回值中的每个 key 与全局 state_schema 中对应的 key 进行合并更新，
具体更新逻辑取决于每个 key 指定的 reducer 函数。
Reducer 常用函数有以下几种：    
- 默认行为：未指定 Reducer 时使用覆盖更新
- add_messages：用于消息列表追加
- operator.add：用于列表追加或数值累加
-  operator.mul：用于数值相乘
- 自定义 Reducer：支持用户自定义合并逻辑

```
from langgraph.graph.message import add_messages
# add_messages Reducer（消息列表专用）
class AddMessagesState(TypedDict):
    messages: Annotated[List, add_messages]


# . operator.add Reducer（列表追加）
class ListAddState(TypedDict):
data: Annotated[List[int], operator.add]
```


### Nodes


在LangGraph中，节点是Python函数(可以是同步的，也可以是异步的)，它们接受以下参数:   

- state: 图的状态
- config: 一个RunnableConfig对象，包含诸如thread_id之类的配置信息以及诸如tags之类的跟踪信息
- runtime: 一个Runtime对象，包含运行时context以及其他信息，如store和stream_writer

定义好node函数后，使用add_node方法将这些节点添加到图中，如果在想图中添加节点时未指定名称，系统会为其分配一个与函数名相同的默认名称。  

LangGraph 支持基于节点输入对任务/节点进行缓存。使用缓存的方法如下：
编译图（或指定入口点）时指定缓存。
为节点指定缓存策略。每个缓存策略支持：
 key_func 用于根据节点的输入生成缓存键，默认情况下是使用 pickle 对输入进行
hash 运算的结果。
 ttl，即缓存的生存时间（以秒为单位）。如果未指定，缓存将永不过期。

在很多使用场景中，你可能希望节点拥有自定义的重试策略，例如在调用 API、查询数
据库或调用大语言模型（LLM）等情况下。
为节点添加重试策略，需要在 add_node 中设置 retry_policy 参数。retry_policy 参数接受
一个 RetryPolicy 命名元组对象。默认情况下，retry_on 参数使用 default_retry_on 函数

### Edges

边定义了逻辑的路由方式以及图如何决定停止。这是智能体工作方式以及不同节点之间
通信方式的重要组成部分。边有几种关键类型：
 Normal Edges: 普通边。直接从一个节点连接到下一个节点。
 Conditional Edges: 条件边。调用函数以确定接下来要前往哪个（哪些）节点。
 Entry Point: 入口点。用户输入到达时首先调用哪个节点。
 Conditional Entry Point: 条件入口点。调用一个函数来确定当用户输入到达时，首
先调用哪个（些）节点。
一个节点可以有多个出边。如果一个节点有多个出边，那么所有这些目标节点都将作为
下一个超级步骤的一部分并行执行。

在传统的图结构中，节点和边都是预先定义好的，但在某些场景下，我们需要动态地根
据运行时状态来决定执行哪些节点。Map-Reduce 模式就是这样一个典型场景：
 一个节点生成一个动态数量的对象列表。
 另一个节点需要对列表中的每个对象进行处理。
 最终将所有处理结果合并。
为了支持这种设计模式，LangGraph 支持从条件边返回 Send 对象。Send 接受两个参
数：
尚硅谷大模型技术之 LangGraph
更多 Java –大数据 –前端 –python 人工智能资料下载，可百度访问：尚硅谷官网
 第一个是节点的名称。
 第二个是要传递给该节点的状态。


### Rumtime概念


```python3
async def extract_keywords(state: DataAgentState, runtime: Runtime[DataAgentContext]):
```

 Runtime[DataAgentContext] 是 LangGraph 自动注入到节点函数中的第二个参数（第一个是
  state）。你不需要自己创建它，框架在调用节点时会把一个运行时对象传进来。
  
  #### 它解决什么问题？ State vs Context
  
  LangGraph里有两类数据:  
  
  
    | | state（第一个参数） | runtime.context（第二个参数） |
  |---|---|---|
  | 生命周期 | 在整个图内流转，节点返回的字段会合并、传递、被下游节点覆盖 | 单次 run 的静态配置，所有节点共享同一份，不会被节点修改 |
  | 适合放什么 | 中间结果、错误信息、候选列表等"流程数据" | user_id、db 连接池、query、API key、日志器等"运行依赖" |
  | 来源 | 节点返回值 + invoke 初始输入 | `graph.ainvoke(input, context={...})` 传入 |
  
  
  
  例如： 在DataAgentState里有个error: str，这是节点间传递的数据流程数据。  
  
 而DataAgentContext里面放的是每次请求不变、但不同请求不同的东西。  例如用户提问的问题、用户id、数据库连接池等。 
 
 #### Runtime提供的属性  
 
- runtime.context        # 本次 run 的静态上下文（上面说的 DataAgentContext）
- runtime.store          # 持久化存储（BaseStore），做记忆/长期存储用，如 InMemoryStore、PostgresStore
- runtime.stream_writer  # 向自定义流写入数据的函数，配合 graph.stream 做流式输出
- runtime.heartbeat      # 长耗时任务的心跳，防止节点被判定为 idle 而超时
- runtime.previous       # 函数式 API + checkpointer 下，上一个返回值
- runtime.execution_info # 当前节点运行的只读执行元数据
- runtime.server_info    # LangGraph Server 注入的元数据（本地开源运行时为 None）
- runtime.control        # 协作式取消/排空控制
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  


