
简单来说，LangGraph允许你自己定义Agent的**工作流**（WorkFlow）中的每一个**节点**（Node）：

- 需要稳定性的节点，就用传统编程控制
- 需要LLM自主控制的，就交给LLM
  
并且，整个工作流不再是一条直线运行的**链（Chain）**，而是一个像真正工作流那样有分支、有循环的**图（Graph）。**


在LangGraph中，工作流由3个基本的要素组成：
- **节点（Node）**：即工作流中的关键工作代码，可以是工具调用、知识检索、LLM调用，用函数来定义。他们接收当前状态作为输入，执行一些计算或副作用，并返回更新后的状态。
- **边（Edge）**：根据当前状态确定下一个要执行的Node的函数。它们可以是条件分支或固定转换。定义节点之间的转换逻辑，决定执行流程。也可以理解为线，就是把各个节点连接起来的路径，是控制工作流走向的关键，连接方式有： 
	- 串行
	- 并行
	- 条件
- **状态（State）**：整个工作流中流转的数据。一种共享数据结构，用于表示应用程序的当前快照。它可以是任何数据类型，但通常使用共享状态模式来定义。在整个图执行过程中共享和传递的数据。

由Node和Edge组成的工作流被称为图(Graph)。   



![[Pasted image 20260908215408.png]]

为了提高应用的运行的稳定程度，LangGraph为节点运行提供了Checkpointer功能，应用运行到每个**节点（Node）**都会形成**检查点（Checkpoint）**，我们可以方便的跳转到任意checkpoint，这样一来应用就具备了三大功能：

- **故障恢复**：即便是程序异常中止，也可以随时恢复到失败节点继续执行。
- **人机交互**：在程序运行中可以暂停工作流，在关键节点允许人工介入，得到人工确认后恢复执行，进一步提高程序的可靠性。
- **持久记忆**：由于Checkpointer可以持久化存储，能方便的拿到历史信息，因此应用还具备了记忆功能。


自从LangChain的1.0版本以后，底层的Chain模式已经逐渐废弃，而是彻底的投入了Graph的怀抱。

LangChain 1.x之后的版本中，Agent的底层实现已经全部替换为基于LangGraph了！

  
不过，两者的定位完全不同：

- **LangChain**：便捷开发AI应用的框架，提供了开发AI应用的各种统一API
    - 统一的LLM调用
    - 统一的Embedding Model调用
    - 统一的Vector Store调用
    - ...
- **LangGraph**: 工作流编排框架，提供了完善的工作流编排工具
    - 基于Node和Edge的工作流编排
    - 基于State的工作流数据传输
    - 基于Checkpointer的数据持久化

总结一下：

- **LangGraph**关注的是工作流的编排，你的工作流甚至可以完全与LLM无关，与Agent无关。如果你利用LangGraph开发Agent，可以自由的控制Agent工作流中的每一个细节
    - 优点：自由度更高。
    - 缺点：比较繁琐
    - 场景：需要完全控制Agent工作流程和细节，或者复杂Agent工作流的情况
- **LangChain**则是提供的统一API，底层基于LangGraph实现Agent，简化了Agent的开发。
    - 优点：简化Agent开发
    - 确定：灵活度不如LangGraph
    - 场景：简单的LLM调用，或者快捷的Agent开发



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



## 节点(Node)

节点(Node)是图(Graph)的执行单元，每个Node是一个Python函数。 

```Python
def my_node(state: State) -> dict: 
    return {"some_key" : new_value}
```


- 输入： 
	- state: 完整的State对象
	- config: 一个RunnableConfig对象，包含诸如thread_id之类的配置信息以及诸如tags之类的跟踪信息
	- runtime: 一个Runtime对象，包含运行时context以及其他信息，如store和stream_writer
- 输出：一个字典，只包含要更新的字段(部分更新)
- 可以做什么：调用LLM、执行工具、读数据库、文件操作，最终把结果更新到State中

需要注意的是，LangGraph中有两个默认的Node是无需定义的，可以直接使用:    
- START: 开始节点，也是入口
- END: 结束节点，也是出口

定义好node函数后，使用add_node方法将这些节点添加到图中，如果在想图中添加节点时未指定名称，系统会为其分配一个与函数名相同的默认名称。  



## 状态(State)

State是图的共享内存，贯穿整个执行过程。 定义图时，首先要做的就是定义图的State。
- State定义Graph中的数据字段
- 每个Node都可以获取State数据、更新State数据(返回要更新的字段值即可)


State由图的schema以及reducer函数组成，其中reducer函数指定了如何对状态进行更新。 
Node返回数据后State的更新处理方式取决于Reducers，而且State中的每个字段都有自己的Reducer。  



State的schema将作为图中所有Nodes和Edges的输入模式，它可以是 TypedDict 或 Pydantic 模型。



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
- 自定义 Reducer：支持用户自定义合并逻辑，自定义需要用到Annotated来定义Annotated[type, reducer]

```Python
from langgraph.graph.message import add_messages
# add_messages Reducer（消息列表专用）
class AddMessagesState(TypedDict):
    messages: Annotated[List, add_messages]


# . operator.add Reducer（列表追加）
class ListAddState(TypedDict):
data: Annotated[List[int], operator.add]
```




## 3. 边(Edge)

边定义了逻辑的路由方式以及图如何决定停止。这是智能体工作方式以及不同节点之间
通信方式的重要组成部分。边有几种关键类型：
- Normal Edges: 普通边。直接从一个节点连接到下一个节点。
- Conditional Edges: 条件边。调用函数以确定接下来要前往哪个（哪些）节点。
- Entry Point: 入口点。用户输入到达时首先调用哪个节点。
- Conditional Entry Point: 条件入口点。调用一个函数来确定当用户输入到达时，首
先调用哪个（些）节点。
一个节点可以有多个出边。如果一个节点有多个出边，那么所有这些目标节点都将作为
下一个超级步骤的一部分并行执行。


#### 3.1 Normal Edge

普通边连接的两个节点流向是固定的，前一个节点执行完一定会执行后一个节点。   

不同之处在于节点之间是串行还是并行:   
- 串行：整个工作流只有一条路径，路径中的节点按照顺序一次执行
- 并行：整个工作流由多条路径，不同路径可以同时执行

#### 3.2 Conditional Edge

条件边的添加方式如下:  
```Python
add_conditional_edges(node_a, router, mapping)
```
接收三个参数:   
- node_a：是当前节点
- router: 路由函数，逻辑自定义，它的返回值默认就是下个节点的名字
- mapping: 如果router返回值与下个节点名不一样，可以用mapping定义返回值与下个节点名字的映射关系

router可以根据情况返回不同的结果，但一次只能有一个结果。 也就是说Conditional Edge连接的多个节点有且只有1个会成为next_node。 


#### 3.3 Command实现条件分支

如果不想编写Conditional Edge，也可以在Node中直接返回下个节点信息。  

由于Node必须返回对State的更新，而没有条件边就需要返回下个节点的名字。函数返回值只能有一个。所以LangGraph就提供了Command API，用Command来指定下个节点以及要更新的字段。  


```Python
# 在Node中基于Command实现条件分支

class RouteState(TypedDict):
    score: int
    result: str

def scorer(state: RouteState) -> Command[Literal["fail", "pass"]]:
    score = int(input("请输入分数:"))

    next_node = "pass" if score >= 60 else "fail"

    return Command(
        update=RouteState(score = score), # 通过update更新State字段
        goto=next_node                    # 通过goto指定下个Node
    )

def pass_node(state: RouteState):
    return {"result": "通过！"}

def fail_node(state: RouteState):
    return {"result": "不通过"}

command_condition_graph = (
    StateGraph(RouteState)          # 1.创建Graph
                                    # 2.添加节点
    .add_node("scorer", scorer)         # 打分节点
    .add_node("pass", pass_node)        # 失败节点
    .add_node("fail", fail_node)        # 成功节点
                                    # 3.添加Edge
    .add_edge(START, "scorer")               # start -> scorer
    .add_edge("pass", END)                   # pass -> end
    .add_edge("fail", END)                   # fail -> end
    .compile()
)

result = command_condition_graph.invoke({"score": 0, "result": ""})
print(f"score={result['score']} -> {result['result']}")
```

其中:    
- update: 用来指定要更新的state字段
- goto: 用来指定下个节点




### Rumtime概念

在LangGraph中同样支持Runtime Context功能，方式也完全一样:   
- 定义ContextSchema
- 创建Graph时指定state、context_schema、store等Runtime


```Python
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
  
  
----

# LangGraph构建Agent


```Python

llm = init_chat_model("deepseek-chat")

class SimpleAgentState(TypedDict): 
    user_input: str
    result: str
    
def call_llm(state: SimpleAgentState): 
    response = llm.invoke(state['user_input'])    
    return {"result" : response.content}
    
llm_graph = (
    StateGraph(SimpleAgentState)
    .add_node("llm", call_llm)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)    
result = llm_graph.invoke(SimpleAgentState(user_input="你好", result = ""))
print(result)
```
  
  上面的Graph中，State比较简单，只记录了两个值:    
  - user_input: 一次用户输入
  - result: 一次LLM结果

然后实际在复杂的Agent中，用户会多次与LLM交互，还有工具调用，产生大量的历史消息，这些消息都需要记录下来。

Agent交互时的消息类型有很多： SystemMessage、AIMessage、HumanMessage、ToolMessage等。

而这些消息有一个共同的父类： BaseMessage。  

因此，可以在State中定义一个字段，类型为list[BaseMessage]。由于要不断添加新的消息到这个list，所以还需要一个Reducer，而LangGraph正好提供了这样的Reducer: add_message。 

```Python
from langgraph.graph.message import add_messages
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, ToolMessage
)

class MessageAgentState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]
```
  
  
  add_messages这个Reducer的效果就是不断累积消息，形成Message列表。 

完整代码:   
```Python
from langgraph.graph.message import add_messages

class MessageAgentState(TypedDict): 
    messages: Annotated[list[BaseMessage], add_messages]
    
llm = init_chat_model("deepseek-chat")    

def call_llm(state: MessageAgentState):
    response = llm.invoke(state['messages'])
    # 更新state
    return {"messages" : [response]}

llm_graph = (
    StateGraph(MessageAgentState)
    .add_node("llm", call_llm)
    .add_edge(START, "llm")
    .add_edge("llm", END)
    .compile()
)
result = llm_graph.invoke({"messages" : messages})
print(result)

```
  
  
  这里就能看到和之前LangChain中create_agent的结果是一样的，这是因为LangChain的create_agent内部就是基于LangGraph实现的。  

#### MessageState

由于记录Agent消息的场景非常常见，所以LangGraph中提供了一个默认的State，专门用来记录消息历史，叫做MessagesState，内部实现是:   

```Python
class MessagesState(TypedDict): 
    messages: Annotated[list[AnyMessage], add_messages]
```
  可以看到内部实现与上面我们自定义的非常类似，只是消息类型改成了AnyMessage。    
  所以，如果只是为了记录消息历史，就不用自定义State了，直接使用MessageState就可以了。 
  
  
  
### 会话记忆


有了历史并不等于有记忆，记忆必须基于会话id(thread_id)来分别管理会话历史。     

这就要使用LangGraph中的Checkpointer来实现了。 


### 节点缓存

LangGraph还提供了节点级别的缓存功能。当请求进入节点时，会根据节点输入的参数生成key，以key和节点的输出建立缓存。如果下次key命中，则直接返回结果而不用再执行该节点的逻辑。

要使用节点缓存需要两步：
- 在编译Graph时指定缓存实现方式，LangGraph提供了`InMemoryCache`、`RedisCache`、`SqliteCache`
- 为节点指定缓存策略。每个缓存策略支持：
    - `Key_func`：函数，用于根据节点的输入生成缓存键，默认为带有pickle的输入参数的hash值。
    - `ttl`：缓存的有效时间，以秒为单位。如果没有指定，缓存将永远不会过期。

#### 失败容错

当一个节点发生故障时，比如：外部API超时、临时网络异常、其他未处理的异常时，LangGraph提供了三种可组合的失败容错机制： 

- 重试(Retries)：根据异常类型和回退设置自动重新运行失败的尝试
- 超时(Timeouts)： 限制单个尝试可能运行的时间
- 错误处理(Error Handling)：在所有重试用尽后的降级处理策略
可以使用set_node_defaults为所有Node统一配置这些机制，也可以每次调用add_node时单独配置。


```Python
llm = init_chat_model("gpt-3.5-trubo")
async def llm_node(state: dict, runtime):
    print(f"调用llm ... 第 {rumtime.execution_info.node_attempt}次")
    response = await llm.ainvoke(state['messages])
    return {'messages' : [response]}
    
async def default_error_handler(state: MessageState, error: NodeError) -> Command:

    print(f"运行异常，{error}")    
    print("执行fallback逻辑")
    return Command(goto=End)
    
builder = StateGraph(dict)
builder.add_node(
    "llm",
    llm_node,
    timeout = TimeoutPolicy(idle_timeout = 3),
    retry_policy = RetryPolicy(max_attempts = 3),
    error_handler = default_error_handler
)    
builder.add_edge(START, "llm")
builder.add_edge("llm", END)
fail_tolerance_graph = builder.compile()

result = await fail_tolerance-graph.ainvoke({"message" : [HumanMessage("你好")]})
print(result)
```



#### Streaming

LangGraph中的streaming方式与LangChain中的Agent一样。 

```Python
for chunk, metadata in agent_graph.stream(
    {"message" : [{"role":"user", "content":"429和517的平方根是多少"}]},
    stream_mode="messages"
): 
    content = chunk.content
    if content: 
        print(content, end="", flush=True)

```

在LangGraph>1.2.0版本后提供了Stream的3.0版本，推荐使用stream_events方式实现streaming:   
- 使用graph.stream_events方法实现stream调用
- 需要在参数中指定version="v3"
与LangChain中的v2版本不同，调用stream_events不会返回Iterator，而是返回一个GraphRunStream类，你可以自己选择想要通过stream获取的内容：

- **stream** : Iterate every protocol event.
    
- **stream.messages** : Stream chat model messages and token deltas.
    
- **stream.values** : Iterate state snapshots and await the final value.
    
- **stream.output** : Await the final output.
    
- **stream.subgraphs** : Discover and observe nested graph executions.
    
- **stream.interrupts** : Inspect human-in-the-loop interrupt payloads.
    
- **stream.interrupted** : Check whether the run paused for human input.
    
- **stream.extensions** : Consume custom stream transformer projections.


```Python
# 先拿到stream
stream = agent_graph.stream_events({
    "messages": [{"role": "user", "content": "429和517的平方根是多少?"}],
}, version="v3")

# 再通过stream.messages拿到一个messages片段的Iterator
for message in stream.messages:
    for token in message.text:
        print(token, end="", flush=True)
```



### 中断(Interrupts)

中断(Interrupts)允许您在特定Node暂停Graph执行，并在继续之前等待外部输入。   

利用这个特性，就能实现Human In The Loop模式，把一些重要决断交给人工来做。   

当中断被触发时，LangGraph使用它的Checkpointer保存Graph的State，并无限期地等待，直到恢复执行。  

你可以在Graph的任何Node调用interrupt()函数来中断工作。该函数接受向调用者现实的任何json可序列化的值。  

当您准备好继续时，您可以通过使用Command(resume={})重新调用Graph来恢复执行，而resume值将成为节点内部interrupt()调用的返回值。  


一定要注意：断点恢复必须指定thread_id，它是找到不同会话Checkpointer的关键。 






















  
  
  
  
  
  
  


