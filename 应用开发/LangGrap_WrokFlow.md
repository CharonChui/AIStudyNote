

LangGraph的能力并不在于通过Node、Edge、State构建简单的Agent，而是用图的方式灵活编排任意复杂的工作流。   

## 1. 顺序执行(Prompt Chaining)

Prompt Chaining是最基础的编排模式：前一个LLM的输出作为后一个LLM的输入，像流水线一样串联起来。   

典型场景：写作流程(大纲 -> 初稿 -> 润色)、翻译 + 校对、分布推理。

与直接LLM调用的区别：顺序链可以在步骤之间加入验证门控(Gate) -- 如果某步质量不达标就退回重做，而不是一路到底。 


示意图：

![[Pasted image 20260909132314.png]]

  

代码示例：
```Python
# Prompt Chaining: 生成笑话 → 检查质量 → 润色 → 加梗
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

class JokeState(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str

def generate_joke(state: JokeState):
    """步骤1: 生成初稿"""
    msg = llm.invoke(f"Write a short joke about {state['topic']}")
    return {"joke": msg.content, "final_joke": msg.content}

def check_punchline(state: JokeState):
    """门控函数：检查笑话是否有笑点"""
    msg = llm.invoke(f"Check if the joke has punchline and wordplay, return 'Pass' or 'Fail' for the joke: ```{state['joke']}```")

    result = msg.content
    if "Pass" == result:
        print(f"  [检查] 通过 ✓")
        return "Pass"
    print(f"  [检查] 不通过 ✗")
    return "Fail"

def improve_joke(state: JokeState):
    """步骤2: 改进——增加双关语"""
    print(f"  [润色] 添加双关语...")
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: JokeState):
    """步骤3: 润色——加一个反转"""
    print(f"  [润色] 添加反转...")
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}

# 构建图
chaining_graph = (
    StateGraph(JokeState)
    .add_node("generate_joke", generate_joke)
    .add_node("improve_joke", improve_joke)
    .add_node("polish_joke", polish_joke)
    .add_edge(START, "generate_joke")
    # 条件边：通过则直接结束，不通过则进入改进流程
    .add_conditional_edges(
        "generate_joke",
        check_punchline,
        {"Pass": END, "Fail": "improve_joke"}
    )
    .add_edge("improve_joke", "polish_joke")
    .add_edge("polish_joke", END)
    .compile()
)

print("=== 测试1: 话题「大树」 ===\n")
r = chaining_graph.invoke({"topic": "大树", "joke": "", "improved_joke": "", "final_joke": ""})
print(f"\n最终笑话:\n{r['final_joke']}")
```




## 2. 并行执行(Parallelization)

通过并行化，可以让多个LLM同时处理一个任务，有两种不同的实现方式和作用：

- 提高效率: 将任务拆分为子任务，让多个LLM同时运行独立的子任务来完成，再汇总结果    
- 提高准确度: 多个LLM运行相同的任务产生不同的输出，再合并优化结果


![[Pasted image 20260909132418.png]]


比如，我需要同时基于某个话题写三种题材：

- 笑话
    
- 故事
    
- 诗歌
    

并行执行的话效率会更高，示例代码：


```Python
# Graph state
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str

# Nodes
def joke_node(state: State):
    """First LLM call to generate initial joke"""

    msg = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": msg.content}

def story_node(state: State):
    """Second LLM call to generate story"""

    msg = llm.invoke(f"Write a short story about {state['topic']}")
    return {"story": msg.content}

def poem_node(state: State):
    """Third LLM call to generate poem"""

    msg = llm.invoke(f"Write a short poem about {state['topic']}")
    return {"poem": msg.content}


def aggregator(state: State):
    """Combine the joke, story and poem into a single output"""

    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


# Build workflow
parallel_builder = StateGraph(State)

# Add nodes
parallel_builder.add_node("joke", joke_node)
parallel_builder.add_node("story", story_node)
parallel_builder.add_node("poem", poem_node)
parallel_builder.add_node("aggregator", aggregator)

# Add edges to connect nodes
parallel_builder.add_edge(START, "joke")
parallel_builder.add_edge(START, "story")
parallel_builder.add_edge(START, "poem")
parallel_builder.add_edge("joke", "aggregator")
parallel_builder.add_edge("story", "aggregator")
parallel_builder.add_edge("poem", "aggregator")
parallel_builder.add_edge("aggregator", END)
parallel_workflow = parallel_builder.compile()

# Show workflow
display_graph(parallel_workflow)
```


## 3. 路由(Routing)

Routing工作流处理用户输入，然后将他们引导到上下文相关的任务Node。   

这允许您为复杂的任务定义专门的流。例如:   
- 智能电商客服，首先识别问题的类型，然后将请求路由到售前咨询、退款、退货等对应的Node。
- 知识问答机器人，首先识别问题知识领域，然后将请求路由到不同领域对应的专业Node上。 


所以路由本质上也是条件分支，让Graph根据问题动态选择下一个节点。   

这是Agent智能决策的核心机制。  

![[Pasted image 20260909132810.png]]


例如，写三个工作节点：
- 天气节点：可以查询天气
- 新闻节点：可以查询新闻
- 翻译节点：可以翻译
然后写一个意图识别节点，根据用户意图路由到对应的工作节点，执行对应任务。
```Python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage, SystemMessage

class Route(BaseModel): 
    step: Literal["weather", "translate", "chat"] = Field(
        Node, description="The next stemp in the routing process"
    )

# 把路由结构化输出绑定到模型，形成一个用于路由的llm
router = llm.with_structured_output(Route)    

class IntentState(TypedDict):
    query: str
    intent: str
    result: str
    
def classify_itent(state: IntentState): 
    """分类用户意图"""
    decision = router.invoke(
        [
            SystemMessage(
                content = "Route the input to weather, translate, or chat based on the user's request"
            ),
            HumanMessage(content=state['query']),
        ]
    )
    return {"intent":decision.step}
    
def handle_weather(state: IntentState):
    return {"result" : f"天气{state['query']} 晴 25度"}    
    
def handle_translate(state: IntentState):
    return {"result": f"翻译: {state['query']} -> Hello World"}    
    
def handle_chat(state: IntentState): 
    return {"result": f"闲聊: {state['query']} -> 你好呀！"} 

def intent_router(state: IntentState) -> Literal["weather", "translate", "chat"]:
     """根据意图路由到不同处理器"""
     return state["intent"]    
     
routing_graph = (  
    StateGraph(IntentState)  
    .add_node("classify", classify_intent)  
    .add_node("weather", handle_weather)  
    .add_node("translate", handle_translate)  
    .add_node("chat", handle_chat)  
    .add_edge(START, "classify")  
    .add_conditional_edges("classify", intent_router, {  
        "weather": "weather",  
        "translate": "translate",  
        "chat": "chat"  
    })  
    .add_edge("weather", END)  
    .add_edge("translate", END)  
    .add_edge("chat", END)  
    .compile()  
)  
  
display_graph(routing_graph)     
```


## 4. Orchestrator-Worker + Send API


**Orchestrator-Worker**，顾名思义，**编排器—工作者**模型。分为三个步骤：
- **编排器（Orchestrator）**：将任务分解为子任务，将子任务分配给worker
- **工作节点（Worker）**：并行执行任务
- **合成器（Synthesizer）**：将worker的输出汇总为最终结果

这听起来似乎与之前讲的并行模式很像，没错这确实也属于并行模式。但不同之处在于：
- 普通并行工作流：任务是需要手动拆分的，且子任务数量是固定的
- Orchestrator-Worker：是用模型**动态的将复杂任务拆分为多个并行子任务**，子任务数量不确定
Orchestrator-Worker模式提供了更高的灵活性，通常用于无法像并行化那样预先定义子任务的场景。例如：AI编程助手、深度研究助手

  

我们以深度研究助手（DeepResearcher）为例，它可以根据你指定的话题写出专业报告：

- 首先，需要根据话题生成报告大纲，细分出多个章节，定好每个章节主题
- 然后，把每个章节作为一个子任务，交给多个工作进程并行执行
- 最后，汇总所有子任务生成的章节，得到完整的研究报告
  

  

但是，这里有一个很严重的问题：

> 子任务的数量是不确定的，因此Node没办法提前定义好，那该如何设计Graph？


不用担心，LangGraph 对此提供了内置支持。通过 **Send** API，你可以动态创建工作节点并向它们发送特定的输入。每个工作节点都有自己的State，而且所有工作节点的输出都会被写入一个共享的State 字段中，编排器图可以访问该字段。这使得编排器能够获取所有工作节点的输出，并将其整合成最终输出。


LangGraph 构建 Agent 有5个核心阶段：
- 将任务拆解为一个个离散的节点，Node
- 明确每个节点的具体任务
- 设计节点中流转的数据格式，State
- 编写节点函数，做好异常处理
- 连接节点，形成图，Graph

传统 Prompt Chaining 是线性的 A→B→C，但真实世界的 Agent 是**有分支、有回路、有人工介入**的图结构。

















