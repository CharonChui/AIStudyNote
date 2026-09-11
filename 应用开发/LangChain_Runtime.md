
LangChain的create_agent底层运行在LangGraph Runtime之上。  

Runtime可以理解为Agent每次执行时拿到的运行环境，其中与数据管理最相关的三个部分是：   
- Context: 本次调用需要的静态身份和依赖
- State: 执行过程中不断变化的短期状态
- Store： 跨会话保存的长期数据

## Runtime


Runtime包含的核心数据

|对象|作用|读写|典型生命周期|典型内容|访问方式|
|---|---|---|---|---|---|
|Context|注入本次运行所需的身份、配置和依赖|只读|单次调用|user_id、tenant_id、权限、数据库连接|`runtime.context`|
|State|保存 Agent 执行中产生的动态数据|可读写|同一会话：同一thread_id的多次调用|消息、步骤结果、计数器、业务流程状态|`runtime.state`|
|Store|保存跨会话需要复用的数据|可读写|跨会话；|用户偏好、长期记忆、历史资料|`runtime.store`|

判断数据放在哪里，可以连续问两个问题：

1. 这份数据在执行过程中会不会变化？不变化，优先考虑 Context。
    
2. 如果会变化，下一次会话还要不要使用？只在当前会话使用放 State，跨会话复用放 Store。



## 1. Runtime Context

Runtime Context是一种依赖注入机制： 调用者在运行开始时把用户身份、租户信息、权限或外部服务传给Agent，节点、中间件和工具在执行时再从Runtime读取。 

它解决的是三个工程问题： 
- 避免把用户ID、数据库链接等信息硬编码进工具
- 同一个Agent可以安全的服务不同用户和租户
- 工具的业务参数只保留真正需要模型填写的内容

Context中的数据默认不会自动发给大模型，只有代码主动读取并把内容加入提示词或工具调用结果时，模型才会看到它。  

例如，在系统中用户通常会先登录，然后访问agent，我们就可以把登录用户信息存入Context，在Agent内部的Middleware或Tool旧能很方便的通过`runtime.context`获取用户信息。而这个用户信息并不会让模型看到，也不需要作为参数让模型传递，纯粹是Agent内部信息。

#### 定义Context Schema

首先，需要约定Context的数据结构，这里使用@dataclass来定义，可以省去定义__init__等魔法函数。 

```Python
from dataclass import dataclass

@dataclass(frozen=True)
class UserContext: 
    """Agent运行时的上下文"""
    user_id: str
    tenant_id: str
```
定义UserContext类就是一个context_schema，作用是给Context提供明确的数据结构。  
frozen=True不是LangChain的硬性要求，但能表达运行内部可修改的设计意图。 



#### 在Tool中使用Context
在tool中，可以添加runtime参数，然后利用runtime.context来访问context
```Python
from langchain.agents import create_agent

@tool
def get_current_user_profile(runtime: ToolRuntime[UserContext]):
    """查询当前登录用户的资料"""
    ctx = runtime.context
    
    if ctx.tenant_id is None or ctx.user_id is None: 
        return "当前用户未登录，无法查看"
	
	profile = USER_DATABASE[ctx.tenant_id][ctx.user_id]        
	return (f"姓名: {profile['name]}")
```



#### 给Agent添加Context
在定义Agent时，需要指定常见好的Context
```Python
agent = create_agent(
    model="deepseek-v4-falsh",
    tools=[get_current_user_profile],
    context_schema=UserContext, # 指定Context类型
    system_prompt="xxx"
)
```
调用agent时，可以传递Context信息:  

```Python
response = agent.invoke(
    {"messages": [HumanMessage("Hello")]},
    context=UserContext(user_id="u1", tenant_id="t1")
)
for message in response['messages']:
    message.pretty_print()

```




## 2. State(短期记忆)

State是Agent的短期记忆，存储当前会话的历史消息、任务状态等信息。 

之前使用的是默认的AgentState，其中只包含会话的历史消息(messages)。 

#### 2. 1 自定义State

自定义AgentState，其实就是定义一个类，然后继承AgentState，在其中添加想要记录的属性信息即可。   

例如，想要实现统计用户的模型调用次数、调用时间等功能，可以这样定义:  
```Python
from langchain.agents import AgentState
from typing import NotRequired

class CustomState(AgentState):
    """Agent的任务状态"""
    mode_call_count: NotRequired[int] # 模型调用次数

```

由于在AgentState中已经具备messages属性，也就是历史消息列表，因此CustomState继承了AgentState以后，不仅可以记录会话的历史消息，也能记录Agent运行的任务状态信息了。 


接下来就是如何操作state中的自定义属性了，在LangChain中通常有两个地方可以操作state:   

- Tool
- Middleware

在定义tool的时候，LangChain内置了一个runtime参数，通过runtime可以获取Agent的内部信息，包括:   
- state: dict结构
- store
- context
访问： runtime中state本质是一个dict，可以这样访问state的属性:   
```python3
mode_call_count = runtime.state.get("model_call_count", 0)
```
修改： 修改state是通过返回一个update格式的Comman指令:   
```Python
@tool
def my_tool(runtime: ToolRuntime): 
    return Command(update= {
        "model_call_count": 1,
        "messages": [ToolMessage("success!", tool_call_id=runtime.tool_call_id)]
    })
```


自定义了AgentState还不够，还需要在创建Agent的时候设定state schema，告诉Agent要使用自定义的state:   
```Python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from datetime import datetime

@tool
def update_state(runtime: ToolRuntime):
    """A tool that update agent state"""
    messages = runtime.state['messages']
    message_count = len(messsages)
    # 组织结果
    command = {
        "model_call_count": runtime.state.get("model_call_count", 0) + 1,
        "messages": [ToolMessage("successfully updated agent state", tool_call_id=runtime.tool_call_id)]
    }
    
    if message_count <= 2:
        command['session_start'] = datetime.now()
        
    return Command(update=command)    


agent = create_agent(
    "deepseek_chat",
    tools=[update_state],
    state_schema=CustomState,
    checkpointer=InMemorySaver(),
    system_prompt="xxx"
)

config = {"configurable" : {"thread_id" : "1"}}
response = agent.invoke(
    {"messages": [HumanMessage(content="hello")]},
    config
)
for message in response['messages']:
    message.pretty_print()

```


## 3.Store(长期记忆)


store是LangChain提供的长期记忆机制，用于在不同会话间共享数据。     

例如：模型以外的数据、用户偏好等。  

LangC提供了多种Store的实现方式，例如:   
- InMemoryStore
- PostgresStore
- RedisStore
- ...

#### 3. 1 Store的数据结构

Store的数据格式是JSON文档，JSON文档采用分级管理:    
- Namespace(命名空间)： 可以理解为一个文件夹
	- Key: 可以理解为文件名，必须唯一
	- Value: 要存储的JSON文档

首先，要初始化Store:   
```Python
from langgraph.store.memory import InMemoryStore

memory_store = InMemoryStore()

memory_store.put(
    ("preferences", ), # namespace是一个tuple
    "user_001",        # key,可以是任意类型
    {                  # value，是JSON格式文档
        "style": "business",
        "langguage": "zh-CN"
    }
)

memory_store_put(("preferences",), 
    "user_002",
    {
        "style": "trump",
        "langguage": "en-US"
    }
)
```

然后是读取Store数据，LangChain提供了两种方式来查询store中的数据:     

- get: 在指定namespace下根据key查找
- search: 在指定namespace下对value做语义搜索或过滤
```Python
user_preferences = memory_store.get(("preferences",), "user_001")
print(f"用户信息: {user_preferences.value if user_preferences else 'Not found'}")

search_results = memory_store.search(
    ("preferences",), 
    filter={"langguage": "zh-CN"},   # 基于字段做过滤查询
    limit=5
)
print(f"搜索结果数量: {len(search_results)}")
```


#### 3.2 基于向量模型的Store

LangChain中的store支持基于向量相似度的语义检索。  



![](https://my.feishu.cn/space/api/box/stream/download/asynccode/?code=YjcyZTUzOGRmM2QxZTViMmE5ZjU1OTVkM2E0NTBiZjhfSnBySzJkMXFXVDlTeWY5MThVZ3gxQkluRmVoanBsWHBfVG9rZW46TmEzRWJHYk5lb3ZqbEt4Q0szamM5REU0bkNjXzE3ODg4NTg5MDk6MTc4ODg2MjUwOV9WNA&add_watermark=true&scene_type=CCM)

  

```Python
from langgraph_cli.schemas import IndexConfig
from langchain_community.embeddings import DashScopeEmbeddings
import os

# 初始化向量模型
embedding_model = DashScopeEmbeddings(
    model="text-embedding-v4", dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 初始化store
memory_store = InMemoryStore(index=IndexConfig(
    embed=embedding_model,  # 向量模型
    dims=1024  # 向量维度
))


# ==================== 3.读取 ====================

# 3.1.基于get查询
user_data = memory_store.get(("users",), "user_001")
print(f"用户信息: {user_data.value if user_data else 'Not found'}")

# 3.2.基于search搜索数据
search_results = memory_store.search(
    ("users",),
    query="001",  # 基于字段语义检索
    limit=5
)
print(f"搜索结果数量: {len(search_results)}")
print(search_results)

```




#### 3.3 在Tool中访问store

与state类似，LangChain中访问store通常也可以有两种场景:   
- Tool
- Middleware

在Tool中访问store也是通过runtime. 

```Python
from langchain.tools import tool, ToolRuntime

@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """获取用户信息"""
    if runtime.store is None:
        return "Store not available"

    # 通过runtime获取store，读取其中的数据
    user_info = runtime.store.get(("users",), user_id)

    if user_info is None:
        return "没有找到用户"

    return f"用户信息: {user_info.value}"
```



#### 3.4 给Agent添加store
创建agent时需要指定store:  

```Python
from langchain.messages import HumanMessage

# 在Agent中集成
agent = create_agent(
    model="deepseek-chat",
    tools=[get_user_info],
    store=memory_store  # 指定store的存储方式
)

response = agent.invoke({
    "messages": [HumanMessage("帮我查询user_001的信息")]
})
for message in response['messages']:
    message.pretty_print()
```



## 总结

1. State(短期记忆)
	- 通过runtime.state参数访问
	- 用于保存会话历史消息、对话状态、计数等临时信息
	- 生命周期是当前会话
	- 存储方式： 可以是InMemorySaver，也可以是数据库
2. Store(长期记忆)
	- 通过runtime.store访问
	- 用于扩展知识、用户偏好等信息
	- 生命周期跨越多个会话
	- 支持InMemoryStore和数据库存储
3. Context(上下文)
	- 通过runtime.context访问
	- 用于传递用户ID、配置参数等
	- 生命周期是当前会话
	- 存储方式：基于内存


























