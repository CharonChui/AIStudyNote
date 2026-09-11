
中间件(Middleware)是一种控制Agent内部运行过程的技术，它在智能体运行的各个过程中预留钩子(hook)，方便迁入自定义操作。 


基于Middleware可以实现各种高级功能，比如：

- **拦截和修改请求** - 在模型调用前后对输入输出进行处理
- **实现PII脱敏** - 自动检测和脱敏敏感个人信息
- **会话摘要管理** - 当对话过长时自动压缩历史消息
- **人工审核机制** - 在执行危险操作前等待人工确认
- **动态模型选择** - 根据运行时条件选择不同的模型
- **自定义状态管理** - 扩展Agent状态以跟踪额外信息


LangChain提供了多种预置中间件，可以直接使用。例如:  

- PIIMiddleware
- ModelFallbackMiddleware
- HumanInTheLoopMiddleware


### PIIMiddleware - 个人信息脱敏

PIIMiddleware是一种预定义的wrap_model_call中间件，可以在调用模型前后自动检测并脱敏输入、输出消息中的个人身份信息(PII)，如邮箱、电话号码、身份证号等。

其中PII脱敏处理策略有四种:   
- block: 抛出异常
- `'redact'` - 用 [REDACTED_{PII_TYPE}] 来替代
- `'mask'` - 关键信息采用**掩码 (例如., `****-****-****-1234`)
- `'hash'` - 用哈希值来替换


### 2. ModelFallbackMiddleware

ModelFallbackMiddleware的作用是在模型调用失败时给出降级处理方案。可以在创建时设置多个模型，如果主模型调用失败，会自动调用备用模型。  


### HumanInTheLoopMiddleware - 人工审核

HumanInTheLoop简称为HITL，其作用是让人工介入到Agent执行流程中，在执行工具调用前暂停，等待人工确认。 
通常用于Agent执行敏感操作前的确认，例如:   
- 发送邮件
- 转账
- 执行脚本
- 读写文件
- ...

j假设有一个负责转账操作的智能体，
```Python
@tool

def transfer_money(amount: int, to: str): 
    """转账"""
    return f"已向账号{to}转账{amout}元"
```

然后，定义Middleware，设置需要人工确认的tool，以及人工确认的可选操作:  
```Python
from langchain.agents.middleware import HumanInTheLoopMiddleware

human_in_loop_middleware = HumanInTheLoopMiddleware(
    interrupt_on = {  
        "transfer_money" : {
            "description" : "请确认转账操作",
            "allowed_decisions" : ["approve", "reject", "edit"]
        }
    }
)

agent_with_hitl = create_agent(
   model="deepseek-chat",
   tools=[transfer_money],
   middleware=[human_in_loop_middleware],
   checkpointer=InMemorySaver(),
   system_prompt="xxx"
)

config = {"configurable" : {
    "thread_id": "3"
}}
response = agent_with_hitl.invoke(
	{"messages":[HumanMessage("帮我转5毛给xx")]},
	config=config
)
for message in response['message']:
    message.pretty_print()
```

执行后response里面就会返回调用了interrupt。 
'__interrupt__': [Interrupt(value={'action_requests': [{'name': 'transfer_money', 'args': {'amount': 2000, 'to': '6123008415124395223'}, 'description': '请确认转账操作'}], 'review_configs': [{'action_name': 'transfer_money', 'allowed_decisions': ['approve', 'reject', 'edit']}]}


这时候就可以根据提示让用户进行确认的选择，但是用户选择后，应该怎么把用户的选择告诉Agent呢？ 

如果用户拒绝Agent调用工具，我们需要再次调用Agent，并通过Command来告知Agent用户的选择是什么
```Python
from langgraph.types import Command

response = agent_with_hitl.invoke(
    Command(
        resume = {"dicisions" : [{"type" : "approve"}]}
    ),
    config=config. # 通过相同的thread_id来恢复之前暂停的会话
)
print(response)



response = agent_with_hitl.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    # 用户reject的原因告知给ai
                    "message": "用户取消转账。"
                }
            ]
        }
    ),
    config=config
)

print(response)



response = agent_with_hitl.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    # Edited action with tool name and args
                    "edited_action": {
                        # Tool name to call.
                        # Will usually be the same as the original action.
                        "name": "transfer_money",
                        # Arguments to pass to the tool.
                        "args": {"amount": 1000, "to": "王小明"},
                    }
                }
            ]
        }
    ),
    config=config  # Same thread ID to resume the paused conversation
)

pprint(response)

```



- interrupt_on：就是需要人工确认的tool信息，可以设置多个
- transfer_money：就是需要人工确认的tool名字
- description：是人工确认时的提示信息
- allowed_decisions：是人工确认时的可选操作，包括三种：
    - approve：允许执行
    - reject：拒绝执行
    - edit：修改tool参数后执行


## 自定义中间件

根据hook的种类，中间件可以分为两类：

- Node-style hooks：在具体某个节点执行的中间件，包含：
    
    - before_agent
        
    - before_model
        
    - after_model
        
    - after_agent
        
- Wrap-style hooks：环绕model或tool调用的中间件，包括：
    
    - wrap_model_call
        
    - wrap_tool_call
        

为了便于开发中间件，LangChain为每一种hook都提供了装饰器，我们只有定义函数并使用装饰器标记即可快速开发中间件。


做法是定义函数，并用`@after_model`装饰器来装饰该函数，但要注意，**函数的参数和返回值必须严格按照下面的示例**：
```Python
from langgraph.runtime import Runtime
from langchain.agents import AgentState
from typing import NotRequired, Any
from langchain.agents.middleware import after_model

class CustomAgentState(AgentState): 
    """扩展Agent状态，添加自定义字段"""
    model_call_count: NotRequired[int] # 模型调用次数
    
@after_model(state_schema=CustomAgentState)    
def increment_counter(state: CustomAgentState, runtime: Runtime) -> dict[str, Any]: 
    """使用装饰器的after_model钩子 - 增加调用计数"""
    
    current_count = state.get("model_call_count", 0) 
    return {"model_call_count": current_count + 1}
```


### wrap-style装饰器

Wrap-style hooks: 环绕model或tool调用的中间件，包括:    
- wrap_model_call
- wrap_tool_call
例如，我们来定义一个可以在调用模型时失败重试的中间件，最大重试次数为3次。那就可以使用`@wrap_model_call`，在模型调用时做出判断，如果失败则重试，重试此时超过3次则结束。

  
同样的，被`@wrap_model_call`装饰的函数，其参数和返回值必须严格按照下面的格式：


```Python
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from typing import Any, Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request) # 这一步就是调用模型
        except Exception as e:
            print(f"Retry {attempt + 1}/3 after error: {e}")
            if attempt == 2:
                raise
    return None


agent = create_agent(
    model="deepseek-chat",
    middleware=[retry_model]
)
```

对于需要同时用到多个hooks的更复杂的中间件逻辑，我们还可以使用自定义类继承AgentMiddleware的方式来创建中间件。

例如，一个用来记录日志的中间件，要在模型调用、工具调用前后记录日志：

```Python
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ToolCallRequest
from langgraph.types import Command


class LoggingMiddleware(AgentMiddleware):

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        try:
            print(f"\n=======About to call model with {len(request.messages)} messages=======")
            return handler(request)
        except Exception as e:
            print(f"\n=======[错误]: {str(e)}=======")
            return AIMessage("调用模型失败，请重试~")

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"\n=======调用工具: {request.tool_call['name']}=======")
        print(f"\n=======参数: {request.tool_call['args']}=======")
        try:
            result = handler(request)
            print("\n=======工具调用成功！=======")
            return result
        except Exception as e:
            print(f"\n=======工具调用失败: {e}=======")
            raise


@tool
def get_weather(location: str):
    """查询指定城市的天气信息"""
    return f"Current weather in {location} is sunny, 25℃."


agent = create_agent(
    model="deepseek-chat",
    middleware=[LoggingMiddleware()],
    tools=[get_weather],
)

for chunk, metadata in agent.stream(
    {"messages": [HumanMessage("杭州今天天气如何？")]},
    stream_mode="messages"
):
    if chunk and chunk.content:
        print(chunk.content, end="", flush=True)
```



中间件除了利用hook做基本的信息记录和判断，还可以有一些高级的用法，例如：

- 动态修改请求：可以拦截发送给模型的请求，动态修改请求中使用的模型、工具、提示词等
    
- 条件跳转：在满足条件的情况下直接跳转到某个Agent执行的节点





































