from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="替换为你的DeepSeek API密钥",  # 替换为你的DeepSeek API密钥
    base_url="https://api.deepseek.com"
)

# 构建系统提示和用户提示
system_prompt = """你是一位顶级的Windows操作系统API专家和网络安全研究员，对勒索软件的行为模式有深入的理解。你的知识库包含了大量关于API功能、API间关系以及它们在恶意软件中潜在用途的信息。"""

user_prompt = """## 任务 ## 
你的任务是根据你内部存储的关于Windows API的知识，并结合勒索软件的典型行为特征，生成一系列示例性的知识图谱三元组。这些三元组应反映API间的不同类型的关联，特别关注那些与勒索软件活动相关的方面。 

## 输出三元组格式 ## 
请严格按照以下格式输出三元组： `(API_名称_1, 关系类型, API_名称_2)` 

其中，"关系类型"必须是以下三者之一： 
* `Link`: 表示 `API_名称_1` 的文档、功能或实现通常会引用、链接到或直接依赖于 `API_名称_2`。这反映了API间的程序执行逻辑或控制依赖。 
* `Sim`: 表示 `API_名称_1` 和 `API_名称_2` 在功能、行为模式或设计目标上具有显著的语义相似性。 
* `Risk`: 表示 `API_名称_1` 和 `API_名称_2` 的组合使用、顺序调用或关联行为，在勒索软件的上下文中构成一个已知的风险点、可疑活动或恶意行为路径的一部分。 

## 指令 ## 
请根据以下三种构图策略，从你的知识库中生成**多样化的示例三元组**。确保每个三元组都符合上述格式和关系类型定义。 

### 策略1：超链接构图策略 (使用 `Link` 关系) ###
* 生成一些三元组，其中 `API_名称_1` 的官方描述或常见用法会直接指向或依赖于 `API_名称_2`。

### 策略2：语义相似度构图策略 (使用 `Sim` 关系) ###
* 生成一些三元组，其中 `API_名称_1` 和 `API_名称_2` 执行相似的核心功能。

### 策略3：危险度构图策略 (使用 `Risk` 关系) ###
* 生成一些三元组，其中 `API_名称_1` 和 `API_名称_2` 的关联使用是已知的勒索软件TTP的一部分。

## 输出要求 ##
* 请为每种关系类型 (`Link`, `Sim`, `Risk`) 提供至少3-5个不同的示例三元组。
* 确保所有API名称都是Windows中真实存在的。
* 直接输出三元组列表，每行一个三元组。
* 解释说明性文字一概不要，仅输出纯粹的三元组，输出三元组，直到输出上限，最少输出500个三元组。
"""

# 调用API
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    temperature=0.7,
    max_tokens=8192,
    stream=False
)

# 打印完整响应
# print("完整API响应:")
# print(response)

# 提取并打印生成的文本
generated_text = response.choices[0].message.content
print("\n生成的三元组:")
print(generated_text)