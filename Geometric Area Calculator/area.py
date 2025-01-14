import math
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

def calculate_circle_area(radius):
    """Calculates the area of a circle."""
    return math.pi * radius**2

def calculate_triangle_area(base, height):
    """Calculates the area of a triangle."""
    return 0.5 * base * height

circle_tool = Tool(
    name="calculate_circle_area",
    func=calculate_circle_area,
    description="useful for calculating the area of a circle given its radius",
)

triangle_tool = Tool(
    name="calculate_triangle_area",
    func=calculate_triangle_area,
    description="useful for calculating the area of a triangle given its base and height",
)

tools = [circle_tool, triangle_tool]

llm = OpenAI(temperature=0)
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Prompt 1: Circle
prompt1 = "What is the area of a circle with a radius of 5?"
result1 = agent.run(prompt1)
print(f"Prompt 1 Result: {result1}")

# Prompt 2: Triangle
prompt2 = "What is the area of a triangle with a base of 4 and a height of 6?"
result2 = agent.run(prompt2)
print(f"Prompt 2 Result: {result2}")
