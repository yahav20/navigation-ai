import os 
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_core.messages import HumanMessage, ToolMessage 
from tools.tools import (
    fetch_flights, 
    fetch_hotels, 
    calculate_trip_cost,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather
) 

load_dotenv()   
 
# 1. Initialize the Model (Gemini 2.5 Flash - 2026 Standard) 
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 

# 2. Tool Binding 
tools = [
    fetch_flights, 
    fetch_hotels, 
    calculate_trip_cost,
    fetch_activities,
    get_best_time_to_visit,
    get_average_weather
]
llm_with_tools = llm.bind_tools(tools) 
 
# 3. Step A: Intent Detection 
query = "I want to fly from London to Paris and find a hotel under 200 dollars" 
ai_msg = llm_with_tools.invoke(query) 
# 4. Step B: Manual Execution (The Manual Loop) 
if ai_msg.tool_calls: 
    print(f"Model identified {len(ai_msg.tool_calls)} tool(s) to call.\n")    
    # Mapping tool names to actual functions 
    tool_map = { 
        "fetch_flights": fetch_flights, 
        "fetch_hotels": fetch_hotels, 
        "calculate_trip_cost": calculate_trip_cost,
        "fetch_activities": fetch_activities,
        "get_best_time_to_visit": get_best_time_to_visit,
        "get_average_weather": get_average_weather
    }

    messages = [
            HumanMessage(content=query),
            ai_msg
        ]
    for tool_call in ai_msg.tool_calls:
        print(f"Executing: {tool_call['name']}...") 
        selected_tool = tool_map[tool_call["name"]] 
        
        tool_output = selected_tool.invoke(tool_call["args"])
        print(f"Tool Output: {tool_output}\n") 
        
        messages.append(
            ToolMessage(
                content=str(tool_output), 
                tool_call_id=tool_call["id"]
            )
        )
    
    # 5. Step C: Closing the Loop 
    final_response = llm_with_tools.invoke(messages)
    
    if isinstance(final_response.content, list) and final_response.content[0]['text']:
        clean_text = final_response.content[0].get('text', '')
        print(f"\nFinal Agent Response: {clean_text}")
    else:
        print(f"\nFinal Agent Response: {final_response.content}")
