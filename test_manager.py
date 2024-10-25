from textwrap import dedent
from phi.agent import Agent
from phi.tools.website import WebsiteTools
from phi.model.openai import OpenAIChat
from phi.playground import Playground, serve_playground_app


analyzer = Agent(
    name="Analyzer",
    role="Analyzes URL content and identifies testable components",
    description=dedent(
        """\
    You are a senior QA analyst specializing in web application testing. Given a URL,
    extract and analyze its content to identify all testable components, features,
    and user interactions.
    """
    ),
    model=OpenAIChat(id="gpt-4o"),
    tools=[WebsiteTools()],
    instructions=[
        "Given a URL, use `extract_url_content` to analyze the webpage content and structure.",
        "Identify all interactive elements, forms, navigation paths, and key functionalities.",
        "Map out the main user flows and critical paths.",
        "List all input fields and their validation requirements.",
        "Remember: thoroughness is crucial for comprehensive test coverage.",
    ],
    add_datetime_to_instructions=True,
)

test_writer = Agent(
    name="TestWriter",
    role="Creates detailed test cases and scenarios",
    description=dedent(
        """\
    You are a test automation expert. Given webpage analysis and component list,
    your goal is to create comprehensive test cases covering functionality, edge cases,
    and error scenarios.
    """
    ),
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Create detailed test cases for each identified component and user flow.",
        "Include positive tests, negative tests, and boundary conditions.",
        "Specify pre-conditions, steps, expected results, and acceptance criteria.",
        "Ensure test cases cover cross-browser compatibility and responsive design.",
        "Focus on both functional and non-functional testing requirements.",
        "Structure test cases in a clear, maintainable format.",
        "Include edge cases and error handling scenarios.",
        "Prioritize test cases based on critical functionality.",
    ],
    add_datetime_to_instructions=True,
    add_chat_history_to_prompt=True,
    num_history_messages=3,
)

test_manager = Agent(
    name="TestManager",
    team=[analyzer, test_writer],
    description="You are a QA manager responsible for ensuring comprehensive test coverage for web applications.",
    model=OpenAIChat(id="gpt-4o"),
    instructions=[
        "Direct the analyzer to extract and analyze the webpage content.",
        "Pass the analysis to the test writer for test case creation.",
        "Review and organize test cases for maximum coverage and efficiency.",
        "Ensure test cases follow best practices and industry standards.",
        "Group test cases by functionality and priority.",
        "Include test execution prerequisites and environment requirements.",
        "Add traceability between requirements and test cases.",
        "Ensure comprehensive coverage of all critical paths.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

app = Playground(agents=[test_manager]).get_app()

if __name__ == "__main__":
    serve_playground_app("test_manager:app", reload=True)

# Example usage:
#test_manager.print_response(
#     "create test cases for this URL: https://www.google.de/",
#     stream=True
#)
