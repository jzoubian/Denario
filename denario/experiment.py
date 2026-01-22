import re
from pathlib import Path

from .key_manager import KeyManager
from .prompts.experiment import experiment_planner_prompt, experiment_engineer_prompt, experiment_researcher_prompt
from .utils import create_work_dir, get_task_result

class Experiment:
    """
    This class is used to perform the experiment.
    TODO: improve docstring
    """

    def __init__(self,
                 research_idea: str,
                 methodology: str,
                 keys: KeyManager,
                 work_dir: str | Path,
                 involved_agents: list[str] = ['engineer', 'researcher'],
                 engineer_model: str = "qwen3-coder:30b",  # Changed from gpt-4.1 for Ollama
                 researcher_model: str = "qwen3:30b",  # Changed from o3-mini-2025-01-31 for Ollama
                 planner_model: str = "qwen3:30b",  # Changed from gpt-4o for Ollama
                 plan_reviewer_model: str = "qwen3:30b",  # Changed from o3-mini for Ollama
                 restart_at_step: int = -1,
                 hardware_constraints: str | None = None,
                 max_n_attempts: int = 10,
                 max_n_steps: int = 6,
                 orchestration_model = "qwen3:30b",  # Changed from gpt-4.1 for Ollama
                 formatter_model = "qwen3:30b",  # Changed from o3-mini for Ollama
                ):
        
        self.engineer_model = engineer_model
        self.researcher_model = researcher_model
        self.planner_model = planner_model
        self.plan_reviewer_model = plan_reviewer_model
        self.restart_at_step = restart_at_step
        if hardware_constraints is None:
            hardware_constraints = ""
        self.hardware_constraints = hardware_constraints
        self.max_n_attempts = max_n_attempts
        self.max_n_steps = max_n_steps
        self.orchestration_model = orchestration_model
        self.formatter_model = formatter_model

        self.api_keys = keys

        self.experiment_dir = create_work_dir(work_dir, "experiment")

        involved_agents_str = ', '.join(involved_agents)

        # Set prompts
        self.planner_append_instructions = experiment_planner_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
            involved_agents_str = involved_agents_str
        )
        self.engineer_append_instructions = experiment_engineer_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )
        self.researcher_append_instructions = experiment_researcher_prompt.format(
            research_idea = research_idea,
            methodology = methodology,
        )

    def run_experiment(self, data_description: str, **kwargs):
        """
        Run the experiment.
        TODO: improve docstring
        """

        print(f"Engineer model: {self.engineer_model}")
        print(f"Researcher model: {self.researcher_model}")
        print(f"Planner model: {self.planner_model}")
        print(f"Plan reviewer model: {self.plan_reviewer_model}")
        print(f"Max n attempts: {self.max_n_attempts}")
        print(f"Max n steps: {self.max_n_steps}")
        print(f"Restart at step: {self.restart_at_step}")
        print(f"Hardware constraints: {self.hardware_constraints}")

        # Import cmbagent only when needed
        import cmbagent

        try:
            results = cmbagent.planning_and_control_context_carryover(data_description,
                                n_plan_reviews = 1,
                                max_n_attempts = self.max_n_attempts,
                                max_plan_steps = self.max_n_steps,
                                max_rounds_control = 500,
                                engineer_model = self.engineer_model,
                                researcher_model = self.researcher_model,
                                planner_model = self.planner_model,
                                plan_reviewer_model = self.plan_reviewer_model,
                                plan_instructions=self.planner_append_instructions,
                                researcher_instructions=self.researcher_append_instructions,
                                engineer_instructions=self.engineer_append_instructions,
                                work_dir = self.experiment_dir,
                                api_keys = self.api_keys,
                                restart_at_step = self.restart_at_step,
                                hardware_constraints = self.hardware_constraints,
                                default_llm_model = self.orchestration_model,
                                default_formatter_model = self.formatter_model
                                )
        except Exception as e:
            print("\n" + "="*80)
            print("⚠️  CMBAGENT EXECUTION FAILED")
            print("="*80)
            print(f"\nError: {e}")
            print(f"\nThe code execution failed after {self.max_n_attempts} attempts.")
            print(f"Generated code is in: {self.experiment_dir}")
            print("\nYou have the following options:")
            print("1. Check the generated code and fix it manually")
            print("2. Skip this step and provide results manually")
            print("3. Retry with different settings")
            print("4. Abort the workflow")
            
            user_choice = input("\nEnter your choice (1/2/3/4) [default: 2]: ").strip() or "2"
            
            if user_choice == "1":
                print("\n📝 Please fix the code in the experiment directory and press Enter when done...")
                input("Press Enter to continue...")
                print("⚠️  Note: Automatic re-execution is not implemented. Please run the code manually.")
                print("Then provide the results manually (option 2).")
                user_choice = "2"
            
            if user_choice == "2":
                print("\n📝 Please enter the results manually.")
                print("You can provide a summary of your findings in markdown format.")
                print("Type 'END' on a new line when finished:\n")
                
                manual_results = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    manual_results.append(line)
                
                self.results = "\n".join(manual_results)
                self.plot_paths = []
                print("\n✅ Manual results saved.")
                return None
            
            elif user_choice == "3":
                print("\n💡 To retry with different settings:")
                print("1. Modify the input files (data_description.md, methods.md)")
                print("2. Adjust max_n_attempts or hardware_constraints")
                print("3. Run the script again")
                raise e
            
            else:  # choice == "4" or anything else
                print("\n❌ Aborting workflow.")
                raise e
        
        chat_history = results['chat_history']
        final_context = results['final_context']
        
        try:
            task_result = get_task_result(chat_history,'researcher_response_formatter')
        except Exception as e:
            raise e
            
        MD_CODE_BLOCK_PATTERN = r"```[ \t]*(?:markdown)[ \t]*\r?\n(.*)\r?\n[ \t]*```"
        matches = re.findall(MD_CODE_BLOCK_PATTERN, task_result, flags=re.DOTALL)
        if matches:
            extracted_results = matches[0]
        else:
            print("Warning: No markdown code block found in results, using full task_result")
            extracted_results = task_result
        
        clean_results = re.sub(r'^<!--.*?-->\s*\n', '', extracted_results)
        self.results = clean_results
        self.plot_paths = final_context['displayed_images']

        return None


