import contextlib
from typing import List, Optional
import random
import openai
import subprocess
import time


class Skill:
    def __init__(self, skill_id, task, chat_completion):
        self.skill_id = skill_id
        self.task = task
        self.chat_completion = chat_completion
        self.code_instances = []  # List to store every instance of generated code
        self.success_rate = 0
        self.use_count = 0
        self.dependencies: List[str] = []
        self.sequences: List[List[str]] = []

    def generate_code(self, task):
        prompt = f"Generate code for the task: {task}"
        completion = openai.ChatCompletion.create(

            engine="gpt-3.5-turbo", 

            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1000,
            n=1,
            stop=None,
            temperature=0.7
        )
        generated_code = completion.choices[0].message.content.strip()
        self.code_instances.append(generated_code)  # Save each instance of generated code
        print(generated_code)
        return generated_code

    def evolve(self):
        self.generate_code(self.task)

    def use(self):
        self.use_count += 1
        # Execute the code here and update success_rate based on the result
        result = self.execute_code()
        self.update_success_rate(result)

    def execute_code(self):
        # Execute the code in a subprocess and capture the result
        result = subprocess.run(['python', '-c', self.code_instances[-1]], capture_output=True, text=True)
        print(result.stdout)
        return result.stdout

    def update_success_rate(self, result):
        # Update the success_rate based on the result of code execution
        # Update the success_rate logic based on your specific requirements
        if "success" in result:
            self.success_rate += 1

    def add_to_sequence(self, sequence: List[str]):
        self.sequences.append(sequence)


class EvolvingProgram:
    def __init__(self, chat_completion):
        self.skills = {}
        self.chat_completion = chat_completion
        self.current_task = None

    def add_skill(self, task, dependencies=None):
        skill_id = str(len(self.skills) + 1)
        skill = Skill(skill_id, task, self.chat_completion)
        if dependencies:
            skill.dependencies = dependencies
        self.skills[skill_id] = skill
        print(f"Added skill ({skill_id}): {task}")
        return skill_id

    def evolve_all_skills(self):
        for skill in self.skills.values():
            skill.evolve()
            print(f"Evolved skill ({skill.skill_id}): {skill.task}")

    def use_skill(self, skill_id):
        if skill := self.skills.get(skill_id):
            for dependency_id in skill.dependencies:
                self.use_skill(dependency_id)
            skill.use()
            print(f"Used skill ({skill.skill_id}): {skill.task}")

    def generate_next_task(self):
        # Implement a method like Goal Generative Adversarial Network (Goal GAN) to generate the next task
        # The task should be slightly more difficult than the current task
        self.current_task = self.chat_completion.create("Generate next task")[0].choices[0].message.content.strip()

    def evaluate_skill(self, skill):
        # Implement a method to evaluate the skill based on its success rate and use_count
        return skill.success_rate / skill.use_count if skill.use_count else 0

    def neuron_decision(self):
        # Sort skills based on priority (e.g., success rate, use count, etc.)
        sorted_skills = sorted(self.skills.values(), key=lambda skill: (skill.success_rate, skill.use_count), reverse=True)

        for skill in sorted_skills:
            self.use_skill(skill.skill_id)

            # Check the success rate of the skill
            success_rate = self.evaluate_skill(skill)
            if success_rate < 1.0:
                # If the success rate is less than 100%, break the loop and stop executing further skills
                break

    def save_state(self):
        with open('storage/program_state.txt', 'w') as state_file:
            state_file.write(self.current_task + '\n')
            for skill_id, skill in self.skills.items():
                state_file.write(skill_id + '\n')
                state_file.write(skill.task + '\n')
                state_file.write(str(skill.success_rate) + '\n')
                state_file.write(str(skill.use_count) + '\n')
                state_file.write(','.join(skill.dependencies) + '\n')
                state_file.write(';'.join([','.join(seq) for seq in skill.sequences]) + '\n')

    def load_state(self):
        with contextlib.suppress(FileNotFoundError):
            with open('storage/program_state.txt', 'r') as state_file:
                lines = state_file.readlines()
                self.current_task = lines[0].strip()
                for i in range(1, len(lines), 7):
                    skill_id = lines[i].strip()
                    task = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    if i + 2 < len(lines):
                        success_rate = int(lines[i + 2].strip()) if lines[i + 2].strip().isdigit() else 0
                    else:
                        success_rate = 0
                    if i + 3 < len(lines):
                        use_count = int(lines[i + 3].strip()) if lines[i + 3].strip().isdigit() else 0
                    else:
                        use_count = 0
                    if i + 4 < len(lines):
                        dependencies = lines[i + 4].strip().split(',') if lines[i + 4].strip() else []
                    else:
                        dependencies = []
                    if i + 5 < len(lines):
                        sequences = [seq.split(',') for seq in lines[i + 5].strip().split(';')] if lines[i + 5].strip() else []
                    else:
                        sequences = []
                    if i + 6 < len(lines):
                        code_instances = lines[i + 6].strip().split(';') if lines[i + 6].strip() else []
                    else:
                        code_instances = []
                    skill = Skill(skill_id, task, self.chat_completion)
                    skill.success_rate = success_rate
                    skill.use_count = use_count
                    skill.dependencies = dependencies
                    skill.sequences = sequences
                    skill.code_instances = code_instances
                    self.skills[skill_id] = skill



    def execute(self):
        self.load_state()
        self.evolve_all_skills()
        self.neuron_decision()
        self.save_state()

        for skill in self.skills.values():
            print(f"Skill ({skill.skill_id}): {skill.task}")
            for code_instance in skill.code_instances:
                print(f"Code Instance: {code_instance}")
                result = skill.execute_code()
                print(f"Result: {result}")
                skill.update_success_rate(result)
                print(f"Success Rate: {skill.success_rate}")
                print("---")

            response = self.chat_completion.create(
                engine="gpt-3.5-turbo",
                messages=[{"role": "system", "content": skill.task}]
            )
            description = response.choices[0].message.content.strip()
            attributes = self.chat_completion.create(f"Describe attributes for skill: {skill.task}", engine="gpt-3.5-turbo")[0].choices[0].message.content.strip()
            print(f"Skill ({skill.skill_id}): {skill.task}")
            print(f"Description: {description}")
            print(f"Attributes: {attributes}")
            print(f"Success Rate: {skill.success_rate}")
            print("---")
            time.sleep(1)

        self.generate_next_task()
        print(f"Next task: {self.current_task}")


openai.api_key = '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'

program = EvolvingProgram(openai.ChatCompletion(engine="gpt-3.5-turbo"))

# Add skills
skill_1_id = program.add_skill('Write a Python function to calculate the Fibonacci sequence')
skill_2_id = program.add_skill('Implement sorting algorithm', dependencies=[skill_1_id])
skill_3_id = program.add_skill('Build a web scraping tool', dependencies=[skill_2_id])
skill_4_id = program.add_skill('Create a machine learning model', dependencies=[skill_2_id])

# Create skill sequences
program.skills[skill_2_id].add_to_sequence([skill_3_id, skill_4_id])

# Execute program
program.execute()
